#!/usr/bin/env python3
"""Run one configured VLM backend over a private four-condition manifest.

Outputs are experimental records and must be written under the ignored
``predictions/`` directory or another non-public location.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Protocol

import pandas as pd

from bjd_cdp.predictions import parse_binary_response, stable_row_id


REQUIRED_COLUMNS = {
    "key",
    "label",
    "language",
    "pvd",
    "tri",
    "condition",
    "T",
    "V",
    "external_text",
    "image_path",
}


class Backend(Protocol):
    def predict(self, image_path: Path, prompt: str) -> str: ...


def load_prompts(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    for language in ("english", "chinese"):
        if not isinstance(config.get(language), str):
            raise ValueError(f"Prompt config lacks a string value for {language}")
    return config


def build_prompt(config: dict[str, object], language: str, external_text: object) -> str:
    language = str(language).lower().strip()
    if language not in {"english", "chinese"}:
        raise ValueError(f"Unsupported language: {language}")
    prefix = str(config[language]).strip()
    field = "External text" if language == "english" else "外部文本"
    return f"{prefix}\n\n{field}: {str(external_text)}"


class Qwen25VLBackend:
    def __init__(self, model_name: str, revision: str | None, max_new_tokens: int):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        kwargs = {"torch_dtype": "auto", "device_map": "auto"}
        if revision:
            kwargs["revision"] = revision
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(model_name, revision=revision)

    def predict(self, image_path: Path, prompt: str) -> str:
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[rendered],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        trimmed = [output[len(source) :] for source, output in zip(inputs["input_ids"], generated)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


class InternVL25Backend:
    def __init__(self, model_name: str, revision: str | None, max_new_tokens: int):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        kwargs = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "torch_dtype": self.dtype,
            "device_map": "auto",
        }
        if revision:
            kwargs["revision"] = revision
        self.model = AutoModel.from_pretrained(model_name, **kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            use_fast=False,
        )

    @staticmethod
    def _load_image(path: Path, input_size: int = 448, max_num: int = 12):
        import torch
        import torchvision.transforms as transforms
        from PIL import Image
        from torchvision.transforms.functional import InterpolationMode

        image = Image.open(path).convert("RGB")
        width, height = image.size
        aspect = width / height
        target_ratios = sorted(
            {
                (i, j)
                for n in range(1, max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if 1 <= i * j <= max_num
            },
            key=lambda ratio: ratio[0] * ratio[1],
        )
        best = min(
            target_ratios,
            key=lambda ratio: (
                abs(aspect - ratio[0] / ratio[1]),
                -int(width * height > 0.5 * input_size * input_size * ratio[0] * ratio[1]),
            ),
        )
        target_width, target_height = input_size * best[0], input_size * best[1]
        resized = image.resize((target_width, target_height))
        blocks = []
        grid_width = target_width // input_size
        for index in range(best[0] * best[1]):
            left = (index % grid_width) * input_size
            top = (index // grid_width) * input_size
            blocks.append(resized.crop((left, top, left + input_size, top + input_size)))
        if len(blocks) != 1:
            blocks.append(image.resize((input_size, input_size)))

        transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        return torch.stack([transform(block) for block in blocks])

    def predict(self, image_path: Path, prompt: str) -> str:
        pixels = self._load_image(image_path).to(self.dtype)
        pixels = pixels.to(next(self.model.parameters()).device)
        generation = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
        }
        with self.torch.inference_mode():
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixels,
                question=f"<image>\n{prompt}",
                generation_config=generation,
                history=None,
                return_history=False,
            )
        return str(response).strip()


class Llava15Backend:
    def __init__(self, model_name: str, revision: str | None, max_new_tokens: int):
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        kwargs = {"torch_dtype": "auto", "device_map": "auto"}
        if revision:
            kwargs["revision"] = revision
        self.model = LlavaForConditionalGeneration.from_pretrained(model_name, **kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(model_name, revision=revision)

    def predict(self, image_path: Path, prompt: str) -> str:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        rendered = f"USER: <image>\n{prompt}\nASSISTANT:"
        inputs = self.processor(images=image, text=rendered, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        return self.processor.decode(generated[0, input_length:], skip_special_tokens=True).strip()


class OpenAICompatibleBackend:
    def __init__(
        self,
        model_name: str,
        api_key_env: str,
        api_base: str | None,
        max_new_tokens: int,
    ):
        import os
        from openai import OpenAI

        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Environment variable {api_key_env} is not set")
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

    @staticmethod
    def _image_url(path: Path) -> str:
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{payload}"

    def predict(self, image_path: Path, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": self._image_url(image_path)}},
                    ],
                }
            ],
            temperature=0,
            max_tokens=self.max_new_tokens,
        )
        return str(response.choices[0].message.content).strip()


def make_backend(args: argparse.Namespace) -> Backend:
    if args.backend == "qwen25vl":
        return Qwen25VLBackend(args.model, args.revision, args.max_new_tokens)
    if args.backend == "internvl25":
        return InternVL25Backend(args.model, args.revision, args.max_new_tokens)
    if args.backend == "llava15":
        return Llava15Backend(args.model, args.revision, args.max_new_tokens)
    if args.backend == "openai-compatible":
        return OpenAICompatibleBackend(
            args.model,
            args.api_key_env,
            args.api_base,
            args.max_new_tokens,
        )
    raise AssertionError(args.backend)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Private condition-manifest CSV")
    parser.add_argument("--output", type=Path, required=True, help="Local prediction CSV")
    parser.add_argument(
        "--backend",
        choices=("qwen25vl", "internvl25", "llava15", "openai-compatible"),
        required=True,
    )
    parser.add_argument("--model", required=True, help="Local path, Hugging Face ID, or API model ID")
    parser.add_argument("--model-label", default=None, help="Stable label stored in the output")
    parser.add_argument("--revision", default=None, help="Pinned model revision or commit")
    parser.add_argument("--prompts", type=Path, default=Path("configs/classification_prompts.json"))
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=None, help="Debug only; omit for a full run")
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-base", default=None)
    args = parser.parse_args()

    if args.flush_every < 1:
        raise ValueError("--flush-every must be positive")
    source = pd.read_csv(args.input, dtype={"key": "string"})
    missing = sorted(REQUIRED_COLUMNS - set(source.columns))
    if missing:
        raise ValueError(f"Input manifest lacks required columns: {missing}")
    if args.max_rows is not None:
        source = source.head(args.max_rows).copy()

    prompt_config = load_prompts(args.prompts)
    model_label = args.model_label or args.model
    image_root = args.image_root or args.input.parent
    backend = make_backend(args)

    previous = pd.DataFrame()
    completed: set[str] = set()
    if args.resume and args.output.exists():
        previous = pd.read_csv(args.output, dtype={"key": "string", "row_id": "string"})
        completed = set(previous.get("row_id", pd.Series(dtype="string")).dropna().astype(str))

    pending: list[dict[str, object]] = []

    def flush() -> None:
        nonlocal previous, pending
        if not pending:
            return
        previous = pd.concat([previous, pd.DataFrame(pending)], ignore_index=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        previous.to_csv(args.output, index=False, encoding="utf-8")
        pending = []

    for row in source.to_dict("records"):
        row["model"] = model_label
        row_id = stable_row_id(row)
        if row_id in completed:
            continue
        prompt = build_prompt(prompt_config, str(row["language"]), row["external_text"])
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        image = Path(str(row["image_path"]))
        if not image.is_absolute():
            image = image_root / image

        raw_output = ""
        parser_status = "ok"
        pred: int | None = None
        error = ""
        try:
            if not image.is_file():
                raise FileNotFoundError(image)
            raw_output = backend.predict(image, prompt)
            pred = parse_binary_response(raw_output)
            if pred is None:
                parser_status = "invalid_label"
        except Exception as exc:  # local audit record; no output is committed
            if args.fail_fast:
                raise
            parser_status = "inference_error"
            error = f"{type(exc).__name__}: {exc}"

        pending.append(
            {
                **row,
                "row_id": row_id,
                "model_revision": args.revision or "unrecorded",
                "prompt_sha256": prompt_sha256,
                "raw_output": raw_output,
                "pred": pred,
                "parser_status": parser_status,
                "error": error,
            }
        )
        completed.add(row_id)
        if len(pending) >= args.flush_every:
            flush()
    flush()
    print(f"Local prediction records written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
