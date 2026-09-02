# 复现说明与使用指南

## 1. 开源边界

本仓库是纯代码开源包，只提供实验代码、配置、单元测试和详细复现步骤，不公开任何论文实验输出或中间结果。仓库中不包含原始样本、样本编号、扰动后样本、模型预测、逐样本正确性记录、指标文件、结果表、生成图、模型检查点或论文结果汇总。

复现者需要自行合法取得原始数据集和相应模型，并在本地生成实验输出。所有生成内容应保存到 Git 已忽略的目录，例如 `data/private/`、`data/manifests/`、`data/pvd/`、`predictions/` 和 `outputs/`。

## 2. 完整流程

实验复现分为六个阶段：

1. 合法获取并核验原始数据集；
2. 整理私有源数据清单；
3. 构造匹配的 clean、text-only、vision-only 和 joint 四种条件；
4. 使用固定提示词和确定性解码运行模型；
5. 将模型响应规范化为配对的长表；
6. 计算 JAS、按样本聚类的 Logit 交互检验、BH-FDR 校正值和 CDP 状态。

本仓库直接实现第 3--6 阶段。模型权重、商业 API 权限和受许可约束的数据集属于外部依赖，不包含在仓库中。

## 3. 软件环境

建议基础环境：

- Python 3.11；
- NumPy、pandas、SciPy，用于 CDP 统计分析；
- OpenCV，用于生成 PVD 图片；
- 各模型对应的推理依赖。

创建分析环境：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

安装 PVD 图像依赖：

```bash
python -m pip install -e ".[pvd]"
```

如果不同模型的依赖发生冲突，应为各模型建立独立环境。每次推理必须记录：

- 操作系统和 Python 版本；
- GPU、CUDA 和驱动版本；
- `torch`、`transformers` 及模型专用库版本；
- 模型仓库及不可变 revision/commit；
- 精度和设备映射；
- 图像预处理参数；
- 解码参数；
- 随机种子；
- 提示词文件的 SHA-256。

API 密钥不得写入代码或配置文件，应通过服务商规定的环境变量提供。

## 4. 获取原始数据

通过官方渠道获取 Hateful Memes 和 ToxiCN-MM。不要将原始图片或文本复制到本仓库。建议将获授权数据放在已被 Git 忽略的 `data/private/` 下：

```text
data/private/
  hateful_memes/
    images/
    source.csv
  toxicn_mm/
    images/
    source.csv
```

使用或分享数据前请阅读 `docs/DATA_AND_LICENSING.md`。

## 5. 准备私有源数据清单

建立 UTF-8 CSV，每个原始样本只占一行，必须包含：

| 字段 | 含义 |
|---|---|
| `sample_id` | 清单内唯一且稳定的样本编号 |
| `label` | 二分类金标准：`1` 为 harmful，`0` 为 non-harmful |
| `external_text` | 与图片配套的原始外部文本 |
| `image_path` | 绝对路径，或相对于 `--image-root` 的路径 |
| `language` | `english` 或 `chinese` |
| `tri_neutral_text` | 预先构造的中性语境替换文本 |
| `tri_positive_text` | 预先构造的风险导向替换文本 |

TRI 是对外部文本通道的替换，不是追加到原文后面。四种匹配条件必须共用同一金标准标签。TRI 模板不得直接包含答案标签或其明显同义词。

更完整的构造和校验规则见 `docs/DATA_CONSTRUCTION.md`。

## 6. 生成四条件清单和 PVD 图片

执行：

```bash
python scripts/build_condition_manifest.py \
  --input data/private/source_manifest.csv \
  --output data/manifests/conditions.csv \
  --pvd-dir data/pvd \
  --image-root data/private/images \
  --operators sp_text sp_image spread \
  --seed 2026 \
  --generate-pvd
```

对每个原始样本、PVD 类别和 TRI 类别，脚本生成四条严格匹配记录：

| 条件 | `T` | `V` | 图片 | 外部文本 |
|---|---:|---:|---|---|
| clean | 0 | 0 | 原图 | 原文 |
| text | 1 | 0 | 原图 | TRI 替换文本 |
| vision | 0 | 1 | PVD 图片 | 原文 |
| joint | 1 | 1 | PVD 图片 | TRI 替换文本 |

PVD 精确参数位于 `configs/pvd_parameters.yaml`。随机算子的种子由基础种子、样本编号和算子名称稳定派生，因此更换源清单的行顺序不会改变生成结果。

检查单张图片的 PVD：

```bash
python scripts/apply_pvd.py \
  --input data/private/images/example.jpg \
  --output data/pvd/example_spread.jpg \
  --operator spread \
  --seed 2026
```

## 7. 模型推理

中英文轨道统一使用 `configs/classification_prompts.json`。对清单中的每一行，必须使用该行的 `image_path` 与 `external_text` 独立推理。

推理时必须遵守：

- 同一匹配组的四种条件使用完全相同的模型 revision；
- 图像预处理和解码参数保持一致；
- 使用确定性解码，例如 `do_sample=false` 或服务支持的温度零；
- 只要求模型返回 `harmful` 或 `non-harmful`；
- 在本地保留未经修改的原始响应以便审计；
- 解析失败单独记录，不能默认归入某个类别；
- 按稳定行键断点续跑，覆盖旧结果时必须保留重跑记录。

开源模型应固定 Hugging Face revision。API 模型应记录服务商、模型标识、调用日期、区域及请求参数。即便温度为零，API 更新和底层非确定性仍可能导致响应无法逐字一致。

所有推理输出保存到 `predictions/`，该目录已被 Git 忽略。

先根据机器的 CUDA 环境安装匹配的 PyTorch，再安装模型推理依赖：

```bash
python -m pip install -r requirements-inference.txt
```

Qwen2.5-VL 示例：

```bash
python scripts/run_model_inference.py \
  --input data/manifests/conditions.csv \
  --output predictions/qwen25vl.csv \
  --backend qwen25vl \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --model-label Qwen2.5-VL-3B \
  --revision <固定的模型提交号> \
  --image-root data/private
```

InternVL2.5 使用 `--backend internvl25`，LLaVA-1.5 使用 `--backend llava15`，OpenAI 兼容多模态接口使用 `--backend openai-compatible`。API 密钥通过 `OPENAI_API_KEY` 提供，也可以用 `--api-key-env` 指定其他环境变量名。生成的 CSV 不得提交。

## 8. 规范化预测记录

CDP 的输入是 UTF-8 CSV，必须包含：

| 字段 | 含义 |
|---|---|
| `key` | 四种条件共享的稳定样本键 |
| `correct` | 解析预测与金标准一致时为 `1`，否则为 `0` |
| `T` | 是否应用文本扰动 |
| `V` | 是否应用视觉扰动 |
| `model` | 模型标识 |
| `language` | 语言轨道 |
| `pvd` | PVD 类别 |
| `tri` | TRI 类别 |

建议另外保留 `label`、`pred`、`condition`、`raw_output`、`parser_status`、`model_revision`、`seed` 和 `prompt_sha256`。分析代码会忽略额外字段。

分析前必须确认：每个 `(model, language, pvd, tri, key)` 在 `(T,V)` 的 `(0,0)`、`(1,0)`、`(0,1)`、`(1,1)` 四个单元中各出现且只出现一次。应先取四条件共同样本交集，不能让不同条件使用不同样本集合。

将一个或多个本地推理文件规范化：

```bash
python scripts/normalize_predictions.py \
  --input predictions/qwen25vl.csv predictions/internvl25.csv \
  --output predictions/aligned_correctness.csv
```

默认策略在发现缺失或重复条件时直接停止。如果实验开始前已经规定删除不完整匹配样本，可使用 `--incomplete-policy drop`，并同时用 `--exclusion-log predictions/exclusions.csv` 保存排除记录。

## 9. 在本地运行 CDP

```bash
python scripts/run_cdp.py \
  --input predictions/aligned_correctness.csv \
  --output-dir outputs \
  --floor 0.05 \
  --alpha 0.05
```

命令会把本地生成的分析表写入 `outputs/`。这些文件被 Git 忽略，不属于公开仓库内容。

每个分析行首先计算：

```text
JAS = min(A_text, A_vision) - A_joint
```

随后拟合二元交互模型：

```text
logit P(correct = 1) = beta_0 + beta_T T + beta_V V + beta_TV T*V
```

协方差使用以 `key` 为聚类单元的 sandwich 估计，从而处理同一原始样本对应四个重复条件的相关性。交互项 p 值在输入的分析网格内执行 Benjamini-Hochberg 校正。

CDP 按 floor-first 顺序判定：

1. 满足预设 floor 条件时记为 `floor-limited`；
2. 非 floor 情况下，只有 joint 同时低于两个单模态条件、JAS 为正、交互系数为负且校正值低于 `alpha` 时，才记为 `confirmed`；
3. 其余非 floor 行记为 `unsupported`。

查看结果后不得再改变 floor 阈值、FDR 校正范围、解析规则或样本交集。敏感性分析必须与预设主分析分开标注。

## 10. 质量核查清单

完成一次本地复现后，至少核查：

- 展开条件前，源样本编号唯一；
- 四种条件的标签保持不变；
- PVD 图片可读取，尺寸和通道符合预期；
- TRI 只做替换，且没有答案标签泄漏；
- 每个样本键都有四种条件；
- 同一网格内不存在重复 `(key,T,V)`；
- 同一比较内模型 revision 和提示词哈希一致；
- 解析器只接受声明的两个标签；
- 解析失败和图片缺失均有记录；
- FDR 校正在预定分析族内执行；
- 生成的输出均未被 Git 跟踪。

常用检查命令：

```bash
git status --short
python -m pytest -q
python scripts/run_cdp.py --help
python scripts/build_condition_manifest.py --help
```

## 11. 常见问题

### 缺少必需字段

按照第 8 节核对字段名。字段名区分大小写。

### 同一样本条件重复

分析前使用事先约定且可记录的稳定规则去重，不能依赖文件系统顺序任意保留一行。

### 四条件不完整

在每个分析行内取严格的四向样本交集。本地记录排除原因，不对缺失预测做插补。

### 完全分离

如果某个条件全部正确或全部错误，普通有限 Logit 估计可能不可用。应保留该行并记录失败，只能采用查看结果前已经声明的修正方法。

### 重跑预测不完全一致

逐项检查模型 revision、CUDA 与依赖版本、图像预处理、数值精度、提示词字节、解码参数、API 模型版本和解析规则。统计流程可复现与模型输出逐字一致是两个不同目标。

## 12. 分享本地复现结果

如果其他研究者确实需要你生成的结果，应在本 GitHub 仓库之外，依据原始数据许可证和所在机构的数据管理要求单独分享。分享前删除原始文本、图片路径、凭据和不受限样本标识。本公开仓库始终保持纯代码状态。
