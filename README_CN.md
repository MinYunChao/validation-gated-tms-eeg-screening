# Validation-Gated TMS-EEG Screening

[English](README.md)

[项目主页](https://minyunchao.github.io/validation-gated-tms-eeg-screening/) |
[匿名论文](paper/validation-gated-tms-eeg-screening-anonymized.pdf)

这是一个面向严格受试者互斥评价的 TMS-EEG 抑郁筛查研究代码库，包含
Clean STGNN 波形主干、决策层 Riemannian residual、validation-gated
非诊断 metadata、受试者级指标和合成数据测试。

> **仅限研究用途。** 本仓库不是医疗器械，不提供抑郁症确诊、治疗建议或
> 临床部署依据。模型阳性输出仍需标准临床评估。

## 开源内容

- 不含作者、单位、基金和逐受试者信息的六页匿名论文；
- 不使用受试者 ID 归一化的 Clean STGNN；
- 带泄漏检查的受试者互斥 train/validation/test 划分；
- 只在决策层加入的 log-covariance residual；
- 由 validation subjects 选择规则的非诊断 metadata 分支；
- epoch-to-subject 概率聚合、临床指标和配对 bootstrap；
- 不包含真实记录的合成 TMS-EEG smoke test；
- 不含逐受试者信息的论文聚合结果。

## 不开源内容

- 真实 TMS-EEG 波形、预处理张量和临床表格；
- 受试者标识、真实数据划分、逐受试者预测和治疗相关变量；
- 原论文 checkpoint、优化器状态和 W&B 文件；
- 内部探索脚本、日志、失败实验缓存和机器绝对路径。

数据与权重并非遗漏的 Git LFS 文件。其限制原因和未来申请边界见
[数据说明](docs/DATA.md)、[权重说明](docs/WEIGHTS.md)及
[论文与代码映射](docs/MANUSCRIPT_MAPPING.md)。

## 目录结构

```text
.
|-- configs/                 # 论文尺度和快速测试配置
|-- docs/                    # 数据、权重和复现范围说明
|-- reports/                 # 非识别性论文聚合结果
|-- scripts/                 # smoke test 和发布审计
|-- src/tms_eeg_screening/   # 可安装 Python 包
|-- tests/                   # 单元和端到端测试
|-- MODEL_CARD.md
|-- LICENSE
`-- pyproject.toml
```

## 安装

需要 Python 3.10 或更高版本，GPU 为可选项。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 快速测试

```bash
tms-eeg make-synthetic --output data/synthetic_demo.npz --subjects 30
tms-eeg inspect-data --data data/synthetic_demo.npz
tms-eeg train \
  --data data/synthetic_demo.npz \
  --config configs/smoke.yaml \
  --output outputs/smoke
```

也可直接运行：

```bash
bash scripts/run_smoke_test.sh
```

合成数据不包含任何真实波形、患者或人口学信息。其结果只验证软件能够运行，
不得与论文结果比较，也不能用于科学或临床有效性主张。

## 使用获得授权的数据

将去标识后的数据整理为 `docs/DATA.md` 定义的 `.npz` 格式：

```bash
tms-eeg inspect-data --data /path/to/authorized_export.npz
tms-eeg train \
  --data /path/to/authorized_export.npz \
  --config configs/paper.yaml \
  --output outputs/paper_protocol
```

必需数组为：

| 数组 | 形状 | 含义 |
| --- | --- | --- |
| `waveforms` | `[epochs, channels, samples]` | 预处理 TMS-EEG epoch |
| `labels` | `[epochs]` | 0=control，1=MDD |
| `subject_ids` | `[epochs]` | 仅用于分组的稳定标识 |

可选数组包括 `aligned_waveforms`、`metadata`、`metadata_subject_ids` 和
`metadata_names`。metadata 只应包含 age、sex、motor threshold 和
stimulation intensity 等基线非诊断变量，不得包含 HAMD、治疗组、疗效、
治疗后结局或直接身份信息。

如需冻结受试者划分，可提供：

```json
{
  "train": ["S001", "S002"],
  "validation": ["S003", "S004"],
  "test": ["S005", "S006"]
}
```

```bash
tms-eeg train --data export.npz --config configs/paper.yaml \
  --split split.json --output outputs/fold01
```

程序会拒绝受试者重叠、遗漏、重复分配、标签不一致及某个集合缺少一类样本。
归一化、模型选择、residual 强度和 gating policy 均不得使用测试集拟合。

## 输出文件及公开边界

| 文件 | 内容 | 是否建议提交 Git |
| --- | --- | --- |
| `metrics.json` | 聚合指标 | 隐私复核后再决定 |
| `history.json` | 训练和验证历史 | 可选 |
| `resolved_config.json` | 完整配置 | 路径检查后可提交 |
| `split.json` | 受试者划分 | **真实数据不得提交** |
| `subject_predictions.csv` | 个体预测 | **不得提交** |
| `best_model.pt` | 本地 checkpoint | **不得提交** |
| `auxiliary_models.joblib` | 辅助分支模型 | **不得提交** |

`.gitignore` 已排除数据、输出、模型、临床表格和实验缓存。公开前必须运行：

```bash
bash scripts/release_audit.sh
```

## 方法边界

1. Clean 主干将波形划分为非重叠时间 patch；
2. 动态相似度图在多尺度 channel-time 节点上执行消息传递；
3. epoch 概率按受试者取均值；
4. 稳定化 log-covariance 只形成独立 residual，不改写波形表示；
5. 低容量 metadata 模型仅在 validation 选择的规则允许时改变决策；
6. 所有规则冻结后才评价 held-out test。

本仓库是依据论文方法原则整理的独立、可运行公开实现，不是内部探索代码的
逐字复制。准确的复现层级见 `docs/REPRODUCIBILITY.md`，各论文模块与公开实现
的对应关系见 `docs/MANUSCRIPT_MAPPING.md`。

## 聚合论文结果

非识别性结果位于 `reports/paper_aggregate_results.json`。固定 18 人测试中，
residual 将 subject F1 从 0.6625 提高至 0.7662，但配对 bootstrap 区间跨越
0，且该提升未在五折 nested evaluation 中稳定复现。因此这些结果只属于
单队列内部研究证据，不构成临床性能保证。

## 数据和权重申请

目前不存在自动下载链接，也不承诺申请后必然获得数据或历史权重。只有在原始
知情同意、伦理审批和数据控制机构允许的情况下，才可建立正式申请流程。建立前
不应在 README 中填写未经确认的联系人，也不得通过 GitHub issue 私下传输文件。

## 验证

```bash
pytest
ruff check src tests
bash scripts/release_audit.sh
```

GitHub Actions 会在 CPU 环境执行单元测试、端到端 smoke test 和发布审计。

## 引用与许可

公开论文有意保持匿名。在出现稳定的公开书目信息前，请引用仓库 URL 和不可变
release tag。本仓库暂不提供 `CITATION.cff`，避免暴露作者身份或使用不完整的
出版信息。

本发布目录中的代码采用 MIT License。该许可不授予真实受试者数据、历史权重
或第三方数据集的访问权。贡献代码前请阅读 `CONTRIBUTING.md`。

## 主要局限

- 单队列且独立受试者数量有限；
- 固定划分的效应区间较宽；
- nested evaluation 仍不是外部或多中心验证；
- 模型容量相对样本量较大；
- 固定 operating point 的 specificity 有限；
- 尚未在冻结研究中比较近期预训练 EEG encoder；
- 外部校准、跨中心迁移和临床效用尚不明确。
