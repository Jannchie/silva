# SILVA 最小版设计（v1-minimal）

**SILVA** = SigLIP-based Illustration Visual Aesthetic Scorer。
一个学习**你个人审美**的插画美学打分模型：训练时按序数回归建模，推理时输出连续分 `[1.0, 5.0]`。

本文档只描述**最小可跑通版本**。范围之外的内容（外部打分器融合、LoRA/全量微调、分布 head、部署服务）在最后「明确不做」一节列出，仅留扩展位、本版不实现。

---

## 1. 目标与非目标

**目标**：用你本人的 1~5 打分（约 9.6 万张，存于 pictoria SQLite），训练一个在 **SigLIP2-SO400M-384 预计算 embedding** 上的序数 head 美学打分器，验证它能学到你的审美偏好（以排序能力为主）。

**最小闭环**：
```
适配器导出 embedding manifest → 数据集 → 序数 head（在 embedding 上）→ ordinal loss（自动 pos_weight）→ 训练 → 验证集指标
```

**非目标（本版不做）**：外部 AI 打分器的融合、LoRA / 解冻 / 全量微调、5-bin 分布 head、ranking loss、Web 部署、数据标注工具。

---

## 2. 工程约定

- 包管理 **uv**；Python **3.12.6**；lint **ruff**（line-length 160，`select=ALL` + 项目既有 ignore 集）。
- 配置用 **Pydantic v2** 模型，从 YAML 加载。
- **训练库不含任何数据源知识**：只消费 manifest 契约（见 §3.1）；把外部源整形成 manifest 是脚本的事（见 §3.4）。训练库不依赖 `transformers`。
- 训练框架：**accelerate + 自定义训练循环**。

---

## 3. 数据管线

### 3.1 manifest 契约（`silva/data/manifest.py`）

**训练库只依赖 manifest 的形状，对 embedding 的来源一无所知**（哪个模型、什么分辨率都不管）。任何来源只要产出符合下表的列式 parquet 即可：

| 列               | 类型           | 必需 | 说明                          |
| ---------------- | -------------- | ---- | ----------------------------- |
| `embedding`      | list<float>[D] | 是   | 固定维度特征向量（v1：1152）  |
| `personal_score` | int (1..5)     | 是   | 你的打分                      |
| `split`          | str            | 是   | `train` / `val` / `test`      |
| `post_id`        | int            | 否   | 溯源 / split 去重 key         |

- `assign_splits(keys, ratios, seed)`：按任意 key 去重分配 split（默认 0.85/0.10/0.05），同 key 不跨 split。
- `build_manifest(post_ids, embeddings, scores, seed)`：把并列的 `(post_id, embedding, score)` 整形成已分配 split 的 df。
- `validate_manifest(df)`：**契约强制点**——校验必需列、`embedding` 非空且维度一致、`personal_score` 整数且 ∈[1,5]、`split` 合法。
- `write_manifest(df, path)`：先校验再写 parquet。Dataset 加载时也会跑 `validate_manifest`。

数据源适配是**脚本的事**，不是训练库的（见 §3.4）。

### 3.2 数据集（`silva/data/dataset.py`）

`AestheticDataset(manifest_path, split)`：
- 过滤指定 `split` 的行。
- 直接读 `embedding` 列，**无图片 / 无 SigLIP processor / 无 backbone**。
- 返回 `{"embedding": Tensor[D], "score": int}`。

### 3.3 分数取值与刻度选择（已确认）

本人打分为**纯整数 1/2/3/4/5**，语义上**不等距**：

| 分  | 含义                              |
| --- | --------------------------------- |
| 1   | 很差（辣眼睛、不希望看到）        |
| 2   | 不好（不希望看到，但没 1 那么差） |
| 3   | 还可以，但不够好                  |
| 4   | 好看                              |
| 5   | 非常好看                          |

关键结构：1、2 同属「不想看」、彼此接近；`2→3`（到「还行」）、`3→4`（到「好看」）是**质变门槛**。这正是 ordinal head 可学习阈值要表达的——档间距自适应，不强制等距。序数阈值目标直接用整数值，无需取整。

**否决纯线性回归（`1→0,…,5→1` + 回归）的理由**：该映射强制等距（断言「`2→3` 的差距 == `4→5` 的差距」），与上面的不等距语义冲突。ordinal 阈值能让 `thr₁`、`thr₂` 自行靠近来表达「1、2 接近」，而线性回归里 `0` vs `0.25` 是写死的等距。故保留 ordinal。同理去掉了原 loss 里的 SmoothL1 回归项（见 §5）。

### 3.4 数据源适配器（`scripts/export_manifest.py`）

把 pictoria SQLite 库整形成训练 manifest 的**脚本**（不属训练库）。读 `post_vectors_siglip2`（现成 SigLIP2 1152 维 embedding）JOIN `posts`（`score > 0`，约 9.6 万行），调 `build_manifest` / `write_manifest` 产出列式 parquet。读 vec0 表需 sqlite-vec：`uv sync --extra export`。换数据源只需另写一个调 `build_manifest` 的适配器；训练库不变。

---

## 4. 模型（`silva/models/`）

```
precomputed embedding[D]   （上游 SigLIP2-SO400M-384，由适配器产出）
  → LayerNorm + Dropout(0.1)
  → MLP trunk（hidden_dims，如 512→256，GELU）   ← 把 test Spearman 0.63 抬到 0.71
  → 个人 ordinal head
内部 ordinal 值：ordinal_score = 1 + Σ sigmoid(logit_k) ∈ [1, 5]   （标签/指标空间）
规范输出：       score = (Σ sigmoid(logit_k)) / 4 ∈ [0, 1]          （见 §4.3）
```

### 4.1 `ordinal_head.py` — `OrdinalHead`

- 一个 `Linear(hidden, 1)` 产生 latent score。
- 4 个**可学习单调阈值**：`base_threshold + cumsum(softplus(raw_deltas))`，保证 `thr_1 < thr_2 < thr_3 < thr_4`。
- `logits = latent - thresholds`（形状 `[B, 4]`）。

### 4.2 `aesthetic.py` — `EmbeddingAestheticModel`

- `embedding[D] → LayerNorm → Dropout → [MLP trunk] → OrdinalHead`，**无 backbone、无 `transformers` 依赖**。`hidden_dims=[]` 是线性探针;非空(如 `[512,256]`,GELU)插入 MLP —— 实测把 test Spearman 从 0.63 抬到 **0.71**(超过现成 waifu 分 0.70)。证明 SigLIP2 embedding 的美学信息**够、但非线性**,瓶颈在 head 容量而非 embedding。
- 因 v1 冻结 backbone，embedding 是固定的，预计算进 manifest 即可：训练库只学 head，每 epoch 不必重跑 SO400M，模型也能在无预训练权重下单测。
- `forward(embedding)` 返回 `{"logits", "score", "ordinal_score"}`。
- backbone 的来源/一致性由适配器与推理端负责，训练库不关心（见 §3.1）。**注意**：将来给新图打分时，必须用产出这批 embedding 的同一个 backbone，否则分布不一致。

### 4.3 输出规范（`[0,1]`，从 0 起算）

- **规范输出 `score ∈ [0,1]`**：序数模型里它 = 4 个阈值概率的均值 `Σsigmoid/4` = "清过了多少比例的质量门槛"。`0` = 连最低门槛都没过，`1` = 全过。这个量**天然 0-based**；`[1,5]` 里的下限"1"只是标签显示约定，不是模型的自然底。
- 预测值通常是小数（如 `0.73`），属正常。sigmoid 取不到精确 0/1，实际落在开区间 `(0,1)`。
- 缩放到任意刻度是一行 helper：`to_scale(score, lo, hi) = lo + (hi - lo) * score`（`to_scale(s,1,5)`、`to_scale(s,1,10)`）。消费方按自己习惯的刻度取用。
- 该 0~1 是按**你的口味**校准的归一分，**不是**通用客观美学。
- 训练目标 / ordinal head / 标签 / 指标仍在 `1~5` 标签空间（见 §5、§7）；`[0,1]` 只是输出层换算，不改动训练。

---

## 5. 损失（`silva/losses.py`）

```
L = ordinal_BCE(logits, ordinal_targets)      # 纯序数 BCE，无回归项
```

- `make_ordinal_targets(scores)`：`score∈{1..5}` → `[B,4]`，例 `5→[1,1,1,1]`、`3→[1,1,0,0]`、`1→[0,0,0,0]`。
- `ordinal_loss`：`binary_cross_entropy_with_logits`。
- **去掉 SmoothL1 回归项（原 `+ 0.2*SmoothL1(pred_score, personal_score)`）**：它把预测往等距整数标签拉，与 §3.3 的不等距语义冲突；且它主要改善 MAE/RMSE 这类**绝对误差**，而选模 / early-stop 按 **Spearman**（§6，排序优先于绝对误差），回归项对此无助益。`ordinal_score = 1+Σsigmoid` 的刻度仍由 BCE 钉住（拟合好时 sigmoid 饱和、自动逼近整数），去掉不丢刻度。
- **自动 pos_weight（类别不平衡）**：`compute_pos_weight(train_scores)` 从 **train split** 算每个阈值的 `#neg/#pos`，平衡"score>k"各自的正负失衡（你的分布里 `>1` 约 70:1、`>4` 约 1:4.9、`>3` 近 1:1）。`config.train.use_pos_weight` 开关（默认开）；建议先跑不加权 baseline 再 A/B。
- **pairwise ranking 项（`ranking_weight`）**：RankNet 式 logistic 排序损失，对每个 `score_i>score_j` 的对推 i 的连续分高于 j —— **直接优化选模指标 Spearman**（ordinal BCE 只间接优化）。在 MLP head 上再 +0.01 Spearman、top-5% 命中更高；`ranking_weight=0` 关闭。
- 预留 v2 多任务加权接口（外部 aux loss、`aux_weight`/`main_weight` 分歧加权），v1 路径不调用。

---

## 6. 训练（`silva/train.py`）

- accelerate 自定义循环；AdamW、**torch 原生 cosine+warmup**（`warmup_ratio=0.03`，无 transformers 依赖）、梯度裁剪。`mixed_precision` 可配（默认 `bf16`，CPU 测试用 `no`）。
- **只训 head**：backbone 不在训练库，embedding 已预计算。`lr_head ~ 3e-4`，3~10 epoch。head 极小，batch 可大（默认 256）。
- 启动时按 `use_pos_weight` 从 train split 算 `compute_pos_weight` 传入 loss（见 §5）。
- 每个 eval 周期在 val 上算全部指标；**按 Spearman 存最优 checkpoint 并 early-stop**（排序能力优先于绝对误差）。
- 配置：`configs/v1_stage1_head.yaml`。

---

## 7. 评估（`silva/metrics.py` + `silva/evaluate.py`）

只看你的验证集。指标：**MAE、RMSE、Pearson、Spearman、QWK、Top-5% precision**。
- MAE / RMSE 在 `1~5` 标签空间计算（"差几颗星"最可读）；需要时可换算到 `[0,1]`（×0.25）。
- Spearman / Pearson / QWK / Top-K **尺度无关**，`[0,1]` 还是 `[1,5]` 结果一致；这也是筛图最该看的指标。
`evaluate.py` 可对指定 checkpoint 在 val/test 上单独出报告。

---

## 8. 测试（`tests/`，纯函数 TDD）

GPU 不是必须（embedding 训练可在 CPU 跑），纯函数 + 端到端全部覆盖：
- `make_ordinal_targets` / `compute_pos_weight`：分数 → 阈值向量 / per-threshold 权重。
- `OrdinalHead` + `EmbeddingAestheticModel`：阈值单调递增；输出形状、`score∈[0,1]`、`ordinal_score∈[1,5]`。
- `silva_loss`：纯 ordinal BCE；`pos_weight` 行为。
- 指标实现：与已知小样本手算结果对拍（尤其 Spearman、QWK、Top-K）。
- manifest：契约校验（embedding 维度/score/split）、`build_manifest` 满足契约、split 按 `post_id` 不跨。
- **端到端 smoke**（`test_train_smoke.py`，**默认运行**）：toy embedding manifest 跑完整训练闭环，产出 checkpoint。

---

## 9. 项目结构

```
silva/
  pyproject.toml
  .python-version          # 3.12.6
  README.md
  configs/
    v1_stage1_head.yaml
  silva/
    config.py              # Pydantic 配置模型
    data/
      manifest.py          # 契约：schema + assign_splits + build_manifest + validate + write
      dataset.py           # 读 embedding（无图片/processor）
    models/
      ordinal_head.py
      aesthetic.py         # EmbeddingAestheticModel（无 backbone）
    losses.py
    metrics.py
    train.py
    evaluate.py
  scripts/
    export_manifest.py     # pictoria SQLite → manifest 适配器（uv sync --extra export）
  tests/
```

---

## 10. 完成定义（Definition of Done）

1. `export_manifest` 能从 pictoria SQLite 产出符合 §3.1 schema 的 embedding parquet。
2. 在 mock/小数据上，训练循环能跑完若干 epoch 且 loss 下降、产出 checkpoint。
3. `evaluate.py` 能加载 checkpoint 并在 val 上输出 §7 全部指标。
4. §8 所列纯函数单测全部通过。
5. `ruff check` 通过。

---

## 11. 明确不做（YAGNI，仅留扩展位）

- 外部 AI 打分器融合：校准（`calibration.py`）、辅助 head、分歧加权、weak ranking、residual head、v2~v4 ablation。
- LoRA / 解冻末 N 层（Stage2）、全量微调（Stage3）。
- 5-bin 分布 head、ranking loss。
- Web 部署 / 推理服务、数据清洗/标注工具。
- **NaFlex 变体对照**：v1 用 fixed-res 384（squish，与 SigLIP 预训练同分布，但插画长宽比会形变、细节受限于 384）。`siglip2-so400m-patch16-naflex` 原生支持长宽比 + 可变分辨率，是长宽比敏感场景的"正确"架构，但需换 model 类 / processor（带 `spatial_shapes`、`attention_mask`）/ 变长 collate。**决定：先出 fixed-res baseline 的 Spearman，再用验证集决定是否值得切 NaFlex，不预判。**

扩展位以参数/桩形式预留（多任务 loss 接口），但本版不实现逻辑。外部打分器（DB 的 `post_aesthetic_scores` siglip-v2-5 / `post_waifu_scores`）v1 不导出，需要时由适配器加列、训练库加 aux head。
