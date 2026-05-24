# SILVA 最小版设计（v1-minimal）

**SILVA** = SigLIP-based Illustration Visual Aesthetic Scorer。
一个学习**你个人审美**的插画美学打分模型：训练时按序数回归建模，推理时输出连续分 `[1.0, 5.0]`。

本文档只描述**最小可跑通版本**。范围之外的内容（外部打分器融合、LoRA/全量微调、分布 head、部署服务）在最后「明确不做」一节列出，仅留扩展位、本版不实现。

---

## 1. 目标与非目标

**目标**：用你本人的 1~5 打分（1万~10万张，存于 Postgres），训练一个 SigLIP2-SO400M-384 + 序数 head 的美学打分器，验证它能学到你的审美偏好（以排序能力为主）。

**最小闭环**：
```
导出 manifest → 数据集 → 冻结 backbone + 序数 head → ordinal loss → 训练 → 验证集指标
```

**非目标（本版不做）**：外部 AI 打分器的融合、LoRA / 解冻 / 全量微调、5-bin 分布 head、ranking loss、Web 部署、数据标注工具。

---

## 2. 工程约定

- 包管理 **uv**；Python **3.12.6**；lint **ruff**（line-length 160，`select=ALL` + 项目既有 ignore 集）。
- 配置用 **Pydantic v2** 模型，从 YAML 加载。
- 数据库连接用 `python-dotenv` 读 `.env` 的 `DATABASE_URL`，不硬编码。
- 训练框架：**accelerate + 自定义训练循环**。

---

## 3. 数据管线

### 3.1 manifest 导出（`scripts/export_manifest.py` → `silva/data/export_manifest.py`）

唯一接触数据库 schema 的地方。连 Postgres，导出 parquet：

| 列 | 类型 | 说明 |
|---|---|---|
| `image_path` | str | 本地绝对/相对图片路径 |
| `personal_score` | float | 你的 1~5 打分（见 §3.3 小数处理） |
| `split` | str | `train` / `val` / `test` |

- 表名/列名作为导出脚本的参数（CLI flag 或小配置），**待真实 schema 填入**；不阻塞其余开发——可先用 mock parquet 跑通。
- split：固定随机种子，**按 `image_path` 去重后划分**（默认 0.85 / 0.10 / 0.05），保证同图不跨 split。
- `scorer_a`/`scorer_b` 列若库中存在则一并导出存档（v2 用），v1 不读取。

### 3.2 数据集（`silva/data/dataset.py`）

`AestheticDataset(manifest_path, split, processor)`：
- 过滤指定 `split` 的行。
- 按 `image_path` 用 PIL 读图（`convert("RGB")`），损坏/缺失图记录并跳过。
- 用 SigLIP2 processor 预处理到 384。
- 返回 `{"pixel_values": Tensor, "score": float}`。

### 3.3 小数分数处理（**待确认**）

假设分数主要是整数 1~5。若存在小数（如 3.5）：
- **序数目标**用 `round(score)` 取整后生成阈值标签。
- **回归 SmoothL1** 用原始小数值。

这样小数信息不丢，又能用序数结构。若你确认全是整数，则两者一致。

---

## 4. 模型（`silva/models/`）

```
image
  → SigLIP2-SO400M-patch14-384 vision encoder (google/siglip2-so400m-patch14-384)
  → pooled feature
  → LayerNorm + Dropout(0.1)
  → 个人 ordinal head
推理：pred_score = 1 + Σ sigmoid(logit_k)   ∈ [1, 5]
```

### 4.1 `ordinal_head.py` — `OrdinalHead`

- 一个 `Linear(hidden, 1)` 产生 latent score。
- 4 个**可学习单调阈值**：`base_threshold + cumsum(softplus(raw_deltas))`，保证 `thr_1 < thr_2 < thr_3 < thr_4`。
- `logits = latent - thresholds`（形状 `[B, 4]`）。

### 4.2 `siglip_aesthetic.py` — `SigLIP2AestheticModel`

- 加载 `Siglip2VisionModel`（bf16，sdpa attention）。
- **v1：backbone 冻结**（`requires_grad=False`），只训 head。
- `forward(pixel_values)` 返回 `{"logits", "score"}`。
- 预留 `aux_heads`（外部打分器回归头）参数，v1 默认不构建。

---

## 5. 损失（`silva/losses.py`）

```
L = ordinal_BCE(logits, ordinal_targets) + 0.2 * SmoothL1(pred_score, personal_score)
```

- `make_ordinal_targets(scores)`：`score∈{1..5}` → `[B,4]`，例 `5→[1,1,1,1]`、`3→[1,1,0,0]`、`1→[0,0,0,0]`。
- `ordinal_loss`：`binary_cross_entropy_with_logits`。
- 预留 v2 多任务加权接口（外部 aux loss、`aux_weight`/`main_weight` 分歧加权），v1 路径不调用。

---

## 6. 训练（`silva/train.py`）

- accelerate 自定义循环；bf16、AdamW、cosine schedule、`warmup_ratio=0.03`、梯度裁剪。
- **仅 Stage1**：冻结 backbone，只训 head。`lr_head ~ 3e-4`，3~10 epoch。
- batch size + 梯度累积默认按单卡 24GB 配；多卡由 accelerate 自动起。
- 每个 eval 周期在 val 上算全部指标；**按 Spearman 存最优 checkpoint 并 early-stop**（排序能力优先于绝对误差）。
- 配置：`configs/v1_stage1_head.yaml`。

---

## 7. 评估（`silva/metrics.py` + `silva/evaluate.py`）

只看你的验证集。指标：**MAE、RMSE、Pearson、Spearman、QWK、Top-5% precision**。
`evaluate.py` 可对指定 checkpoint 在 val/test 上单独出报告。

---

## 8. 测试（`tests/`，纯函数 TDD）

GPU 训练不单测，纯函数全部先写测试：
- `make_ordinal_targets`：5 个分数 → 正确阈值向量。
- `OrdinalHead`：阈值单调递增；`score` 落在 `[1,5]`。
- 指标实现：与已知小样本手算结果对拍（尤其 Spearman、QWK、Top-K）。
- manifest split：同一 `image_path` 不跨 split。

---

## 9. 项目结构

```
silva/
  pyproject.toml
  .python-version          # 3.12.6
  .env.example             # DATABASE_URL
  README.md
  configs/
    v1_stage1_head.yaml
  silva/
    config.py              # Pydantic 配置模型
    data/
      export_manifest.py
      dataset.py
    models/
      ordinal_head.py
      siglip_aesthetic.py
    losses.py
    metrics.py
    train.py
    evaluate.py
  scripts/
    export_manifest.py     # 导出 CLI 入口
  tests/
```

---

## 10. 完成定义（Definition of Done）

1. `export_manifest` 能从 DB（或 mock）产出符合 §3.1 schema 的 parquet。
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

扩展位在代码中以参数/桩形式预留（`aux_heads`、多任务 loss 接口、导出的 `scorer_a/b` 列），但本版不实现逻辑。
