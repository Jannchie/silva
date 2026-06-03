# SILVA HuggingFace Demo (Gradio Space) 设计

- 日期: 2026-06-03
- 状态: 待评审
- 关联: [silva-monorepo-publish-design](2026-05-31-silva-monorepo-publish-design.md)

## 目标

在 HuggingFace 上提供一个**可交互的 demo**:用户在网页上传一张插画,实时看到这个"个人审美"模型给出的分数。从已发布的模型页 `Jannchie/silva-aesthetic` 一键可达。

## 非目标(本轮明确不做)

- README / 模型卡里的"低→高一排展示图 + 分数"。
- demo 的预置示例图(one-click examples)。
- 上述两项都依赖"展示图图源"决策(避开真实画师作品),已与用户约定**延后**到后续迭代。

## 背景:为什么是 Space 而不是内嵌 widget

模型页右侧那个"上传图"widget,背后跑的是 HF serverless 推理,只支持标准库(transformers / diffusers / sentence-transformers 等)的**标准架构**。SILVA 是自定义的两段式 forward(先用 SigLIP2 把图编码成 1152-d embedding,再过 ordinal head),serverless 不会执行它;把 so400m + head 硬塞成标准 `image-classification` transformers 模型既工作量大、serverless 也未必愿意加载。

因此"网页上传图 → 打分"的现实落地是 **Gradio Space**:体验与 widget 几乎一致,推理代码就是现有 `silva[backbone]` 的 `score()`,且可通过 Space card 的 `models:` 字段与模型页关联(模型页会自动出现 "Spaces using this model" 卡片)。

## 架构与组成

代码维护在 monorepo 的 `space/` 目录(受版本控制,随库演进),再 push 到独立的 HF Space repo。

约定 Space repo: **`Jannchie/silva-aesthetic-demo`**(评审时可改)。

三个文件:

### 1. `space/app.py` — Gradio 界面 + 推理

- 进程启动时一次性加载:`scorer = SilvaScorer.from_pretrained("Jannchie/silva-aesthetic")`。
  - 这会懒加载 so400m backbone;CPU 冷启动几十秒,但只发生一次。
- `predict(image: PIL.Image) -> ...`:直接 `score = scorer.score(image)`。
  - `SilvaScorer.score()` 的入参类型已是 `str | PathLike | PIL.Image`,`_load` 对非路径直接透传 → **无需改 `silva` 包**。
  - 返回值是 `calibrated_score ∈ [0, 1]`(已校准到训练标签分布)。
- 输出呈现:
  - 大号显示 `0–1` 分数;
  - 一个只读进度条(`gr.Slider(interactive=False)` 或等价)直观表示高低;
  - 一个友好的 1–5 星估计 `≈ {1 + 4*score:.1f} / 5`,**明确标注为"估计"**(calibrated_score 已对齐 1–5 分布,这只是线性近似的友好展示)。
- 文案需强调这是 **某一个人的私人审美**,不是通用质量分,不代表任何他人的偏好。
- 不带预置示例图。

### 2. `space/requirements.txt`

```
silva-scorer[backbone] @ git+https://github.com/Jannchie/silva.git@main#subdirectory=packages/silva
gradio
```

- 用 **git 安装**(与 model_card 的安装示例一致),确保 Space 的推理代码与已发布 head 的架构严格匹配,避免 PyPI 版本滞后导致的架构漂移。
- CPU 版 torch 由 pip 在 CPU Space 上自动选择;`[backbone]` extra 带 `transformers>=5.0`(CVE 下限)+ `pillow`。

### 3. `space/README.md` — Space card

YAML 头大致:

```yaml
---
title: SILVA Aesthetic Scorer
emoji: 🎨
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: mit
models:
  - Jannchie/silva-aesthetic
---
```

- `models: [Jannchie/silva-aesthetic]` 声明后,模型页会自动出现 "Spaces using this model"。
- 正文简述用途 + 同样的"个人审美、非通用质量分"免责声明。

## 模型卡链接(本轮一并做)

改 `packages/silva-train/src/silva_train/model_card.py`:

- 新增常量 `DEMO_URL = "https://huggingface.co/spaces/Jannchie/silva-aesthetic-demo"`。
- 在卡片标题 `# SILVA — Personal Aesthetic Head` 下方加一行醒目链接,如:
  `**[▶ Try it in your browser](DEMO_URL)**`
- 下次运行 `scripts/push_to_hub.py` 重新生成模型卡时生效(本轮不强制重新 push,除非用户要)。

## 硬件 / 取舍

- 免费 CPU basic(2 vCPU / 16 GB):足够跑 so400m 单图推理(几秒/张)。
- 取舍:免费 Space 闲置会休眠,唤醒需重新加载模型(几十秒冷启动)。要常驻无冷启动才需付费硬件。本轮按免费 CPU 设计。

## 验证

- 本地起 demo:在仓库根 `uv run python space/app.py`(workspace 已含 `silva[backbone]` 依赖),浏览器上传任意一张本地图,确认能稳定出分、UI 呈现正确。
  - 该验证图仅用于本地冒烟,不入库、不展示,不涉及图源决策。
- 打分逻辑本身已被 `silva` 包既有测试覆盖;demo app 是薄封装,无需额外单测。
- `model_card.py` 改动:既有 `test_model_card.py` 若校验卡片内容,补一条断言链接出现(若无相关测试则不强加)。

## 部署步骤(执行阶段,需要用户的 HF 凭据)

1. 在 HF 新建 Space `Jannchie/silva-aesthetic-demo`(Gradio,CPU basic)。
2. 把 `space/` 下三个文件 push 到该 Space repo。
3. 等待构建,打开 Space 上传图验证。
4. (可选)重跑 `push_to_hub.py` 让带 demo 链接的模型卡生效。

## 未来工作

- 展示图图源决策(AI 生成 / 自有素材 / CC0 / 退化图,避开真实画师作品),据此生成"低→高一排 + 真实分数"展示图,补进 README 与 Space 示例图。
