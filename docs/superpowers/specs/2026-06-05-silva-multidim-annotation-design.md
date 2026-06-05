# SILVA 多维度打分与 pictoria 标注基础设施设计（v-next）

> **方向修正（2026-06-05，实标后）**：维度绝对分被实际标注体验否定——单图多维连按
> 全面 halo，随机 pair 上维度比较退化为整体比较（"差别大时只在意总分"）。美学判断
> 整体优先（holistic-first），维度分解缺乏独立信号。**主线改为 pairwise overall**：
> 双图选总分（默认 UI 流），比较判断绕开绝对分的 retest 天花板 / 5 分通胀 / 标准漂移，
> ranking loss 与 107k 绝对分混训精炼总分模型。pair 规模规划：1k held-out →
> 每 +2~3k 增量看学习曲线，预期 8~15k 见效；接近对采样（按回填分配 50% 胜率对）
> 可砍半。维度降级为实验假设：仅在"总分匹配对"（整体接近的 pair）上可能有信号，
> 小规模验证不通过即放弃，维度标签走 tag 代理。基础设施（事件表/流式/pairwise UI）
> 全部直接服务新主线。以下原文保留作为设计记录。

把单一总分拆成**颜色 / 完成度 / 构图**三个个人偏好维度；把标注从一次性 sprint 升级为
**pictoria 内置的长期标注基础设施**（append-only 事件流，兼容 5 级 / 二元 / 三元 / pairwise 全形态）。

---

## 1. 背景与动机

来自 v1 的三个已验证结论驱动本设计：

1. **总分混轴**：tag 去混淆分析证实，单一 1~5 分混了画质 / 格式（sketch、速涂、3koma）/
   装饰精致度 / 内容偏好多条轴。低分最强关联是格式轴，高分是装饰精致度代理。
2. **标注天花板**：intra-rater 复测 spearman 仅 0.577（上限 ~0.76），模型 0.773 已饱和。
   继续卷模型无意义；噪声的相当部分来自轴混淆——同一张图两次打分时对各轴的权衡漂移。
3. **重标循环是最大杠杆**（test 0.737→0.773）：OOF audit → review_page → refresh.sh
   的增量循环已被验证，本设计延续此形态，只是把标注端搬进 pictoria 并支持多维。

最终目的：**做出更好的模型**——维度分准确、排序能力强（选品）、可全库回填，
并为将来生成模型训练提供按维度的条件标签（quality tags 按维度拆开）。

---

## 2. 目标与非目标

**目标**：

- pictoria 内置标注系统：标注记录**立即落库、立即可查**，需要时导出训练。
- 标注存储兼容全部形态：原始 5 级、二元、三元（多维度）、pairwise。
- SILVA 多头模型：在 frozen SigLIP2 embedding 上，总分 head（107k 旧标签）+
  颜色 / 完成度 / 构图三个维度 head。
- 形态对比实验：用数据裁决二元 / 三元 / pairwise 哪种是主力，不靠直觉赌。
- 全库回填多维分 + 离散桶（沿用直方图规定化校准）。

**非目标（本版不做）**：

- 构图子维度细分（姿势 / 角度 / 场景拆开标）——等构图维度 retest 数据说话。
- 内容 / 题材维度——难以界定好坏，不标；halo effect 用细则 + 可选 flag 缓解。
- 画质维度——已证实模型 intrinsic 读得到（lowres / jpeg_artifacts），不花人工。
- 多人标注 / 标注者管理——单人系统，schema 留 `session_id` 即可。
- VLM 蒸馏预标注——与"个人寡好"定位冲突。

---

## 3. 维度定义

| key           | 维度   | 定义                                                     | 不属于本维度                     |
| ------------- | ------ | -------------------------------------------------------- | -------------------------------- |
| `color`       | 颜色   | 色彩**运用**得好不好（合理性、品味），不是丰富 / 鲜艳程度 | 题材喜恶、完成度                 |
| `finish`      | 完成度 | sketch→精修的精修程度、细节装饰精致度                     | 纯格式问题（3koma 等留给 tags）  |
| `composition` | 构图   | 广义演出：姿势动态、镜头角度、场景安排                     | 画质、上色质量                   |
| `overall`     | 总分   | 兼容旧 1~5 流程的合法维度                                 | —                                |

**评分细则（rubric）是一等公民**：

- 每个维度动手标之前先写 5~10 条细则，含每档锚点描述与切点定义
  （二元切点建议："这个维度好到让我想收藏这张图吗"）。
- 细则带版本号（`rubric_version`），存进每条标注事件；改版后旧标注可识别、可降权。
- 首批 200 张标完后回头修订一次细则（v1 教训：细则是比维度数量更大的杠杆）。
- 细则文件放 `docs/rubrics/<dimension>.md`，版本号即文件内声明的 `v1`、`v2`。
- 每个维度细则中明确写"忽略题材偏好"；标注 UI 提供可选单键题材 flag
  （特别喜欢 / 讨厌时按，默认不按，不算分，仅用于事后检查 halo 污染）。

---

## 4. 标注事件模型（pictoria 侧）

**核心决定：append-only 事件流，永不覆盖。** 聚合策略（最新优先 / 多数票 / 时间衰减）
在导出时决定，不固化进存储。

三种事件三张表（pictoria 无 ORM、hand-written repository：单表多态没有复用收益，
反而列稀疏、约束要靠 `CASE WHEN`、每个查询都得 filter kind；且 absolute / pairwise
的消费路径本来就完全分离——导出即两种 parquet）。新增 migration（`00xx_annotations.sql`）：

```sql
CREATE TABLE absolute_annotations (
    id             INTEGER PRIMARY KEY,
    created_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    post_id        INTEGER NOT NULL,
    dimension      TEXT    NOT NULL,            -- 'color' | 'finish' | 'composition' | 'overall'
    scale          INTEGER NOT NULL CHECK (scale IN (2, 3, 5)),
    value          INTEGER NOT NULL,            -- 档位：1..scale
    rubric_version TEXT    NOT NULL,
    session_id     TEXT    NOT NULL,            -- 一次连续标注会话一个 id
    elapsed_ms     INTEGER                      -- 从呈现到判断的耗时
);
CREATE INDEX idx_absolute_annotations_post ON absolute_annotations (post_id, dimension);

CREATE TABLE pairwise_annotations (
    id             INTEGER PRIMARY KEY,
    created_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    post_a         INTEGER NOT NULL,
    post_b         INTEGER NOT NULL CHECK (post_b != post_a),
    dimension      TEXT    NOT NULL,
    winner         TEXT    NOT NULL CHECK (winner IN ('a', 'b', 'tie', 'skip')),
    rubric_version TEXT    NOT NULL,
    session_id     TEXT    NOT NULL,
    elapsed_ms     INTEGER
);
CREATE INDEX idx_pairwise_annotations_posts ON pairwise_annotations (post_a, post_b, dimension);

-- 题材 flag 是对图的独立动作（不依附于某次维度判断），单独成事件表
CREATE TABLE content_flag_events (
    id         INTEGER PRIMARY KEY,
    created_at TEXT    NOT NULL DEFAULT (datetime('now')),
    post_id    INTEGER NOT NULL,
    flag       TEXT    NOT NULL CHECK (flag IN ('love', 'hate', 'none')),  -- 'none' = 撤销
    session_id TEXT    NOT NULL
);
CREATE INDEX idx_content_flag_events_post ON content_flag_events (post_id);

-- 统一时间线（跨形态的标注活动视图，供历史展示 / session 统计）
CREATE VIEW annotation_timeline AS
    SELECT id, created_at, 'absolute' AS kind, post_id, dimension, session_id FROM absolute_annotations
    UNION ALL
    SELECT id, created_at, 'pairwise' AS kind, post_a AS post_id, dimension, session_id FROM pairwise_annotations;
```

这个模型买到的能力：

1. **重复标注 = 免费 retest 数据**：同图同维度再标一次即 intra-rater 样本；
   按 `created_at` 看时间序列即标准漂移检测（v1 的 5 分通胀就是这种漂移）。
2. **`rubric_version`**：细则改版后旧标注可识别。
3. **`elapsed_ms`**：犹豫久 = 边界样本 = 免费的主动学习信号；也可分析疲劳效应。
4. **聚合可反悔**：导出时任选策略，原始判断永在。

Pydantic entity 与表一一对应（`AbsoluteAnnotation` / `PairwiseAnnotation` / `ContentFlagEvent`），
不需要多态校验。

---

## 5. 标注 UI（pictoria web）

两个模式，全键盘流，每次判断自动记 `elapsed_ms`：

**模式 A：单图多维（absolute）**

- 一屏一图，按当前队列配置的 scale 显示档位。
- 快捷键示例（三元）：`1/2/3` 颜色、`q/w/e` 完成度、`a/s/d` 构图；
  二元时每维两键；5 级时单维 `1~5`（兼容旧 overall 流程）。
- `x` = 题材 flag（循环 none→love→hate），`space` = 跳过。
- 配置的维度全部选完自动翻页；三维一图约 3~8 秒。
- 已标过的图显示历史标注（小徽章），但**默认进入队列时不显示**（避免锚定），
  复测场景刻意盲标。

**模式 B：双图 pairwise**

- 左右两图，顶部显示当前问的维度（一次只问一个维度）。
- `←/→` 选边，`↓` = tie，`space` = skip。

**即时存档与展示**：事件落库即生效；post 详情页新增"标注历史"区块
（各维度的事件时间线）。统计面板（已标计数、维度覆盖、kappa）后置，非首版必需。

---

## 6. 出题：流式采样为默认，队列仅用于固定批次

**日常标注无队列**：标注页选好维度/档位/策略直接开始，服务端按需采样下一张/下一对
（`GET /annotations/sample-absolute` / `sample-pairwise`），已标过的图自动排除。
打开即标、标到不想标，无批次管理负担。

**队列只服务固定集合场景**：形态对比实验（同一批图标三遍）和 intra-rater 复测
（同一批图隔期再标）必须锁定一组图——这是队列唯一不可替代的用途，UI 中折叠为次要功能。

采样由 **pictoria 内置**（依赖方向：silva → pictoria 单向，pictoria 不依赖任何 silva 侧脚本）。
采样所需信息 pictoria DB 全有：`posts.score`（旧手工分）、`post_vectors_siglip2`（embedding）、
`post_aesthetic_scores`（silva 回填分）、标注事件表（排除已标）。

生成策略（P0 实现前两个）：

- `random`：随机抽未标注、未排队、有 embedding 的图。
- `stratified`：按旧 `posts.score` 1~5 分层均匀抽，层内随机，不足回退 random 补齐。
- 后续：coverage（embedding 多样性）、boundary（回填维度分边界带出 pair）、
  OOF 分歧——全部基于 pictoria 库内数据，不需要外部输入。

POST 导入端点保留为通用接口（任何外部进程仍可导入自定义队列），但不是常规路径。

```sql
CREATE TABLE annotation_queues (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,                  -- 'coldstart-2026-06', 'oof-color-r1', ...
    kind       TEXT NOT NULL CHECK (kind IN ('absolute', 'pairwise')),
    dimensions TEXT NOT NULL,                  -- JSON list，模式 A 可一次配多维
    scale      INTEGER,                        -- absolute 队列用
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
-- items 与事件表同样按形态拆，避免同一种多态稀疏
CREATE TABLE absolute_queue_items (
    queue_id   INTEGER NOT NULL REFERENCES annotation_queues(id),
    position   INTEGER NOT NULL,
    post_id    INTEGER NOT NULL,
    done       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (queue_id, position)
);
CREATE TABLE pairwise_queue_items (
    queue_id   INTEGER NOT NULL REFERENCES annotation_queues(id),
    position   INTEGER NOT NULL,
    post_a     INTEGER NOT NULL,
    post_b     INTEGER NOT NULL,
    done       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (queue_id, position)
);
```

- 生成入口：`POST /annotation-queues/generate-absolute` / `generate-pairwise`
  （参数：dimensions、scale、count、strategy），UI 提供新建队列表单。

---

## 7. 导出契约（pictoria → silva）

导出脚本（pictoria 侧 CLI 或 silva 侧直读 sqlite，沿用 `score_pictoria.py` 直连
`pictoria.sqlite` 的既有模式）产出两种 parquet：

**absolute 导出**（join `post_vectors_siglip2`）：

| 列               | 类型           | 说明                                   |
| ---------------- | -------------- | -------------------------------------- |
| `embedding`      | list<float>[D] | SigLIP2 embedding（库里现成）          |
| `dimension`      | str            | 维度 key                               |
| `scale`          | int            | 2 / 3 / 5                              |
| `value`          | int            | 聚合后档位（策略：默认最新优先）       |
| `n_events`       | int            | 该图该维度事件数（>1 即有 retest）     |
| `rubric_version` | str            |                                        |
| `post_id`        | int            | 可选溯源（切分已用内容 hash，不依赖）  |

**pairwise 导出**：`embedding_a, embedding_b, dimension, winner, rubric_version, post_id_a, post_id_b`。

silva 已支持 multi-parquet manifest（list 即时合并、内容 hash 切分），
新维度数据作为追加 parquet 进入，旧 107k manifest 不动。

---

## 8. silva 侧训练

**架构**：frozen SigLIP2 embedding → 共享 trunk（沿用已验证配置）→ 多头：

- `overall` head：继续消费 107k 旧标签（QWK + LS 已验证配方），充当表征辅助任务，
  防止小数据维度 head 过拟合。
- `color` / `finish` / `composition` 各一个小 head。

**统一 loss 框架——形态不赌**：

- 所有 absolute 形态统一为 **ordinal threshold**：二元 = 1 阈值、三元 = 2、5 级 = 4。
  同一 head 的连续 latent 输出可同时消费不同 scale 的标签（不同批次标注形态可混存）。
- pairwise → margin ranking loss（同 head latent 上比较），与 absolute loss 加权混训。
- 维度 head 输出连续 latent；回填时沿用直方图规定化（PCHIP）校准切桶——
  **二元标注 ≠ 二元输出**，排序信息在 latent 里，桶数导出时定。

**队列生成不在 silva 侧**（§6 修正）：采样策略内置于 pictoria。silva 对采样的影响
通过**回填分数**间接实现——多维分回填 `post_aesthetic_scores` 后，pictoria 的
boundary / OOF 策略读这些分数出队列。silva 保持对 pictoria 的单向依赖。

---

## 9. 评估

每个维度独立评估，第一天就测天花板：

1. **per-dim intra-rater 天花板**：复测事件自动累积（§4），二元 / 三元用 Cohen's kappa，
   连续用 spearman。**哪个维度 retest 烂，先停哪个**（v1 教训：别对着噪声卷模型）。
2. **维度间相关矩阵**：标注层面相关 >0.8 → 没拆开，合并或改细则。
3. **对旧总分的解释力**：三维度分回归旧 1~5 总分的 R²，验证拆出的轴覆盖总分主要方差；
   残差即"未建模口味"。
4. 常规 per-dim：spearman、QWK（按 scale）、top-k 选品命中。
5. **halo 检查**：content_flag 图与非 flag 图的维度分分布对比，量化题材污染。

---

## 10. 形态对比实验（标注基础设施跑通后的第一件事）

裁决"二元 / 三元 / pairwise 哪个是主力"：

- 同一批 ~200 张图（coverage + 分层采样），三种形态各标一遍（pairwise 用其中的图组 ~200 对），
  间隔 ≥3 天后每种复测一轮。
- 三个指标，全部自动可测：
  1. **速度**：`elapsed_ms` 中位数。
  2. **可靠性**：retest 一致性（kappa / pairwise 翻转率）。
  3. **单位时间模型增益**：各形态数据分别训 head，固定一个 held-out 评估集
     （三种形态共同覆盖的图 + 维度），比 spearman / 每分钟标注的提升。
- **待验假设**（写在前面，输了认）：三元赢绝对分赛道（"无功无过"是真实感知类别，
  强行二分反而慢且噪）；pairwise 赢精炼赛道（边界带两图选边可靠性最高）。
- 实验结论决定冷启动主力形态；pairwise 无论谁赢都保留为精炼工具。

---

## 11. 路线图

| 阶段 | 内容                                                         | 产出                         |
| ---- | ------------------------------------------------------------ | ---------------------------- |
| P0   | 细则 v1（三维度）+ pictoria migration + 标注 UI 模式 A/B + 队列 | 可标注、事件落库             |
| P1   | 形态对比实验（200 张 ×3 形态 + 复测）                          | 主力形态结论、per-dim 天花板 |
| P2   | 冷启动队列 500~1000 张 × 主力形态                              | 维度标注 v1 数据集           |
| P3   | 导出契约 + silva 多头训练 + 评估（§9）                          | head v0、维度相关矩阵、R²    |
| P4   | OOF / 边界带增量循环（refresh.sh 维度版）                       | 持续改进闭环                 |
| P5   | 全库回填多维分 + 校准切桶（score_pictoria 多列化）              | 生成模型条件标签             |

P0 内部顺序：migration → 导出 CLI（先通数据路径）→ UI。

---

## 12. 风险与开放问题

- **构图维度可学性**：SigLIP 对纯几何布局不敏感；但本设计的构图偏语义
  （姿势动态 / 角度 / 场景），danbooru 体系内有对应概念，embedding 应有信号。
  P3 评估见真章；若 head 学不动且 retest 尚可，升级路径是换 / 加位置敏感特征
  （如 DINOv2），属后续设计。
- **构图 retest 噪声**：预计三维度中最大。若 P1 复测显著差于其他维度，
  对该维度加大 pairwise 配比或暂停。
- **维度间高相关**：颜色与完成度可能纠缠（精修图往往配色也好）。>0.8 时合并或改细则。
- **类不平衡**（二元切点偏上时正例 ~20-30%）：v1 有 pos_weight 经验教训
  （关 pos_weight 双峰不消），维度 head 校准依赖回填时直方图规定化，不强求 latent 形状。
- **pictoria API/UI 工作量**：标注系统是全栈功能（migration + Litestar 路由 +
  Vue 页面 + genapi），是本设计最大的工程投入，P0 控制在最小可标注闭环。
