# 实验框架搭建指南（给 Claude Code）

2026-10-12 · 作者：Zx · 在线版：https://claude.ai/code/artifact/06d59d7c-66fc-4bf4-92d0-579185b19506

先搭能跑 E0 的最小框架：一个可穷举的 4 节点审计空间、一个带 token 记账与缓存的执行器、两个优化器、一套不碰最终测试集的分析脚本；按下面的顺序建，每步有验收。

## 目标与范围

第一阶段只做 E0：在一个模型、一个任务族、一个可穷举的审计空间上，把一个优化器的平台期分解成四项并给出置信区间。框架要为后面的 E1–E6 留好接口，但不提前实现它们。

E0 要交付的东西：

- 审计空间 S₀ 里每张图在审计集、训练集上的分数（每张图在审计集上 2 次重复）。
- 单智能体参照 Π_single 四种策略在同预算下的分数，取最优者。
- 随机搜索与 GRPO 式优化器各 5 个种子的完整轨迹：每次候选评估的图、任务、分数、token。
- 分析脚本输出：Fig 1b（学到的图 / 经验最优图 / oracle 参照图在最终测试集上的分数与 CI）、学习曲线、四项分解的堆叠条、单次 max 与交叉拟合的差。
- 一份 run manifest，任何一张图、任何一次评估都能从缓存里重放。

第一阶段不做：多任务族、异构模型池、工具空间、MCTS 与其他优化器、逐 query 策略。这些只需要接口上可扩展（见数据结构一节），不需要实现。

默认实验参数（写进 `configs/e0.yaml`，可改）：模型 Qwen3-8B（vLLM，OpenAI 兼容端点）；任务族数学（MATH 训练集抽样，MATH-500 只做最终测试）；S₀ = 4 节点、6 条可选前向边；训练集 N = 256，审计集 300，验证集 150，最终测试集 500（即 MATH-500 全集）；温度 0.7 固定；每个节点最多 1,024 个生成 token；优化器预算 200 次候选评估 × 每次 24 道训练题；种子 5 个；预登记 ε = 1.0 分，Δt = 50 次评估。

## 整体架构

数据流：任务数据 → 四角色划分 → 图空间 S₀ 与单智能体参照 → **执行器（唯一的 LLM 端点出口）** → RunRecord 缓存 → 优化器 / 审计 → 分析与图表。

```text
  Data & splits          Space S₀ + Π_single          Optimizers
  data/loaders           space/s0.py                  random_search · grpo
  data/splits            single_agent.py              预算 200 评估 × 24 题
  四角色零重叠            64 边子集 → 去重 DAG + hash
        │                        │                          │
        │ tasks                  │ graphs · Π_single        │ evaluate_batch ⇄ scores
        └──────────────┐         ▼         ┌────────────────┘
                   ┌───────────────────────────┐
                   │        Executor           │   ← 唯一的端点出口，记账 + 缓存
                   │   runtime/executor.py     │
                   └───────────────────────────┘
        ┌──────────────┘         │         └────────────────┐
        │ LLM calls              │                          │ RunRecord
        ▼                        ▼                          ▼
  LLM endpoints            Analysis                   RunRecord cache
  vLLM Qwen3 on H20        decompose · plateau · CI    runtime/cache.py
  ds / glm flash（本地）     只读记录 → Fig 1b/2/3        key = model·graph·task·seed
```

读法：上排是输入（任务、图空间与单智能体参照、优化器），中间的执行器是唯一能调用 LLM 端点的模块，它把每次运行写成 RunRecord 进缓存；优化器只通过 evaluate_batch 拿分数，分析脚本只读缓存里的记录。这样算力记账、重放和"不碰最终测试集"都只需要在一个地方保证。

## 仓库结构与构建顺序

按下面的顺序建，每一步先过验收再往下走；第 1–4 步可以全部用 mock 端点跑通，不耗 GPU。

```text
topo-audit/
├── configs/            # e0.yaml 等；一个实验一个文件，命令行只传配置路径
├── data/
│   ├── loaders/        # math.py, musique.py, masbench.py：统一产出 Task
│   └── splits/         # 固定种子生成的 train/audit/val/test 任务 id 列表（提交到 git）
├── space/              # topology.py（数据结构、规范化、hash）、s0.py（枚举审计空间）
├── runtime/
│   ├── endpoints.py    # OpenAI 兼容客户端：并发、限速、重试、token 计数
│   ├── executor.py     # 按拓扑顺序执行节点，产出 RunRecord
│   ├── single_agent.py # Π_single 四种策略，同预算
│   └── cache.py        # key=(model, graph_hash, task_id, seed, repeat) → RunRecord
├── optimizers/         # base.py（接口与预算）、random_search.py、grpo.py、mcts.py（第一阶段只留桩）
├── audit/              # run_audit.py：S₀ × 审计集 × 训练集的全量评估
├── analysis/           # decompose.py、plateau.py、bootstrap.py、figures.py
├── scripts/            # e0_*.sh：每个实验一条命令
├── tests/              # 规范化、打分、记账、分解恒等式的单元测试
└── runs/               # 产物：runs/<exp>/<model>/<ts>/{manifest.json, records.jsonl, figures/}
```

| 步骤 | 建什么 | 验收 |
| --- | --- | --- |
| 1 | `space/`：Topology 数据结构、剪枝、规范化、hash；S₀ 枚举 | 单元测试：64 个边子集枚举后去重，打印不同的可执行 DAG 数；同一张图任意边序 hash 相同 |
| 2 | `data/`：MATH 加载与打分（boxed 答案归一化后精确匹配）；固定种子划分 | 四个角色互不重叠；重跑划分脚本结果逐字节一致；20 道人工核对的答案打分全对 |
| 3 | `runtime/`：端点客户端、执行器、缓存、token 记账 | mock 端点下跑通 S₀ 全部图 × 5 道题；每条 RunRecord 的 prompt/gen token 之和等于端点返回的用量；重跑命中缓存、零次调用 |
| 4 | `runtime/single_agent.py` | 四种策略在同一预算上限下跑通并记账 |
| 5 | `optimizers/`：接口 + 随机搜索 + GRPO 式 | 预算计数一致（候选评估次数 × 每次任务数）；轨迹 JSONL 可重放；两个优化器在 mock 端点上各跑 5 种子不报错 |
| 6 | `audit/`、`analysis/` | 用 mock 分数验证分解恒等式左右相等到 1e-9；交叉拟合、bootstrap、平台期检验有单元测试 |
| 7 | `scripts/e0_*.sh` + 真端点 | 先跑 1 小时吞吐基准，再跑 E0 全量，产出验收一节要求的图表 |

## 核心数据结构与接口

六个对象贯穿整个框架；后面所有实验（异构模型、工具、更大空间）都只是往 `Topology` 的节点属性里加字段，不改接口。

| 对象 | 字段 | 说明 |
| --- | --- | --- |
| `Topology` | `nodes: [Node]`、`edges: [(src, dst)]`、`space_id`、`hash` | `Node = {id, role, prompt_id, model, tools: []}`；第一阶段 `model` 全部相同、`tools` 为空。`hash` 由规范化后的可执行子图算出 |
| `Task` | `task_id`、`family`、`split_role`、`prompt`、`answer`、`meta` | `split_role ∈ {train, audit, val, test}`，由划分文件决定，加载时写死 |
| `RunRecord` | `run_id`、`exp`、`model`、`graph_hash`、`task_id`、`split_role`、`seed`、`repeat`、`score`、`prompt_tokens`、`gen_tokens`、`latency_s`、`node_outputs_path`、`endpoint_usage` | 一行 JSONL；`score` 为 0/1 或 [0,1]；`endpoint_usage` 原样保存端点返回的用量，便于对账 |
| `Executor` | `run(topology, task, seed, repeat) -> RunRecord` | 先查缓存；按拓扑序执行；每个节点的输入 = 任务 + 入边邻居的输出；最终答案取终端节点输出 |
| `Optimizer` | `propose(state) -> [Topology]`、`observe(evals)`、`budget: Budget`、`trajectory` | 候选评估通过统一的 `evaluate_batch(topologies, task_ids)` 走执行器，优化器自己不能调端点 |
| `Budget` | `max_evals`、`tasks_per_eval`、`spent_evals`、`spent_tokens` | 一次"候选评估" = 一张图在 `tasks_per_eval` 道训练题上各跑一次；token 同时累计，供等算力比较 |

配置文件示例（`configs/e0.yaml`）：

```yaml
exp: e0
model:
  name: qwen3-8b
  endpoint: http://localhost:8000/v1
  max_concurrency: 64
  temperature: 0.7
  max_tokens_per_node: 1024
  thinking: false
task:
  family: math
  splits: data/splits/math_v1.json   # train 256 / audit 300 / val 150 / test 500
space: s0                            # 4 nodes, 6 optional forward edges
audit:
  repeats: 2
optimizers: [random_search, grpo]
budget:
  max_evals: 200
  tasks_per_eval: 24
seeds: [0, 1, 2, 3, 4]
plateau:
  epsilon: 1.0
  delta_t: 50
  checkpoints: [25, 50, 100, 150, 200]
```

接口约束：`Executor` 是唯一能调用端点的地方；`Optimizer` 只能通过 `evaluate_batch` 拿分数；`analysis/` 只读 `records.jsonl`，不调用端点。

## 审计空间 S₀ 的定义与执行语义

S₀ 是整篇论文的"尺子"，定义要写死、可枚举、可重放：4 个节点按固定顺序执行，6 条可选前向边，剪掉不通向终端节点的部分后去重。

**节点与角色（固定顺序 n1 → n2 → n3 → n4）。** n1 Solver：独立解题，给出推理与答案；n2 Critic：检查收到的解答，指出错误；n3 Reviser：根据收到的内容给出修订后的解答；n4 Finalizer：综合收到的全部内容，输出最终答案（固定格式 `FINAL: <answer>`）。每个角色一个固定提示模板，放在 `prompts/`，第一阶段不优化提示。

**边。** 6 条可选前向边：n1→n2、n1→n3、n1→n4、n2→n3、n2→n4、n3→n4，共 2^6 = 64 个边子集。节点的输入 = 任务原文 + 所有入边邻居的输出（按节点序拼接，带角色标签）。

**执行与剪枝规则。**

- n4 永远执行：没有任何入边时它就是单独解题的单 agent（这是空间里的"零边"基线）。
- 没有路径通向 n4 的节点不执行（不消耗 token）；例如只有边 n1→n2 的图，n1、n2 都不通向 n4，实际执行的只有 n4。
- 规范化：把不执行的节点和它们的边删掉，得到可执行子图；对可执行子图做确定性序列化后取 SHA-256 作为 `graph_hash`。64 个边子集会坍缩成更少的可执行 DAG，S₀ 以去重后的集合为准，并在论文里报告这个数字。
- 一张图的 token 成本 = 执行过的节点的 prompt + generation token 之和；不同图成本不同，等算力比较在分析阶段按 token 分层或配预算做，执行阶段不截断。

**单智能体参照 Π_single（同模型、同温度，预算上限 = S₀ 中各图在审计集上的 token 中位数）。**

| 策略 | 做法 | 预算怎么对齐 |
| --- | --- | --- |
| CoT | 单次解题，`FINAL:` 格式 | 自然低于上限，直接报 |
| 自洽 @k | k 次独立采样，多数投票 | k 取使总 token ≤ 上限的最大值 |
| best-of-N + 自校验 | N 次采样，模型自评打分后取最高 | N 取使总 token ≤ 上限的最大值 |
| 自精炼 | 解题 → 自我批评 → 修订，循环 r 轮 | r 取使总 token ≤ 上限的最大值 |

四种策略都在验证集上选参数（k、N、r），在最终测试集上报告，取其中最高者为 P_single。`space/s0.py` 的单元测试要覆盖：零边图的 hash 等于"只有 n4"的图；任意同构边序 hash 一致；每张图执行节点数与剪枝规则一致。

## 数据划分、执行器与算力记账

四个数据角色在加载时就写死，`split_role` 跟着 `Task` 走，任何脚本都不能在最终测试集上做选择；执行器是唯一的端点出口，每一次调用都记账、都进缓存。

| 角色 | 来源（数学任务族） | 规模 | 谁能用 |
| --- | --- | --- | --- |
| train | MATH 训练集（7,500 题）固定种子抽样 | 256 | 优化器的候选评估；计算 P̂_N |
| audit | MATH 训练集中与 train 不重叠的抽样 | 300 | S₀ 全量评估、交叉拟合选 G* |
| val | 同上，与前两者不重叠 | 150 | Π_single 选参数、学习曲线与平台期检验 |
| test | MATH-500 全集 | 500 | 只在 `analysis/report.py` 里读一次，产出最终数字 |

划分脚本 `data/make_splits.py --family math --seed 20261012` 产出 `data/splits/math_v1.json`，提交到 git；加载器按题目文本 hash 去重，保证四个角色零重叠。换任务族只换加载器与打分器，角色与规模规则不变。

**打分。** 数学：从 `FINAL:` 行抽答案，用与 MATH-500 社区实现一致的归一化（去空格、LaTeX 等价形式、分数与小数统一）后精确匹配；放 20 道人工核对样例进测试。无法解析到答案记 0 分并在 `RunRecord.meta` 里标 `parse_fail`，分析时单独统计比例。

**端点客户端（`runtime/endpoints.py`）。** OpenAI 兼容接口，一个模型一个客户端；信号量限并发（`max_concurrency`），指数退避重试最多 3 次且每次重试记入日志；超时与 5xx 不静默吞掉；返回对象里保留端点的 `usage` 原文。两个 flash 模型与 vLLM 上的 Qwen3 都走同一客户端，只换 `endpoint` 与 `name`。

**执行器（`runtime/executor.py`）。** 输入 `(topology, task, seed, repeat)`；先算 `cache_key = (model.name, graph_hash, task_id, seed, repeat)` 查缓存；未命中则按拓扑序执行可执行子图，节点间只传输出文本；每个节点的 `max_tokens_per_node` 固定；采样时把 `seed` 传给端点（vLLM 支持 `seed` 参数；flash 端点若不支持，记录即可，不要求位级可复现）；把所有节点的输出写到 `runs/.../outputs/<run_id>.json`，`RunRecord` 只存路径。

**记账。** `prompt_tokens`、`gen_tokens` 按节点累加写入 `RunRecord`；每个实验结束时 `manifest.json` 汇总总 token、总调用次数、重试次数、缓存命中率、墙钟时间；探测与重复评估的 token 同样计入，后面做等算力比较时不能漏。

**缓存（`runtime/cache.py`）。** SQLite 或按 key 分目录的 JSON 都可以，要求：并发安全、重跑同一实验零端点调用、可按 `exp` 导出。缓存里的 RunRecord 是分析脚本唯一的数据来源。

## 优化器

所有优化器共用一个预算定义和一个评估入口，彼此只在"下一批候选怎么选"上不同；第一阶段实现随机搜索与 GRPO 式，MCTS 式只留接口桩。

**统一预算。** 一次候选评估 = 一张图在 `tasks_per_eval = 24` 道训练题上各跑一次（同一种子内，24 道题由固定的批次序列给出，所有优化器用同一序列，保证配对）。预算 `max_evals = 200`，用完即停；优化器在检查点 `[25, 50, 100, 150, 200]` 各输出一张"当前最优图" `G_{O,t}`，由 `analysis/` 在验证集上评估得到学习曲线，优化器自己不碰验证集。

**统一接口（`optimizers/base.py`）。**

```text
class Optimizer:
    def __init__(self, space, budget, seed, rng): ...
    def propose(self) -> list[Topology]          # 下一批要评估的候选
    def observe(self, evals: list[EvalResult])   # EvalResult = (graph_hash, task_ids, scores[], tokens)
    def incumbent(self) -> Topology              # 当前最优图（按优化器自己的经验估计）
    def state(self) -> dict                      # 可序列化，写入 trajectory.jsonl
```

驱动循环在 `optimizers/runner.py`：`while budget.left(): batch = opt.propose(); evals = evaluate_batch(batch, next_task_batch()); opt.observe(evals); log(...)`。每一步写一行 trajectory：步数、候选 hash、每道题的分数、累计 token、`incumbent` 的 hash。

| 优化器 | 候选怎么选 | 要记录的额外状态 |
| --- | --- | --- |
| 随机搜索 | 从 S₀ 去重后的集合均匀抽样；维护每张图的运行均值与评估次数；`incumbent` = 均值最高且评估次数 ≥ 2 的图（避免单次运气） | 每张图的均值、次数 |
| GRPO 式 | 6 条边各一个 logit θ，边独立伯努利采样；每步采一组 G = 8 张图，在同一批 24 道题上评估（配对）；组内相对优势 A_i = (r_i − mean) / (std + 1e-6)；按 REINFORCE 更新 θ，学习率 0.1，裁剪 θ ∈ [−4, 4]，熵正则 0.01；`incumbent` = 边概率取整后的图 | θ 轨迹、组内分数、优势 |
| MCTS 式（桩） | 以"加一条边 / 删一条边"为动作的树搜索，UCB 选边，叶子用一次候选评估打分 | 第二阶段实现 |

**可分辨性诊断量。** 每一步额外记录组内（或当前 top-2）候选在同一批题上的配对分数差及其标准误，写入 trajectory，供分析脚本解释优化器何时停住；不设阈值、不用它做停止判断。

**用 mock 端点验收。** `tests/mock_endpoint.py` 给每张图一个固定的真实准确率（按边数单调递增加噪声），两个优化器在 5 个种子上都应在 200 次评估内把 `incumbent` 推到真实前三的图；这是优化器实现正确性的最低标准。

## 分析脚本：四项分解、交叉拟合、CI、平台期检验

`analysis/` 只读缓存里的 `records.jsonl`，按角色严格分工：审计集选图，训练集算经验分数，验证集画学习曲线，最终测试集只在 `report.py` 里读一次。

**输入矩阵。** 把记录聚成三张得分矩阵：`A[g, i, r]`（审计集，每张图 × 题 × 重复）、`T[g, i]`（训练集，每张图 × 256 题）、`V[g_t, i]`（验证集，各检查点的 `G_{O,t}` × 150 题）；最终测试集 `X[·, i]` 只对被选中的图与 Π_single 存在。

**选图（`decompose.py` 第一步）。**

- Ĝ_N = argmax_g mean_i T[g, i]（训练集经验最优）。
- G* 的交叉拟合：审计集按题随机对半分 A₁/A₂（固定种子）；在 A₁ 上选均值最高的图，记它在 A₂ 上的均值；交换再做一次；重复 20 次随机对半分，取平均作为 P(G*) 的无偏估计 P̂_cf；同时记录单次 max（在全部审计集上既选又报）与 P̂_cf 的差，这个差就是"选择偏差"，要进论文。
- 最终报告用的 G* = 全部审计集上的 argmax（只用来决定评估哪张图），它的真实分数来自最终测试集。

**四项（每个优化器、每个种子、每个检查点 t 各算一次，再跨种子汇总）。**

```latex
\begin{aligned}
C &= P_{\mathrm{test}}(G^{*}) - P_{\mathrm{test}}(\pi^{*}_{\mathrm{single}}) \\
E &= P_{\mathrm{test}}(G^{*}) - P_{\mathrm{test}}(\hat G_N) \\
O &= \hat P_N(\hat G_N) - \hat P_N(G_{O,t}) \\
\Delta_{\mathrm{gen}} &= \big[P_{\mathrm{test}}(\hat G_N) - \hat P_N(\hat G_N)\big] - \big[P_{\mathrm{test}}(G_{O,t}) - \hat P_N(G_{O,t})\big] \\
\text{check:}\quad P_{\mathrm{test}}(G_{O,t}) - P_{\mathrm{test}}(\pi^{*}_{\mathrm{single}}) &= C - E - O - \Delta_{\mathrm{gen}}
\end{aligned}
```

`decompose.py` 必须对每一行断言 check 成立（容差 1e-9），这是分解实现正确性的硬测试。

**置信区间（`bootstrap.py`）。** 所有差值都是同一批题上的配对差：对题目做 bootstrap（2,000 次），每次重采样题目索引后重算四项；跨种子的量（涉及 G_{O,t}）再对种子做一层 bootstrap。报告 95% 区间。`P(G*)` 另给一个区间：下界 = G* 在最终测试集上的分数，上界 = 审计集单次 max 加有限类修正 √(ln(2·|S₀|/0.05) / (2·300))。

**平台期检验（`plateau.py`）。** 对每个优化器：在检查点 t 与 t + Δt（Δt = 50）之间，取每个种子的 `G_{O,t}` 与 `G_{O,t+Δt}` 在验证集上的配对差，跨题与跨种子 bootstrap 得改善量的 95% 置信上界；上界 < ε = 1.0 分则记"在 t 处平台"。判据是排除而非未检出：上界 ≥ ε 时只写"未能宣布平台期"。渐近线拟合 `L(t) = P∞ − a·t^(−b)` 放在 `plateau.py --exploratory`，默认不运行。

**可分辨性（`discriminability.py`）。** 从 trajectory 里取每步的配对分数差与标准误，画"剩余候选差距 / 评估精度"随步数的曲线，与各优化器的停止点并列。

**输出（`figures.py`，全部存 `runs/<exp>/figures/`，附带数据 CSV）。**

| 图 | 内容 |
| --- | --- |
| Fig 1b | 一个平台期的归因实例：学到的图、经验最优图、oracle 参照图、Π_single 在最终测试集上的分数与 CI，并列 |
| Fig 2 | 两个优化器 × 5 种子的学习曲线（验证集），标出平台期判定点 |
| Fig 3 | 四项分解的堆叠条，按优化器分组，带 CI |
| Fig S1 | 单次 max 与交叉拟合的差（选择偏差）随审计集规模的变化 |
| Fig S2 | 可分辨性曲线与停止点 |

## E0 执行步骤、验收与硬规则

一条命令对应一步，顺序执行，前一步的产物是后一步的输入；全部通过后再谈扩模型。

1. `scripts/e0_bench.sh`：在 H20 上起 vLLM（Qwen3-8B），跑 1 小时吞吐基准（batch 64），记 prompt 与 generation tok/s，把结果写进 `runs/e0/bench.json`，并据此更新预算表的卡时估计。
2. `python data/make_splits.py --family math --seed 20261012`：生成并提交划分文件。
3. `python space/s0.py --enumerate`：枚举 64 个边子集，去重后打印可执行 DAG 数，写 `runs/e0/space_s0.json`。
4. `python audit/run_audit.py --config configs/e0.yaml`：S₀ 全部图 × 审计集 300 题 × 2 次，以及 × 训练集 256 题 × 1 次（约 64 × 856 ≈ 5.5 万次运行）。
5. `python runtime/single_agent.py --config configs/e0.yaml`：Π_single 四种策略，验证集选参、最终测试集报分。
6. `python optimizers/runner.py --config configs/e0.yaml --optimizer random_search` 与 `--optimizer grpo`：各 5 个种子，每个 200 次评估；检查点图在验证集上评估。
7. `python analysis/report.py --config configs/e0.yaml`：唯一读取最终测试集的脚本；产出四项、CI、平台期判定、Fig 1b / 2 / 3 / S1 / S2 和 `manifest.json`。

**验收（全部满足才算 E0 通过）。**

- [ ] 分解恒等式的 check 在每一行都成立（容差 1e-9）。
- [ ] 四项的 95% CI 能区分至少两种成因（例如 O 的区间与 E 的区间不重叠，或某一项显著大于 0 而另一项与 0 无法区分）。
- [ ] 单次 max 与交叉拟合的差被量化并报告，且交叉拟合估计落在 P(G*) 的区间内。
- [ ] 平台期检验在 5 个种子下对每个优化器给出明确结论（平台或未能宣布）。
- [ ] `manifest.json` 的总 token 与端点侧用量对账误差 < 1%；重跑 `report.py` 零端点调用。
- [ ] Fig 1b 能清楚看出学到的图与 oracle 参照图之间的差距及其来源。

**给 Claude Code 的硬规则。**

- 最终测试集只能被 `analysis/report.py` 读取；其他任何脚本加载到 `split_role == test` 的任务时直接抛错。
- 优化器不得直接调用端点，不得访问验证集与审计集；所有分数只能来自 `evaluate_batch`。
- 每一次端点调用都要有 `RunRecord`，重试也要记录；禁止静默吞掉异常或返回伪造分数。
- 温度、`max_tokens_per_node`、提示模板在一个实验内不得变化；变了就是新的 `exp`。
- 图的身份只认 `graph_hash`；任何按边列表直接比较的代码都要改成按 hash。
- 不引入付费 API；端点配置只允许本地地址与两个 flash 模型的内网地址。
- 先用 mock 端点把第 1–6 步跑通并通过单元测试，再碰真端点；真端点的第一次运行先跑 5 道题的冒烟测试。
- 每个脚本都要能从中断处续跑（靠缓存），不要实现自己的断点逻辑。
- 改任何默认参数先改 `configs/e0.yaml` 并在 PR 说明里写原因；不要在代码里写死数字。
