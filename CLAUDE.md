# topo-audit

审计 MAS 拓扑学习的平台期：把"学到的拓扑相对等算力单智能体的收益"分解为容量 C、估计 E、
经验优化 O、泛化修正 Δ_gen 四项，并给出置信区间。

完整设计见 `topo-audit-docs/`（01 提案 / 02 可行性 / 03 框架指南 / 预实验总表）。
本文件只列**已冻结的决定**、硬规则与构建顺序。

## 已冻结的决定（2026-10-13，10-18 补充）

| | 决定 | 依据 |
| --- | --- | --- |
| 模型条件（10-18 确认） | **DeepSeek-V4.1-Flash**（gateway 上的 model id `DeepSeek`），温度 0.7，每节点上限 8192 token，思考 = gateway 默认（`model.request_options: {}`）。P1 另探测过 GLM-5.3-Flash、gpt-oss-120b、Laguna-S-2.1 | P1 与真端点冒烟都在这个条件下；P1c 发现 `reasoning_effort` 可调，要做思考轴就填 `request_options` 并换 exp（它进 bind_experiment） |
| 审计空间 | S₀ = 4 节点、6 条可选前向边。**主空间允许缺输入节点**；S₀-strict（要求 Critic/Reviser 有入边）作嵌套子空间一起报 | strict 版剔掉的恰是并行投票图，而那是最可能的 G\*；两个都报则 C(S₀) − C(S₀strict) 直接量出"容量里属于并行采样的部分" |
| 空间规模 | 64 个边掩码 → **34** 张带标号可执行 DAG → **25** 种互不相同的计算。三个数都要在论文里报 | 优化器搜的是掩码；容量与估计项由计算数决定；张量按计算数付钱 |
| 缺输入的 Critic/Reviser/Finalizer | 退化成 Solver，**用 Solver 提示模板**，`effective_role` 记进日志与 hash。零边图 G_empty 的 n4 也一样（10-18） | 否则就是"批判一段空输入"；G_empty 原来拿到的是让它"reconcile the received work"的 Finalizer 模板。现在它和 π_single 的 CoT 提示逐字节相同，P(G_empty) vs P(CoT) 成了两条代码路径的一致性检查。S₀-strict 的规则不变（只管 Critic/Reviser），计数仍是 34 / 25 / 16 |
| n4 聚合规则 | `concat_rereason`：按节点序拼接入边输出后重新推理。**进 hash** | 换成多数投票会改变每张多入边图的分数，也就改变 C 的符号 |
| 单智能体参照 | π_single 的 4 种策略各报一条随预算变化的曲线（在 val 上），主数字取 S₀ token 中位数那一点：**在 val 上选策略与参数，只在 test 上评估被选中的那一个**，逐题分数写进 `frontier.json`，C 是配对差。截断规则与图相同：用到的任一调用被截断，这次策略运行记 0 分 | 提案 §3.3 要前沿不要单点；`n` 参数不可用，8 个样本得分 8 次调用；在 test 上挑最好的策略就是硬规则禁止的 argmax；截断规则不一致会让同一个计算在 C 的两边得分不同 |
| ε | **待定，不再是 2.0** | 原依据有两处错：按 n×种子 降噪，以及用 q(1−q) 当配对差的方差。噪声下界是 `1.645·√(Var(d)/n)`，**与种子数无关**；二值评分、真改善为 0 时 Var(d) = q：300 题、q=0.2 时下界 **4.25 分**（不是 3.80），排除 ε=2.0 需约 **1,350** 道验证题，300 题时能排除的最小 ε 约 4.3。Var(d) 用 P2 在 top-5 图对上实测（`scripts/p2_spread.py` 直接给出各 ε 需要的 val 题数），不能假设 |
| 种子 | **20** | 两条理由，都不是“种子能降噪”：50 个会让样本被复用 3.4 次（20 个只 1.4 次）；而平台期检验里各种子收敛到同一对图时，种子对任务噪声完全不降噪 |
| 训练集重复次数 | R = **8** | 20 种子下单个 (图,题) 约命中 11 次，复用 1.4 次；R=4 会复用 2.8 次，人为抬高种子间相关性 → 平台期偏容易被宣布 |
| 主格任务族 | **DeepMath-103K 难度 ≥8.5**（待定，见下）。MATH Level 4–5 在默认思考条件下已证伪；MMLU-Pro 降为"H1 证据格" | MATH L4–5 对 2026 思考模型饱和（单智能体 0.91–0.93，开思考测的；若改成关思考要重测），没有归因空间；DeepMath 带连续难度字段，8.5 档实测 p=0.690 且比 8.0 档更便宜。选项题上最优图几乎必然是纯投票图，C 会被 π_single 的自洽抵掉 |
| 待定 | 主格难度档最终取值 | P1b：7.0→0.769 / 8.0→0.795 / 8.5→0.690 / 9.0→0.605，工期 2.7 / 6.8 / 4.8 / 17 天。P1b 脚本默认每档 40 题；若用的是默认值，95% CI 约 ±0.14，8.0 与 8.5 的差在噪声内；且旧脚本用闭区间，难度恰为 9.0 的题同时进了 8.5 和 9.0 两档（已改半开区间）。下一步跑 P2（`scripts/p2_spread.py`，默认 25 种计算 × 150 道留出题 + π_single）：最好 − 最差 ≥5 分只是必要条件，还要看**最好 − 第五名**（top 端平则 E、O 天然很小）、**交叉拟合的 C**（对等 token 的 π_single）和 **top-5 配对差方差**（定 ε 与 val 规模） |

## 硬规则（不要违反）

- DeepMath 的 P1 / P2 / 正式划分共用 `load_band()`：剔除空答案及忽略大小写、首尾空白后的
  Yes/No/True/False。旧 P1b 数字包含判断题，不能直接替代此题池的测量；先核实过滤后题量，
  下载更多 shards，再留出 P2 并生成四角色划分。划分记录 `answer_filter`。
- P2 必须逐一验证 `(exec_hash, task_id, repeat)` 完整且唯一、分数有限；缺格时报错，
  靠缓存续跑补齐后才能排名或给出 PASS。单智能体费用按真实调用去重计入 P2 总费用，
  多个策略共用的采样只付一次；缓存重放的 fresh token 为零。
- **最终测试集上不得出现 argmax / 排序 / 阈值筛选**。选图只能在审计集上做。
  （34 张图会全部在测试集上评估过，所以"只有 report.py 能读 test"这条流程纪律不够——
  真正的风险是手滑把 `max_g P_test(g)` 当成天花板报出去。）
- `runtime/executor.py` 是唯一能调用 LLM 端点的模块（π_single 的 `SingleAgentRunner` 是唯一例外：
  它复用同一个缓存、记账与实验绑定）；`analysis/` 只读 `records.jsonl`。
- 优化器不得直接调端点，不得访问审计集 / 验证集 / 测试集；分数只能来自 `evaluate_batch`。
- **缓存与张量按 `exec_hash` 寻址，不是 `graph_hash`**；repeat 索引的命中计数也按 `exec_hash`，
  否则两个等价掩码都会取到 repeat 0，优化器会把同一个样本读成两次独立确认。
- **任务批次序列必须含搜索种子。** 跨优化器共享序列是对的（配对靠它），但跨种子共享会抹掉
  “重跑时 minibatch 也会不同”这个变异源，让 incumbent 比真实重跑更相似 → frozen 偏多、
  种子维度 CI 偏窄。见 `TaskBatches.search_seed`。
- **张量建好后，优化器扫描必须开 `replay_only`**：缓存未命中直接抛 `CacheMiss`，
  否则“零端点调用”只是口头承诺。用 `optimizers/sweep.py`，它默认用 `ReplayOnlyEndpoint`（不需要 key、连不了网）。
- **比较优化器一律先对齐到同一条预算轴**（`analysis/trajectory.py`：预算 t 的 incumbent =
  最后一个 cum_evals ≤ t 的轮次之后的 incumbent）。按 cum_evals 精确查表会静默丢掉按组评估的优化器
  （reinforce_edge 每轮 8 次，t 与 t+50 永远不同时在网格上）。覆盖不到的预算要显式标 not_covered。
- **π_single 只在 val 上选**，test 上只评估被选中的策略（`runtime/single_agent.py`）。
- 思考档位等请求附加字段只能经 `model.request_options` 传；它进 bind_experiment，改了就是新 exp。
- 建张量后看 `manifest_audit.json` 的 `truncation_by_row`：任一节点截断整次记 0，深图更吃亏；
  行间截断率差超过 `acceptance.max_row_truncation_spread` 先加 token 上限再继续。
- **优化器的搜索种子不得进缓存键。** 一个张量格子的身份是 `(exec_hash, 题, repeat)`——
  那是一次 LLM 采样，和谁来要它无关。种子只决定抽哪个 repeat（`_repeat_index`）。
  种子进了键，50 个种子里有 49 个会 miss 掉整张张量，优化器扫描要花 8.7 倍张量的算力
  （42 天 vs 4.8 天）。见 `Executor.sample_seed` 与 `tests/test_matrices.py`。
- 每一次端点调用都要有 `RunRecord`，重试也要记；禁止静默吞异常或返回伪造分数。
- 温度、`max_tokens_per_node`、提示模板、聚合规则在一个实验内不得变化；变了就是新的 `exp`。
- 图的身份只认 hash；任何按边列表直接比较的代码都要改。
- 端点只允许内网 gateway 与本地地址，**且必须绕过 clash 代理**（`http_proxy` 等会让请求 502）。
  key 从 `.env` 读，不进 config、不进代码、不进 git。
- 先用 mock 端点把第 1–6 步跑通并过单测，再碰真端点；真端点第一次先跑 5 道题冒烟。
- 每个脚本都要能靠缓存从中断处续跑，不要自己实现断点逻辑。
- 改默认参数先改 `configs/e0.yaml` 并在 PR 里写原因；不要在代码里写死数字。
- `analysis/decompose.py` 必须对每一行断言
  `P_test(G_learned) − P_test(pi_single) = C − E − O − Δ_gen`（容差 1e-9）。

## 符号（冻结）

`G_star` 总体最优图 · `G_hat_N` 训练集经验最优图 · `G_learned` 优化器在预算 t 的输出
· `G_empty` 零边图（只有 n4，空间内的单 agent 基线） · `pi_single` 声明策略集合的最优者

## 构建顺序与现状（10-18）

七步都已实现并有 mock 单测（`python3 -m pytest tests/ -q`，默认跳过 slow）；真端点只跑过冒烟（250 格）。

1. ✅ `space/` — Topology、剪枝、规范化、双 hash、S₀/S₁ 枚举。34 / 16 / 422 对上闭式；计算数 25 / 298。
2. ✅ `data/` — MATH L4–5 与 DeepMath（单个难度档）的加载与打分、固定种子四角色划分（`make_splits.py --family deepmath --band B --exclude-p2 …`）；`data/tasks.py` 供所有驱动共用。
3. ✅ `runtime/` — 端点客户端（重试、`request_options`、`ReplayOnlyEndpoint`）、执行器、缓存、token 记账。
4. ✅ `runtime/single_agent.py` — π_single 四种策略 × 预算曲线；`main()` 在 val 上选、test 上报。
5. ✅ `optimizers/` — 随机搜索、边级 REINFORCE/GRPO、prune_from_dense、贪心局部搜索；`optimizers/sweep.py` 只重放。
6. ✅ `audit/` + `analysis/` — 张量、四项分解、交叉拟合、配对 bootstrap、平台期检验、`prescribe()` 决策。
7. 真端点（按顺序，每步靠缓存可续跑）：
   1. `python3 scripts/p2_spread.py --pool deepmath --band 8.5`（或 `--pool math`）→ 定主格、ε、val 规模
   2. `python3 audit/run_audit.py --config configs/e0.yaml` → 张量 + `manifest_audit.json`
   3. `python3 runtime/single_agent.py --config configs/e0.yaml` → `pi_single/frontier.json`
   4. `python3 optimizers/sweep.py --config configs/e0.yaml` → 轨迹（只重放）
   5. `python3 analysis/report.py --config configs/e0.yaml` → `report.json` + `decomposition_rows.jsonl`

## 目录

```text
configs/  data/{loaders,splits}/  space/  runtime/  optimizers/  audit/  analysis/
scripts/  tests/  prompts/  runs/  topo-audit-docs/
```
