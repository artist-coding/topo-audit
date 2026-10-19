# topo-audit

DeepMath 的 P1、P2 和正式划分现统一剔除空答案及 Yes/No/True/False 判断题。
下文旧 P1b 的准确率和工期来自过滤前题池，选主格前须重新测量并检查剩余题量。
P2 遇到缺失评估会停止并要求缓存续跑；其总费用包含图评估和去重后的单智能体调用。

把"学到的 MAS 拓扑相对等算力单智能体的收益"分解成四项并给出置信区间的实验框架。

论文设计见 [`topo-audit-docs/`](topo-audit-docs/)；已冻结的决定与硬规则见
[`CLAUDE.md`](CLAUDE.md)。本文件是给审阅者的入口。

## 要测的是什么

一条学习曲线走平了，有四种可能的原因，只有第一种是"框架到顶了"：

```
P(G_star)   ─── 空间里真正最优的拓扑          ← 天花板
P(G_hat_N)  ─── 按训练题挑出来的最优拓扑
P(G_learned)─── 优化器交出来的拓扑            ← 曲线停在这里
P(pi_single)─── 等 token 预算下最强的单智能体
```

四级阶梯，三个相邻间隔各对应一个动作：

| 间隔 | 等于 | 决策 |
| --- | --- | --- |
| P(G_star) − P(G_hat_N) | 估计项 E | 加任务 |
| P(G_hat_N) − P(G_learned) | **O + Δ_gen** | 继续搜 |
| P(G_star) − P(pi_single) | 容量 C | 扩空间 / 停 |

恒等式 `P(G_learned) − P(pi_single) = C − E − O − Δ_gen` 是伸缩和，无需假设。
`O + Δ_gen` 把训练目标上的量 O 换算成总体量，所以 Δ_gen 不是第四个待解释的东西，
而是汇率。详见 [`analysis/decompose.py`](analysis/decompose.py) 的模块文档。

## 审计空间 S₀

4 个节点固定顺序执行（Solver → Critic → Reviser → Finalizer），6 条可选前向边。
三个数都要在论文里报，因为它们各有用处：

| 数 | 含义 | 决定什么 |
| --- | --- | --- |
| **64** | 边掩码 | 优化器实际搜索的空间，决定搜索难度 |
| **34** | 带标号的可执行 DAG | 空间的名义大小 |
| **25** | 互不相同的计算 | 容量与估计项里的 log∣S∣；**张量按这个付钱** |

34 → 25 的塌缩来自"缺输入的 Critic 退化成 Solver 并用 Solver 提示"：
`n1→n3 n3→n4` 和 `n2→n3 n3→n4` 发出的是逐字节相同的请求。

闭式校验：`f(k) = Σ_m C(k−1,m)·Π_{j=1..m}(2^j−1)` 给出 34 / 422 / 11586（k=4/5/6）。

## 架构

```
数据 → 四角色划分 → 图空间 S₀ + π_single → 执行器（唯一端点出口）→ 缓存 → 分析
```

| 模块 | 职责 |
| --- | --- |
| [`space/`](space/) | Topology、剪枝、规范化、双身份 hash、S₀/S₁ 枚举 |
| [`data/`](data/) | MATH / DeepMath 加载、boxed 打分、固定种子四角色划分 |
| [`runtime/`](runtime/) | 端点客户端、执行器、SQLite 缓存、token 记账、π_single |
| [`optimizers/`](optimizers/) | 随机搜索、边级 REINFORCE、prune_from_dense、贪心局部搜索 |
| [`audit/`](audit/) | 建得分张量 |
| [`analysis/`](analysis/) | 四项分解、交叉拟合、bootstrap、平台期检验、报告 |

**两种图身份。** `graph_hash` 认带标号的图（优化器选的那个掩码），
`exec_hash` 认计算（缓存与张量按它寻址）。`space_id` 是 provenance 不是 identity，
所以一张张量同时服务 S₀ 和嵌套的 S₀-strict。

**张量一次建好，之后全是重放。** 25 行 ×（train 256×8 + audit 300×2 + val 300×2
+ test 1000）= 106,200 格 / 382,320 次节点调用。建好之后任意优化器、种子、预算、
N 值都是离线查表，零端点调用（`optimizers/sweep.py` 用 `ReplayOnlyEndpoint`，缺格直接报错）。
种子设为 20：重放虽然免费，但种子共享张量格子，50 个种子会让样本被复用 3.4 次；
而且收敛后各种子持有同一对图，种子对平台期检验的任务噪声没有帮助（见 CLAUDE.md）。

## 审阅时值得盯的几处

| 位置 | 为什么 |
| --- | --- |
| `analysis/decompose.py` 的 `check()` 文档 | 恒等式断言是**结构性成立**的（50 组随机数验过），只能抓公式回归，抓不住坏数据。真正的隐患是混用 P(G_star) 的两个估计量，由 `mixed_star_residual()` 量化 |
| `analysis/decompose.py` 的 plug-in vs 交叉拟合 | 两个估计量回答不同问题，故意分开：plug-in（audit 选图、test 报分）是 headline；交叉拟合估的是"选择流程"的期望，它与朴素 max 的差才是 winner's curse |
| `analysis/plateau.py` 的 frozen vs moved_flat | 把"incumbent 冻住"和"变了没涨"分开报。合并它们，就是把搜索卡住写成天花板 |
| `analysis/matrices.py` | 没跑到的格子保持 NaN。0 的含义是"答错了"，和"没跑"是两个断言 |
| `runtime/executor.py` 的 `render_inputs()` | 无入边节点不能拿到一个空的 "Received work:" 标题——否则投票图里的裸 solver 和 π_single 的 CoT 吃残缺提示，而 C 正是这两者的差 |
| `optimizers/base.py` 的预算定义 | 缓存命中和等价掩码仍然要花掉一次候选评估及其 token，否则重放过的搜索会获得免费算力 |
| `prompts/` | 四个角色模板 + selfcheck，E0 内不优化。改它们会触发 `bind_experiment` 报错（变了就是新的 exp） |
| `analysis/trajectory.py` | 各优化器每轮扣的评估数不同（1 / 8 / 5–7），先对齐到同一条预算轴再比较；按 cum_evals 精确查表曾让 reinforce_edge 从平台期检验里整个消失 |
| `runtime/single_agent.py` 的 `select_on_val_report_on_test` | π_single 在 val 上选、test 上只评估被选中的那一个，逐题分数让 C 成为配对差；截断规则与图一致 |
| `scripts/p2_spread.py` | 建张量前的门：看最好 − 第五名、交叉拟合的 C、top-5 配对差方差，而不只是最好 − 最差 |

## 已实测的端点约束

在 gateway（DeepSeek = ds-v4.1-flash）上测到的，都已写进代码：

- **必须绕过 clash 代理**，否则 502。而 HuggingFace 必须走代理——两边相反。
- **`n > 1` 被拒**，所以 k 个样本 = k 次调用。
- **思考默认开**，`thinking.type=disabled` 无效；P1c 发现 `reasoning_effort`（none / low /
  high / xhigh / max）能调。E0 用默认，经 `model.request_options: {}` 显式记录；改了它就进
  `bind_experiment`，必须换 exp。
- **`max_tokens=1024` 会把实验做废**：思考链吃光额度，`content` 返回空，记 0 分。
  而 4 节点图有 4 次机会撞上这个上限，单节点只有 1 次——会人为造出"人多更差"。
  实测 8192 下截断率 3%，真端点冒烟 0%。
- **聚合吞吐在 ~1000–1200 生成 token/s 饱和，与并发数无关。** 并发 8 和 32 总量相同，
  64 开始排队，128 触发 429。所以工期由"要生成多少 token"唯一决定。

## 怎么跑

```bash
cp .env.example .env            # 填 gateway 的 key
python3 -m pip install -r requirements.txt
python3 -m pytest tests/ -q     # 默认跳过 slow；全量用 -m slow

python3 space/s0.py --nodes 4 --enumerate      # 应打印 34 / 25 / strict 16
python3 data/fetch_math.py                     # 需要代理；DeepMath 用 data/fetch_deepmath.py
python3 data/make_splits.py                    # MATH L4-5；DeepMath: --family deepmath --band 8.5

# E0，按顺序，每步靠缓存可续跑
python3 scripts/p2_spread.py --pool deepmath --band 8.5    # 门：主格、ε、val 规模
python3 audit/run_audit.py --config configs/e0.yaml --dry-run
python3 audit/run_audit.py --exp e0_smoke --splits val --limit-tasks 5   # 真端点冒烟
python3 audit/run_audit.py --config configs/e0.yaml        # 张量 + manifest_audit.json
python3 runtime/single_agent.py --config configs/e0.yaml   # π_single：val 上选，test 上报
python3 optimizers/sweep.py --config configs/e0.yaml       # 只重放
python3 analysis/report.py --config configs/e0.yaml        # report.json + decomposition_rows.jsonl
```

## 现状

七步都有驱动和 mock 单测；真端点只跑过冒烟（250 格、0 失败、0% 截断）。模型条件已确认：
DeepSeek-V4.1-Flash、温度 0.7、每节点 8192 token、思考用 gateway 默认。

**悬着的一个决定：主格用哪个难度档。** MATH Level 4–5 对 2026 的思考模型已饱和
（单智能体 0.91–0.93），所以换成带连续难度字段的 DeepMath-103K。实测：

| 难度档 | 单智能体 p | 生成长度中位数 | 全量张量工期 |
| --- | --- | --- | --- |
| 7.0 | 0.769 | 1001 | 2.7 天 |
| 8.0 | 0.795 | 2492 | 6.8 天 |
| **8.5** | **0.690** | **1769** | **4.8 天** |
| 9.0 | 0.605 | 6316 | 17 天 |

注意 P1b 脚本默认每档 40 题；若当时用的是默认值，95% CI 约 ±0.14，8.0 与 8.5 的差在噪声内；旧脚本的闭区间还让难度恰为
9.0 的题同时进了两档（已改为半开区间）。下一步跑 P2（`scripts/p2_spread.py`）再决定要不要付全量
工期——决定实验成败的不是题有多难，而是 top 端的拓扑分不分得开、最好的图能不能赢过等 token
的 π_single，以及 val 能不能分辨 ε。
