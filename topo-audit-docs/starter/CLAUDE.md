# topo-audit

审计 MAS 拓扑学习的平台期：把"学到的拓扑相对等算力单智能体的收益"分解为容量 C、估计 E、经验优化 O、泛化修正 Δ_gen 四项，并给出置信区间。完整设计见 `docs/03-experiment-framework-guide.md`，本文件只列硬规则与构建顺序。

## 硬规则（不要违反）

- 最终测试集只能被 `analysis/report.py` 读取；其他任何脚本加载到 `split_role == test` 的任务时直接抛错。
- 优化器不得直接调用端点，不得访问验证集与审计集；所有分数只能来自 `evaluate_batch`。
- `runtime/executor.py` 是唯一能调用 LLM 端点的模块；`analysis/` 只读 `records.jsonl`。
- 每一次端点调用都要有 `RunRecord`，重试也要记录；禁止静默吞掉异常或返回伪造分数。
- 温度、`max_tokens_per_node`、提示模板在一个实验内不得变化；变了就是新的 `exp`。
- 图的身份只认 `graph_hash`；任何按边列表直接比较的代码都要改成按 hash。
- 不引入付费 API；端点配置只允许本地地址与内网 flash 模型地址。
- 先用 mock 端点把第 1–6 步跑通并通过单元测试，再碰真端点；真端点第一次运行先跑 5 道题的冒烟测试。
- 每个脚本都要能从中断处续跑（靠缓存），不要实现自己的断点逻辑。
- 改任何默认参数先改 `configs/e0.yaml` 并在 PR 说明里写原因；不要在代码里写死数字。
- `analysis/decompose.py` 必须对每一行断言 `P_test(G_O) − P_test(π*_single) = C − E − O − Δ_gen`（容差 1e-9）。

## 构建顺序（一步一个 PR，先过验收）

1. `space/` — Topology、剪枝、规范化、hash、S₀ 枚举。验收：64 个边子集去重后打印可执行 DAG 数；同一张图任意边序 hash 相同。
2. `data/` — MATH 加载与打分、固定种子四角色划分。验收：四角色零重叠、重跑逐字节一致、20 道人工核对样例打分全对。
3. `runtime/` — 端点客户端、执行器、缓存、token 记账。验收：mock 端点跑通 S₀ 全图 × 5 题；token 对账一致；重跑零调用。
4. `runtime/single_agent.py` — Π_single 四种策略，同预算上限。
5. `optimizers/` — 接口 + 随机搜索 + GRPO 式。验收：预算计数一致、轨迹可重放、mock 下 5 种子把 incumbent 推进真实前三。
6. `audit/` + `analysis/` — 全量评估、四项分解、交叉拟合、bootstrap、平台期检验。验收：恒等式断言通过；统计函数有单元测试。
7. `scripts/e0_*.sh` + 真端点 — 先跑 1 小时吞吐基准，再跑 E0 全量。

## 目录

```text
configs/  data/{loaders,splits}/  space/  runtime/  optimizers/  audit/  analysis/  scripts/  tests/  runs/
```
