# 注意点与已知陷阱（冻结基线之上）

## 目标错位

- 训练 reward：`negative_node_increment` ≈ `-NNodes`
- 选模：`validation_nodes` 最小
- 评测：wall time / gap
- `real_04` 已证明：节点少可以 gap 更差

→ C5 之前禁止再开以节点数为唯一目标的大训。

## Reward loophole

`bootstrap_on_truncation=false` + `time_limit`/`node_limit` 截断时，可能奖励“节点少但每个节点很贵”的策略。必须在 C5 修复 censored episode。

## Prim-A 真因未拆开

Empty \(S\) 时所有 z 得 +0.5 → **变量族先验**，不是连通扩张。  
gated（depth≥1 + require grown）削弱 `real_01`，强烈暗示 root z-bias。

C0.2 必须回答：`root-z-bias ≈ full-prim on real_01`？

## Prim 几何过粗

- \(S\) = 活跃边端点 union ≠ root-connected component
- `both_in` ≠ cycle
- 0.5 hard threshold 使 0.49/0.51 类别翻转
- 固定 λ=0.5 相对 Q 尺度不可迁移

## 全树接管

`kRlBranchrulePriority = 1000000`，`max_depth=-1` → GCNN 几乎全面覆盖 relpscost。  
C4 才改 ownership；C0 只诊断，不立刻改默认部署策略。

## Train/serve parity

- Python grown：主要 `LP > 0.5`
- C++ grown：`lb > 0.5 || LP > 0.5`

C0.3 必须对齐并加自动化测试。

## 评估协议污染

`real_09` / 部分 transfer 已用于 λ 扫描 → 只能标 exploratory，不能当 untouched test。

## 数据与优化

- B：仅 real_06/07，3000 steps，`updates_per_env_step=8`，epsilon 按 gradient 衰减 → 有效交互极少
- normalizer warmup=8 过激进
- GCNN 仅 1 轮 mean message passing，难捕获长程连通

## Frozen baselines（勿删除）

- P0 / P1 结果与报告
- Phase A / A-extend / B 结果与模型
- 旧 GCNN TorchScript
