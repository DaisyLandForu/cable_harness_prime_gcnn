# S05 formal-v3 result — GPT audit request

请只读审计 S05 formal-v3 confirmatory result，不修改仓库，不访问 test/final，
也不要把 formal-v1/v2 的历史 FAIL 改写成 PASS。

## 固定对象

- branch：`research/steiner-migration`
- v3 preregistration base：`55833c54d02f3e5bcbc4bd2e19d95f6f3f019293`
- v3 protocol content head：`8d7accd2a948935174d3113f8787e58dae7936b3`
- activation commit：`9122b123a2b036cc83a7b83e45a4ffbd1883842c`
- implementation content head：`42faf09560ecbfb782df7c89fa068e9f575c7e4b`
- committed selection-seal head：`fab9b720e513da66294e95dba0511a5bfa32b9a3`
- result content head：`6cf7acab57525a744233ed3fdfd463f00fcd470c`
- implementation range：
  `9122b123a2b036cc83a7b83e45a4ffbd1883842c..42faf09560ecbfb782df7c89fa068e9f575c7e4b`
- result evidence range：
  `fab9b720e513da66294e95dba0511a5bfa32b9a3..6cf7acab57525a744233ed3fdfd463f00fcd470c`
- audit entry：`docs/steiner/phases/S05/S05_AUDIT_PACKET.md`

## 固定 Git 文件和 hashes

- protocol YAML：`99e75a4d4fa69f805232c637d4fc0ae750fcfccb57e9979b561f422591006242`
- candidate identities：`e65fc9a03fd683277570befe13b11f4b15ea981ee427568d56aeeccdd3847b56`
- protocol explanation：`c61b58d23f4de6bb2709aa9c40ccab11d6db4507a6bd1c9d2e544eb68595d873`
- activation record：`e1d9a4f16c3649d2736149f48d95b834d9b5ab56fb798e429b2ee4591a8b79d8`
- implementation explanation：`e107bfa5a83ad816da1c05aa1cc444e6384fd064e049f8045c2328c445edf90d`
- selection seal：`10283ee44681a547ffb61373f30325f51888e9bd17ff98980abb4265f862f56d`
- committed Gate summary：`4aa035ad7538219da72fa4a8028223cf340181d5d1495539a051acf7601f742d`
- result analysis：`d2a6be192eb09218ea837f704ed78e17954ba7834c60c6d2511fdda7a9d3ca14`
- test report：`f560e895b9dd19f3b1bc207e80c622202743bc19ff3f2e8c7958ad59ce8d7610`
- audit packet at result content head：
  `d376e879de7fe9df5713de7dc6b2acdf0cd669eab08fc2e680dd3ab9cafece90`

## Hash-bound runtime evidence

大体积 raw shards、完整 manifests、GPU reports 和 checkpoints 按研究契约不提交
Git；以下身份由 committed selection seal、Gate summary 和审计包登记：

- teacher manifest：
  `0dfc4b3cc6fc28192c7477b9f209aab6b6a0c7bc373dba2e491e6c677df8755e`
- selected manifest：
  `0a95f47337338ab58891056f773a99467da2b327b714d1a2cd4ee5e96c3b6981`
- five evaluation reports：
  - 101 `c21595ffdd70b406c076e4172b9b23491310d13ec7206f31dd936e027c117826`
  - 202 `c52d8a26a2c6deaf07f21b1dd292117973d1ea5b5b2b57dc35759de336cb80f7`
  - 303 `a49e1cdd494b0d84d743be25f40d3f05e6620ba40605fb9e1c5206f2eaf46c71`
  - 404 `7f6cfb3bb5c1bd4eb76c8c0fedd88eaf6682ede296dc5ed7d8fb8478eba8392f`
  - 505 `d361a974c920ae307a27d92dd3a0fe0af23f75a0bf5731a7012ae3cfc1b3ea2e`
- aggregate runtime report / committed Gate summary bytes：
  `4aa035ad7538219da72fa4a8028223cf340181d5d1495539a051acf7601f742d`

请明确区分“可从 Git 静态核验”与“只能由 committed hash-bound runtime evidence
支持”的结论，不要声称重新运行了未实际运行的实验。

## 已登记结果

- teacher：240/240 terminal（237 completed、3 root_solved），零失败；
  3,383/3,840 valid；13/3,383 all-tie；69,709/69,709 mapped；每 family
  eligible lineage 至少 14；teacher Gate PASS。
- pre-model barrier：30 unique lineages、6/family、320 states、64/family、每
  lineage 有状态、semantic unique、零 prior/old-v2/cross-role/test-final；PASS。
- 五个冻结 checkpoints：5/5 completed；没有 retraining；repeat inference 和
  reload max error 对五 seed 都为 0.0。
- primary mean effect `random_regret - model_regret = 0.11269713933795965`；
  30-lineage paired bootstrap 10,000 次、seed 20260902、95% CI
  `[0.05781044428556909, 0.16788781910680703]`。
- lineage-weighted mean model regret `0.5733266530445258`，random
  `0.6860237923824855`；seed CV `0.016792729371067443 <= 0.15`。
- 五 seed effect 全正；五 family aggregate effect 全正：sparse 0.108978、
  geometric 0.082680、grid 0.108814、community 0.165791、bridge 0.097222。
- 30 个单图中 22 个 effect 正、8 个负，完整保留；family Wilcoxon/Holm 是
  supplemental、非 Gate，不能误写成每 family 单独显著。
- committed summary 的 `all_five_training_seeds_complete` 是复用统计器的历史
  字段名；v3 的实际语义是五个 frozen model-seed checkpoints 全部完成评估，v3
  没有训练。
- local scientific Gate：9/9 true，PASS；`s06_authorized=false`；test/final
  access=false。

## 请重点判断

1. v3 是否严格只修复 fresh validation denominator，保持五个 v2 模型 byte-frozen，
   没有根据 v3 结果重训、换 checkpoint 或调 Gate；
2. activation 是否在 collection 前存在，selection seal 是否在任何 checkpoint
   load 前提交，生产 evaluator 是否真正由 committed byte-exact seal fail closed；
3. teacher 60%/40%/100%、零 task failure 和每 family 至少六 eligible lineage
   是否按冻结 denominator 计算且全部满足；
4. first-six teacher-validity-only selection 是否独立于 logits/regret/teacher score
   magnitude，并形成精确 30×5-seed paired matrix；
5. 统计是否以 30 个 base graph lineage 为 bootstrap unit，而非把 320 states 当
   独立样本；CI、all-seed、all-family 和 CV Gate 是否实现正确；
6. 8 个负 graph effects 与 non-significant family supplemental tests 是否被诚实
   保留，同时不错误否定预注册的 aggregate Gate；
7. 结论是否只限 teacher-evaluable small/medium envelope，不外推 online solve
   time/node、every-scale 或 final test；
8. formal-v1/v2 FAIL 是否原样保留，S06/test/final 是否在本次结果审计前保持禁止。

## 验证与运行插曲

- implementation targeted：13 passed；结果后 targeted：13 passed；
- complete frozen-stack suite：103 passed、1 expected PACE skip；
- selection-seal runtime reload、SCIP 8.0.4 wrapper、compileall、shell syntax、
  `git diff --check`：PASS；
- 自动 watcher 在 teacher/barrier PASS 后因失效 localhost Git proxy 停在 push，
  未启动 GPU；seal 随后以同一 commit 成功 push；
- aggregate watcher 曾在 seed 505 report 仍为 `running` 时安全停止；待 5/5
  completed 后只运行一次正式 aggregation。两次调度插曲都没有改变 inputs、
  outputs、seed、checkpoint 或 Gate。

请返回 `PASS`、`CONDITIONAL PASS` 或 `FAIL` 并逐项列出 blocking findings。
只有结果审计 `PASS` 才允许创建 S05 audited tag，并开始 S06 实现；这不授权提前
访问后续冻结 final test。
