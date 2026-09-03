# SteinLib/DIMACS 获取与发布政策

状态：生效；公开发布前复核许可

机器可读入口：`configs/steiner/data_provenance_v1.yml`

SteinLib 和 DIMACS 数据只从配置中登记的官方 HTTPS 入口下载。下载后先验证
SHA-256，再允许进入预注册 split 对应的流程；archive 或 member hash 不一致时
必须 hard fail，不得换镜像、删实例或修改 selector 来继续运行。

Git 只保存官方 URL、source revision/selector、archive 与逐 member checksum、
下载/验证脚本、引用信息和当前许可状态。raw archive、解压实例以及派生的大型
数据继续放在 `data/steiner/raw/` 或 ignored artifact 目录，不提交 Git。

目前没有确认 SteinLib/DIMACS 对 raw 数据的显式再分发许可。因此在权限得到
书面或官方条款确认前，任何公开仓库、GitHub Release、论文附件、容器镜像或
模型发布包都不得携带这些 raw bytes。checksum 只能证明内容身份，不构成许可。
S13 公开发布检查必须二选一：取得并记录明确再分发许可，或发布可复现下载脚本
和 checksum manifest、让使用者自行从官方源获取数据。

这项许可待办不阻止当前服务器上的内部研究，但它是“把 raw 数据公开打包”的
blocker。final-test 的封存、禁止调参和首次运行阶段仍由原 selector manifest
控制，不因许可处理而改变。
