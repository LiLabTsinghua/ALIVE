# ALIVE 发布前审查清单

更新时间：2026-08-17

本文记录 `alive/` 发布前审查发现的问题。后续按编号逐项确认；未经确认，不修改对应源码。

状态定义：

- `待确认`：等待决定是否修改。
- `已确认`：已决定处理，但尚未完成。
- `已完成`：代码和测试均已完成。
- `保留现状`：明确决定不修改。

## P0：指标与结果可信度

### P0-1 空结果被记录为全零

- 状态：已完成
- 位置：`alive/base.py`、`alive/metrics.py`
- 现状：无预测、无共同 perturbation 或空 gene set 时返回全零指标。
- 风险：`MSE=0`、`MAE=0` 会被误解为最佳结果。
- 处理结果：无有效评测对象时输出 `NaN`；真正可计算但相关性未定义的零方差向量仍按既定规则返回 0。CSV 保留完整指标行。

### P0-2 离线重建 coverage 与运行时可能不一致

- 状态：已完成
- 位置：`alive/tools/rebuild_runtime_summary.py`
- 现状：存在 condition record 就计算 coverage；离线 pool 使用 true bulk 中全部非 control perturbation。
- 风险：非 ActiveLearning 任务也可能被追加 coverage；validation perturbation 可能被错误纳入 pool。
- 处理结果：确定不再报告 `coverage_max` 和 `coverage_mean`；已删除运行时计算、离线重建逻辑及相关测试，仅保留统一的五个预测指标。

### P0-3 padding 顺序不稳定且 mode 校验可被绕过

- 状态：已完成
- 位置：`alive/padding.py`
- 现状：通过 set 差集产生缺失 perturbation；无缺失项时会在校验 mode 前返回。
- 风险：不同 `PYTHONHASHSEED` 下 single-cell padding 可能不完全复现；非法 mode 可能被静默接受。
- 处理结果：先严格校验 mode；目标 perturbation 统一转为字符串，缺失项按稳定排序处理，并补充顺序与非法 mode 测试。

### P0-4 ActiveLearning 策略名称与实际语义不完全一致

- 状态：已完成
- 位置：`alive/tasks/active_learning.py`
- 子问题：
  - `random_sampling` 优先 model-visible candidates，并非全候选池均匀随机。已确认这是正式应用语义：优先保证采样结果能用于下一轮模型训练，保留现有实现并修正注释。
  - `representative_sampling` 没有把已训练 perturbation 作为已有 medoid。已确认保留现状：representative 只代表当前候选分布，历史覆盖仅由 diversity 考虑。
  - `diversity_sampling` 第一次选择使用向量模长，后续使用 `1 - PCC`。已确认保留现状：以预测效应最强的点作为初始锚点，再按相关距离扩展多样性。
- 风险：baseline 公平性、方法名称和论文描述可能不一致。
- 处理结果：三个策略定义均已确认；保留现有算法，仅修正 `random_sampling` 注释以准确表达 model-visible 优先语义。

### P0-5 InDistribution 首轮 result budget 可能不准确

- 状态：已完成
- 位置：`alive/tasks/in_distribution.py`
- 现状：样本数向下取整后仍返回配置的 target budget。
- 示例：pool 为 3、target 为 50% 时实际选择 1 个，即 33.3%，当前记录为 50%。
- 处理结果：统一按实际选择数量计算 result budget；默认允许最多 `5.0` 个百分点误差，可通过 `max_budget_gap` 配置。InDistribution 和 ActiveLearning 所有采样轮次超过阈值均明确报错。

### P0-6 PyG 图缓存没有数据指纹

- 状态：保留现状
- 位置：`alive/utils.py`
- 现状：固定读取 `cell_graphs.pkl`，不校验数据集、gene 顺序、perturbation 集合或预处理配置。
- 风险：复用缓存目录时可能静默加载旧数据，产生错误结果。
- 决定：本次发布暂不修改，保留为已知风险。

## P1：公开名称与输出协议

### P1-1 配置和对象名称不统一

- 状态：已完成
- 候选修改：
  - `objection` -> `task`：已完成一次性硬迁移；核心和 baseline 仅接受 `task`，旧字段出现时明确报错，不保留兼容读取逻辑。
  - `modelwrapper` -> `model_wrapper`：已完成一次性硬迁移；核心、测试和全部 baseline 已统一使用 `model_wrapper`、`model_wrapper_factory`、`build_model_wrapper()`，不保留旧参数、旧属性或旧工厂名称。
  - `PertData_Simple` -> `PertDataSimple`：已完成一次性硬迁移；实现已移至 `baseline/perturbation_generalization/pert_data.py`，GEARS、scGPT 和 pseudobulk 适配器均使用新名称，核心不再定义或导出该类。
- 风险：直接修改会影响 baseline，需一次性同步或提供兼容期。

### P1-2 CSV 列名语义不明确

- 状态：保留现状
- 候选修改：
  - `gene` -> `gene_set`
  - `set` -> `split`
  - `x` -> `budget`
- 风险：会影响现有结果分析脚本与历史 CSV 兼容性。
- 决定：本次发布不修改，继续使用 `x`、`set`、`gene`。

### P1-3 任务名称存在两套体系

- 状态：已完成
- 核心名称：`InDistribution`、`OutOfDistribution`、`ActiveLearning`
- 历史 baseline 名称：`extrapolation`、`extrapolation_cluster`、`active_extrapolation`
- 处理结果：`InDistribution`、`OutOfDistribution`、`ActiveLearning` 已成为唯一正式术语；核心和 baseline 的模块名、`task` 值、结果路径及错误文案均统一为 `in_distribution`、`out_of_distribution`、`active_learning` 对应体系。
- 处理进展：核心控制台 `goal_type` 和 OutOfDistribution 错误文案已统一为正式名称，不再显示 `In-distribution`、`Out-of-distribution`、`Active learning` 或 `Cluster extrapolation`。
- 测试整理：核心任务测试文件已改为 `test_tasks_smoke.py`，fixture 和测试函数统一使用 `in_distribution`、`out_of_distribution`、`active_learning` 命名；仅用于验证拒绝旧输入的测试保留旧术语字面量。

### P1-4 指标术语和旧文案需最终确认

- 状态：已完成
- 现状：正式指标暂定为 `Systema Correlation`、`Systema Bias`；启动摘要仍写 `higher PCC is better`。
- 决定：`Systema` 是最终正式术语，固定使用 `Systema Correlation` 和 `Systema Bias`，不改为 `Systematic`。
- 处理结果：已更新 Goal 文案，明确 MSE/MAE 越低越好、Correlation/Systema Correlation 越高越好、Systema Bias 越接近零越好。

## P2：健壮性与接口约束

### P2-1 CSV schema 校验依赖字典插入顺序

- 状态：已完成
- 位置：`alive/benchmark.py`
- 现状：字段集合正确但插入顺序不同也会报错。
- 处理结果：逐行严格校验字段集合，允许任意插入顺序；写出前统一按标准 schema 重排。缺字段和多字段仍明确报错。

### P2-2 config 缺少集中校验

- 状态：已完成
- 范围：`repeat`、`seed`、`use_support_data`、task、model wrapper 接口。
- 风险：错误通常在运行中后期才以 `KeyError` 或 `AttributeError` 暴露。
- 处理结果：runtime config 构建时集中校验必需字段、正式 task、各 task 允许的 strategy、字符串字段、正整数 repeat、整数 seed、布尔 use_support_data 和非空 budget plan；wrapper 创建后校验 `fit()`、`predict()` 和可写 `config`。

### P2-3 budget 和数据 split 校验不完整

- 状态：已完成
- 现状：布尔值会被当作数值 budget；atlas split 的合法取值没有完整校验。
- 处理进展：已在 runtime config 和任务校验两层明确拒绝布尔 budget。
- 处理结果：原始 atlas 的 `adata.obs['split']` 仅允许 `pool`、`control` 和 `val`；ALIVE 根据 `pool` 在每轮动态生成 `train` / `test`。缺失、其他值，或 `split == control` 与 control perturbation 不一致时，会在构建数据视图前明确报错。

### P2-4 缺少 prediction schema 校验

- 状态：已完成
- 范围：预测 `AnnData` 的 perturbation 列、gene 维度、gene 顺序、重复 perturbation 和额外 perturbation。
- 处理结果：模型预测进入 acquisition、padding 或指标计算前，统一校验 `AnnData` 类型、`obs['perturbation']`、非空字符串标签、唯一且与 atlas 顺序完全一致的 genes；同一 perturbation 允许对应多个细胞，缺失目标仍由 padding 补齐，但本轮未请求的额外 perturbation 会明确报错。

### P2-5 空训练 DataLoader 风险

- 状态：已完成
- 位置：`alive/utils.py`
- 现状：训练 loader 固定 `drop_last=True`。
- 风险：训练样本数小于 batch size 时 loader 可能为空。
- 处理结果：训练图数量不少于 batch size 时继续丢弃末尾残缺 batch；不足一个 batch 但非空时保留小 batch；训练图为 0 时抛出明确错误。

### P2-6 clipping 评测协议需要确认

- 状态：保留现状
- 位置：`alive/base.py`、`alive/padding.py`
- 现状：所有负预测在评测前统一截断为 0。
- 风险：若模型输出标准化表达或允许负值，会改变模型预测语义。
- 决定：负值截断是 ALIVE 的统一正式评测协议，不增加配置开关；模型应返回与 atlas 同尺度的非负绝对表达量，负预测在计算指标和保存结果前统一截断为 0。

### P2-7 coverage 空集合语义不合理

- 状态：已完成
- 现状：未选择任何 perturbation 和已选择全部 perturbation 都返回 coverage 0。
- 风险：空选择可能被误解为最佳 coverage。
- 处理结果：coverage 指标已从公开结果和实现中移除，此问题不再适用。

### P2-8 大候选池的二次复杂度

- 状态：已完成
- 位置：`representative_sampling`。
- 现状：构造完整两两距离矩阵，时间和空间复杂度为 O(n^2)；coverage 已删除，不再涉及。
- 处理结果：新增正整数配置 `max_representative_candidates`，默认 5000；仅在 representative 策略构造距离矩阵前检查，超过上限时明确报错并提示显式调高或更换策略。

### P2-9 baseline 运行路径不可移植

- 状态：已完成
- 现状：baseline 入口曾写死个人数据目录、结果目录和 GPU 编号；GenePert 与 scLambda 还写死 gene embedding 的绝对路径。
- 处理结果：所有入口统一通过 `baseline/_runtime.py` 读取 `ALIVE_DATA_DIR`、`ALIVE_RESULTS_DIR` 和 `ALIVE_GENE_EMBEDDING_PATH`，未配置时使用仓库相对默认路径；GPU 选择完全交给运行环境。GenePert 和 scLambda 将 embedding 路径保存在 `model_config` 中，并在加载前明确校验类型和文件存在性。

## P3：冗余与代码边界

### P3-1 未使用的 validation 状态

- 状态：已完成
- 位置：`alive/base.py`
- 现状：`self.val` 和 `self.adata_val` 初始化后没有消费方。
- 处理结果：删除两个冗余属性；validation 行仍保留在 `self.adata_model`，每轮 split 仍将其标记为 `val`，供 baseline 的 validation loader 和早停使用。

### P3-2 evaluate 吞掉未使用参数

- 状态：已完成
- 现状：`run()` 传入 `split_key`，`evaluate()` 通过 `**kwargs` 静默忽略。
- 处理结果：删除无用的 `split_key` 传参和 `evaluate()` 的 `**kwargs`；不支持或拼错的参数现在会由 Python 直接报错。

### P3-3 无效的内存释放语句

- 状态：已完成
- 位置：`alive/benchmark.py`
- 现状：创建 benchmark 后执行 `del adata`，但 benchmark 仍持有同一对象。
- 处理结果：已删除无效的局部引用清理语句；benchmark 对 AnnData 的持有和生命周期不变。

### P3-4 重复保存并打印 achievement records

- 状态：已完成
- 位置：`alive/base.py`
- 现状：每轮已经打印指标，仍累计完整嵌套字典供末尾再次打印。
- 处理结果：删除仅供末尾重复打印的 `achievement_records` 累积和 `_print_run_summary()`；每轮指标表、runtime rows、CSV 和预测输出均保持不变。

### P3-5 ActiveLearning 内部参数和小函数冗余

- 状态：已完成
- 现状：diversity helper 的 `strategy` 参数只能是固定值；部分包装函数价值有限。
- 处理结果：删除 diversity 私有 helper 中恒定为 `diversity_sampling` 的参数及重复校验；strategy 的显式合法分支仍统一保留在上层分发入口，算法不变。其余小函数用于隔离不同评分逻辑，保留以方便审查。

### P3-6 baseline 专属适配器位于核心 utils

- 状态：已完成
- 位置：`alive/utils.py`
- 现状：`PertData_Simple` 明确服务 GEARS/scGPT，却占据核心通用 utils 的主要部分。
- 处理结果：`PertDataSimple` 已移至 `baseline/perturbation_generalization/pert_data.py`，供 GEARS 和 scGPT 共享；`alive/utils.py` 仅保留通用随机种子逻辑，核心导出和所有调用方均不再使用旧名称，也不保留兼容层。

### P3-7 baseline 集成测试混在核心测试目录

- 状态：已完成
- 现状：PRESAGE、state、CATP 等测试位于 `alive/tests`，在 sparse worktree 中被跳过。
- 决定：不保留 baseline 集成测试；已删除 PRESAGE、CATP、STATE 以及 baseline 脚本接口测试，`alive/tests` 只保留 ALIVE 核心测试。

### P3-8 docstring 风格与内容不统一

- 状态：已完成
- 范围：`alive/` 生产模块、类、函数和方法。
- 处理结果：统一采用 Google style 的 `Args:`、`Returns:`、`Yields:` 和 `Raises:`；补齐模块级 docstring，核对参数名、返回类型与显式异常，并修正 `get_dataloader()` 错写为返回字典、零样本预测仍描述旧 PCC 指标等不准确说明。测试函数不添加无信息量的模板 docstring。

### P3-9 baseline 上游文档和开发文件冗余

- 状态：已完成
- 现状：新加入的 baseline 携带旧 README、docs、examples、tutorials、demo、`.github`、`.vscode` 和 Read the Docs 配置，与 ALIVE 发布入口无关。
- 处理结果：删除上述非运行内容及失效的打包引用；保留模型源码、许可证、环境与安装元数据、运行数据和 ALIVE adapter。

## 发布验证缺口

### V-1 完整 baseline 集成矩阵尚未执行

- 状态：保留现状
- 当前结果：删除 baseline 专属测试并完成本轮整理后，ALIVE 核心测试为 `71 passed`，不再包含 baseline skips。
- 决定：本仓库不保留或执行 baseline 集成测试；baseline 加入后的接口硬迁移仍需由对应 baseline 自行验证。

### V-2 缺少关键回归测试

- 状态：已完成
- 已决定：
  - 不增加跨 `PYTHONHASHSEED` 子进程测试；保留当前稳定排序实现和进程内顺序测试。
  - 不增加 runtime 与 rebuilt summary 的逐项端到端数值一致性测试；两条路径继续共用同一指标计算实现，离线重建保留为辅助工具。
- 已覆盖：
  - 空预测和空 gene set 输出 `NaN`，已有回归测试。
  - malformed config 和 prediction schema 均已有明确失败测试。
- 已知例外：PyG 缓存指纹已在 P0-6 决定保留现状，因此不增加自动失效测试。
