**1. 总体架构**
1. 配置层：`maple/function/read/command_control.py`  
2. 构建层：`maple/function/engine.py` + `maple/function/utility/uncertainty/builder.py` + `maple/function/calculator/set_calculator.py`  
3. 运行层：`maple/function/utility/uncertainty/ensemble_calculator.py`  
4. 指标与日志层：`maple/function/utility/uncertainty/metrics.py` + `maple/function/utility/uncertainty/logger.py`

**2. 启动时发生什么**
1. 输入里 `#ensemble(...)` 被 `CommandControl` 解析并校验。  
`command_control.py` 里要求：`active=true` 时 `models>=2`，`on_error` 目前只支持 `warn`。
2. `engine._mlp_initiator()` 读到 `ensemble.active=true` 后，不走普通单模型 `set_calculator()`，而是调用 `wrap_with_ensemble(...)`。  
文件：`maple/function/engine.py`
3. `wrap_with_ensemble()` 用 `models[0]` 构建主模型，`models[1:]` 构建 observer。  
文件：`maple/function/utility/uncertainty/builder.py`  
文件：`maple/function/calculator/set_calculator.py`（`build_calculator_from_path`）
4. 最终返回 `EnsembleCalculator`，挂到 `atoms.calc` 上。之后所有任务都还是原逻辑调用 `atoms.get_potential_energy/get_forces`，只是 calc 被包装了。

**3. 运行时核心逻辑（每次 calc 调用）**
1. 任务代码调用 `atoms.calc.calculate(...)`（通过 ASE API 间接触发）。  
2. `EnsembleCalculator.calculate()` 先让 `primary_calc` 计算，并把结果作为“正式结果”返回给任务。  
文件：`maple/function/utility/uncertainty/ensemble_calculator.py`
3. 再用同一几何、同一属性请求让每个 observer 计算。  
4. 统计不确定度并写入 `.out`。  
5. 某 observer 出错就 `drop`，后续调用不再参与。

**4. 指标定义（当前实现）**
1. `energy_stats`：对 `[primary + observers]` 算 `mean/std/rms`，并给每个 observer 的 `dE = E_obs - E_primary`。  
文件：`maple/function/utility/uncertainty/metrics.py`
2. `force_stats`：对每个力分量先算模型间标准差 `sigma`，再聚合成 `sigma_rms` 和 `sigma_max`。  
文件：`maple/function/utility/uncertainty/metrics.py`
3. 日志文本格式由 `format_uncertainty_lines()` 统一生成。  
文件：`maple/function/utility/uncertainty/logger.py`

**5. eval 很多的根因**
1. `eval` 计数是“calculator 调用次数”，不是“算法确认步次数”。  
2. `PRFO/relaxed scan/autoneb` 内部会做很多 trial / line-search / force reeval。每次都会触发 wrapper 一次。  
3. 当前去重是“几何 + props 哈希”。同一几何若先 `energy` 再 `forces`，会记两条，因为 props 不同。  
文件：`maple/function/utility/uncertainty/ensemble_calculator.py` 的 `_make_sig(...)` + `calculate(...)`

**6. 现在这版架构的边界**
1. 优点：低侵入，任务代码几乎不改。  
2. 代价：日志粒度天然是“调用级”，不是“确认步级”。  
3. `freq/md` 在 builder 层直接跳过 ensemble。  
文件：`maple/function/utility/uncertainty/builder.py`
4. `get_hessian` 等方法直接转发主模型，不做 Hessian uncertainty。  
文件：`maple/function/utility/uncertainty/ensemble_calculator.py`
5. `build_calculator_from_path` 当前是“按当前主模型家族”构建 observer，不是任意混搭。  
文件：`maple/function/calculator/set_calculator.py`
