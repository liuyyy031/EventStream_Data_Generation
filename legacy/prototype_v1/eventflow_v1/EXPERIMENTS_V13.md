# EventFlow v1.3.1 验证与校准

## 1. 多种子稳定性实验

```bash
python data_generation/run_eventflow_stability.py \
  --seed-count 5 \
  --episodes-per-seed 100 \
  --nodes-per-network 1000 \
  --judge-mode heuristic \
  --output-dir data_generation/output_eventflow_v13_stability_5x100
```

主报告是 `stability_report.json`，重点查看：

- `passed`：工程稳定性硬检查是否全部通过。
- `checks`：是否全部写入、无截断、分支受限、拓扑度受限、场景完整覆盖。
- `generation_retry_fraction`：生成层失败后换种子重试的比例，默认硬阈值为 5%。
- `across_seed_metrics.*.coefficient_of_variation`：各种子之间统计量的相对波动。

该实验证明生成程序的可重现性和稳定性，不能单独证明交通统计真实性。

## 2. Judge 反例挑战

```bash
python data_generation/run_eventflow_judge_challenge.py \
  --judge-mode llm \
  --output-dir data_generation/output_eventflow_v13_judge_challenge
```

挑战集包含 1 条合法对照和 6 类反例：

| 反例 | 预期拦截层 |
|---|---|
| 子事件早于父事件 | Common deterministic |
| 文本 grounded fact 与事件不一致 | Common deterministic |
| 传播引用不存在的道路边 | Transportation domain |
| 滞后无法由记录的公式输入复现 | Transportation domain |
| 恢复事件没有来源关系 | Transportation domain |
| 文本宣称未记录的直接因果 | LLM semantic Judge |

`--judge-mode heuristic` 只验证前两层，“文本宣称未记录的直接因果”会标记为
`assessable_in_mode=false`；正式报告应使用 `llm`。恢复事件的来源关系属于可精确
程序化判断的不变量，不再依赖 LLM 长上下文比对。

## 3. 交通参数校准

默认配置使用 FHWA 文档中的 BPR 链路时间函数：

```text
t = t0 * (1 + alpha * (v/c)^beta)
```

每条传播关系保存 `lag_calibration`，包括公式、输入、系数、随机因子、
来源和限制，验收器会重算滞后并检查是否与事件时间一致。

默认 `alpha=0.15, beta=4`只是参考系数。基础流量和恢复时间参数仍为合成先验，
因此 profile 状态是：

```text
reference_formula_with_synthetic_inputs_pending_empirical_fit
```

将真实观测整理为 `config/calibration_input_schema.csv` 的字段后运行：

```bash
python data_generation/run_eventflow_calibration.py \
  --input-csv /path/to/transport_reference.csv \
  --output-profile data_generation/calibration/transport_fitted.json
```

拟合器会输出：

- BPR `alpha` 与 `beta`；
- 恢复时间的严重程度权重和分场景中位数；
- 训练集与留出集的 MAE/RMSE；
- 是否具有留出验证的明确状态。

`calibration_input_schema.csv` 中的数字只是字段格式示例，不能当作真实校准数据。
