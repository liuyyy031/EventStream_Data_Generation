# EventFlow v1 服务器部署

## 必须上传

```text
data_generation/run_eventflow_v1.py
data_generation/eventflow_v1/
```

上传 `eventflow_v1` 时排除：

```text
__pycache__/
*.pyc
.test_output/
.test_tmp/
test_output*/
```

旧的 v27 目录和适配器不需要为了新流程重复上传，新流程没有修改它们。

## 可选依赖

- `--judge-mode heuristic`：新流程只使用 Python 标准库，不新增依赖。
- `--judge-mode llm`：还需要服务器已有的
  `data_generation/llm_client.py`，并配置 `LLM_API_KEY`、`LLM_BASE_URL`、
  `LLM_MODEL`。该 Judge 只使用结构化数据和文本，不需要多模态模型。
- `--network-file`：需要额外上传符合
  `config/external_network_example.json` 格式的稀疏网络 JSON。

## 首次服务器冒烟测试

在仓库根目录运行：

```bash
python data_generation/run_eventflow_v1.py \
  --episode-count 8 \
  --nodes-per-network 80 \
  --judge-mode heuristic \
  --output-dir data_generation/output_eventflow_v1_smoke
```

确认 `quality_report.json` 中 `passed` 为 `true` 后，再扩大规模：

同时确认以下字段：

```text
distribution.truncated_episode_count = 0
distribution.max_propagation_children_observed <= 3
topology[*].max_out_degree <= 6
judge_protocol_failure_count = 0
```

Judge 的 JSON、网络或空理由错误会对同一个 episode 重试，不会换种子。
`judge_protocol_retry_count` 大于 0 可以接受；如果协议重试全部失败，任务会
停止并写出 `judge_protocol_failure_*.json`，不会通过重新生成数据掩盖接口错误。

```bash
python data_generation/run_eventflow_v1.py \
  --episode-count 1000 \
  --nodes-per-network 10000 \
  --judge-mode heuristic \
  --output-dir data_generation/output_eventflow_v1_transportation
```

生产数据启用 LLM Judge 时，建议先用少量 episode 测试接口和费用：

```bash
python data_generation/run_eventflow_v1.py \
  --episode-count 10 \
  --nodes-per-network 1000 \
  --judge-mode llm \
  --output-dir data_generation/output_eventflow_v1_llm_judge_test
```
