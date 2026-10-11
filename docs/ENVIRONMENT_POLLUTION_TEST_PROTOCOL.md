# Environment & Pollution：服务器上传与验收

所有命令在服务器 STReasoner 仓库根目录执行。`.env` 已在服务器时无需每次 `export`；需要 LLM 的命令仍须显式指定模型。输出目录必须是新目录。

## 上传清单

```text
data_generation/domain_packages/environment_pollution/__init__.py
data_generation/domain_packages/environment_pollution/package.py
data_generation/domain_packages/environment_pollution/domain_spec.json
data_generation/domain_packages/environment_pollution/mechanism_specs.json
data_generation/domain_packages/environment_pollution/config/default.json
data_generation/domain_packages/environment_pollution/README.md
data_generation/domain_packages/registry.py
data_generation/tests/test_environment_pollution.py
data_generation/tests/test_distributed_systems.py
data_generation/README.md
data_generation/docs/ENVIRONMENT_POLLUTION_TEST_PROTOCOL.md
```

不需为这次修改单独上传 `generation_core/`，前提是服务器已经有上一轮完整的公共调度、验收、文本/QA 和矫正代码。旧服务器若缺这些基础模块，应先同步对应公共目录；不要上传 `.env`、`artifacts/`、本地输出或 `__pycache__/`。

## 1. 回归与无 LLM 机制检查

```bash
python -m unittest discover -s data_generation/tests -p 'test_*.py' -q

python data_generation/run_data_generation.py \
  --domain environment_pollution \
  --episode-count 100 \
  --nodes-per-context 1000 \
  --seed 20261001 \
  --judge-mode none \
  --output-dir data_generation/output_environment_pollution_smoke_100
```

本地回归共 114 项测试通过。相同规模试跑：100 条、4 个千节点河网、2963 个事件；结构、文本对齐、QA 失败均为 0，无事件数上限截断。场景为旁路 31、工业排放 34、降雨径流 35；有 666 次脉冲到达、335 次监测告警、335 次取水口关闭；46 条脉冲在观察窗内物理清除并汇总，54 条到窗结束仍右删失。`semantic_judge_call_count=0` 是因为本步骤明确使用 `--judge-mode none`，不能视为 LLM 审核。

如需定位单类问题，分别使用 `--scenario-family industrial_spill`、`wastewater_bypass` 或 `storm_runoff`，每次换新输出目录。检查 `quality_report.json`、`validation.jsonl`，并抽查相关 episode 的 `events.jsonl`、`event_relations.jsonl`、`candidates.jsonl`。重点看：下游父事件是否只跨一条真实流向边；通量与浓度单位能否复算；尾缘清除是否在负荷结束后；取水口是否只依据监测报告行动。

## 2. 服务器 LLM 小批量

```bash
python data_generation/run_data_generation.py \
  --domain environment_pollution \
  --episode-count 20 \
  --nodes-per-context 1000 \
  --seed 20261003 \
  --judge-mode llm \
  --judge-model deepseek-v4-flash \
  --judge-base-url https://api.deepseek.com \
  --judge-workers 4 \
  --semantic-correction-mode llm \
  --output-dir data_generation/output_environment_pollution_llm_20
```

验收 `quality_report.json` 的 `passed=true`，三个 deterministic failure count 为 0，`semantic_judge_mode=llm` 且 `semantic_judge_call_count>0`。再看 `judge_results.jsonl`、`semantic_corrections.jsonl` 中的驳回、协议重试及文本矫正。矫正次数为 0 可以是正常现象；Judge 调用次数为 0 则说明没有完成所要求的 LLM 检查。20 条通过后更换目录、扩至 100 条。

## 3. 共享前缀续写

```bash
python data_generation/run_continuations.py \
  --domain environment_pollution \
  --scenario-family industrial_spill \
  --nodes-per-context 1000 \
  --prefix-events 2 \
  --continuation-count 5 \
  --seed 20261002 \
  --output-dir data_generation/output_environment_pollution_continuations_5
```

本地同配置五条续写均通过，共享前缀相同、后缀文件各不相同。它们是条件随机续写，不是因果反事实；同一个前缀组不能拆到不同数据集分割中。

领域公式、节点语义和未覆盖的环境过程见 [领域说明](../domain_packages/environment_pollution/README.md)。
