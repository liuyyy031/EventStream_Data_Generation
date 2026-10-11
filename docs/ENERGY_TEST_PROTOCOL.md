# Energy 领域：服务器上传与验收

在服务器的 STReasoner 仓库根目录执行以下命令。本机只做无 LLM 的机制回归；有 `.env` 的服务器端才调用指定模型。每次运行换一个新输出目录，生成器不会覆盖已有目录。

## 需要上传的文件

```text
data_generation/domain_packages/energy/__init__.py
data_generation/domain_packages/energy/package.py
data_generation/domain_packages/energy/domain_spec.json
data_generation/domain_packages/energy/mechanism_specs.json
data_generation/domain_packages/energy/config/default.json
data_generation/domain_packages/energy/README.md
data_generation/domain_packages/registry.py
data_generation/tests/test_energy.py
data_generation/tests/test_distributed_systems.py
data_generation/README.md
data_generation/docs/ENERGY_TEST_PROTOCOL.md
```

如果服务器还没有上一轮最新的 `generation_core/`、`run_data_generation.py`、`run_continuations.py`、QA/训练和语义矫正实现，应先同步这些公共文件/目录；**仅上传 Energy 包不能补齐旧版服务器**。不要上传本地 `.env`、`artifacts/`、`__pycache__/` 或临时输出。

## 1. 基础回归

```bash
python -m unittest discover -s data_generation/tests -p 'test_*.py' -q
python data_generation/run_data_generation.py --help
```

本地结果：107 项单元测试通过；另以 100 episode × 1000 节点运行，质量报告 `passed=true`，结构、文本对齐和 QA 失败数均为 0，未发生事件数上限截断。100 条包括故障 40、需求突增 35、容量不足 25；实际有 65 次断路器打开，容量不足中有 15 次备用调度，另 10 次由主电源先恢复，候选取消共 20 次。这里**没有调用 LLM**，也不能据此宣称参数分布贴合现实。

## 2. 分场景结构/时间检查

先逐类跑小批量，避免总体 `passed=true` 掩盖某一机制缺陷：

```bash
python data_generation/run_data_generation.py --domain energy --scenario-family line_fault --episode-count 20 --nodes-per-context 1000 --seed 20260930 --judge-mode none --output-dir data_generation/output_energy_fault_20
python data_generation/run_data_generation.py --domain energy --scenario-family demand_surge --episode-count 20 --nodes-per-context 1000 --seed 20260931 --judge-mode none --output-dir data_generation/output_energy_surge_20
python data_generation/run_data_generation.py --domain energy --scenario-family generation_shortfall --episode-count 20 --nodes-per-context 1000 --seed 20260932 --judge-mode none --output-dir data_generation/output_energy_shortfall_20
```

应检查 `quality_report.json`：`passed=true`；`validation_failure_count`、`text_alignment_failure_count`、`qa_validation_failure_count` 均为 0；`administratively_truncated_episode_count=0`。逐条抽看 `events.jsonl`、`event_relations.jsonl` 和 `candidates.jsonl`：故障下游停电/复电集合相同；过载场景既允许未跳闸，也允许跳闸后安全复电；容量不足场景只出现一次 `reserve.restored`。一次 20 条不能证明三种分支的长期比例；所有数值仍属合成先验。

## 3. 服务器 LLM 正式小批量

```bash
python data_generation/run_data_generation.py \
  --domain energy \
  --episode-count 20 \
  --nodes-per-context 1000 \
  --seed 20261001 \
  --judge-mode llm \
  --judge-model deepseek-v4-flash \
  --judge-base-url https://api.deepseek.com \
  --judge-workers 4 \
  --semantic-correction-mode llm \
  --output-dir data_generation/output_energy_llm_20
```

这一步才应看到 `semantic_judge_mode=llm`、`semantic_judge_call_count>0` 和逐条 `judge_results.jsonl`。矫正调用数可以为 0：那只意味着 Judge 没有提出可修复的文本问题，不能说明 LLM 未调用。核查最终 `passed`、拒绝/协议重试原因和 `semantic_corrections.jsonl`。LLM 不改结构、标签、时间戳或 QA 标准答案。

小批量通过后，将 `--episode-count` 改成 100 并使用新目录，检查类型/关系/候选分布及失败记录，再决定扩展领域范围。

## 4. 共享前缀续写

```bash
python data_generation/run_continuations.py \
  --domain energy \
  --scenario-family demand_surge \
  --nodes-per-context 1000 \
  --prefix-events 2 \
  --continuation-count 5 \
  --seed 20261002 \
  --output-dir data_generation/output_energy_continuations_5
```

检查 `continuation_report.json` 的 `passed=true`、共享前缀一致、后缀可变。分支是**同一前缀下的条件随机续写**，不是因果反事实；整个前缀组必须放在同一个训练/验证/测试分组。
