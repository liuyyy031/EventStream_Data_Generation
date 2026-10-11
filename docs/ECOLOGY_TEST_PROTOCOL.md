# Ecology：服务器上传与验收

以下命令均从服务器 STReasoner 仓库根目录执行。服务器已有 `.env` 时无需逐次 `export`，但 LLM 模型与接口地址仍需显式指定。每次运行使用新的输出目录。

## 上传文件

```text
data_generation/domain_packages/ecology/__init__.py
data_generation/domain_packages/ecology/package.py
data_generation/domain_packages/ecology/domain_spec.json
data_generation/domain_packages/ecology/mechanism_specs.json
data_generation/domain_packages/ecology/config/default.json
data_generation/domain_packages/ecology/README.md
data_generation/domain_packages/registry.py
data_generation/generation_core/mechanisms.py
data_generation/tests/test_ecology.py
data_generation/tests/test_distributed_systems.py
data_generation/README.md
data_generation/docs/ECOLOGY_TEST_PROTOCOL.md
```

假定服务器已经同步之前的公共调度器、验证器、QA/训练导出和 LLM Judge/矫正模块。不要上传 `.env`、本地 `artifacts/`、输出目录或 `__pycache__/`。

## 1. 先验机制与导出自测（不调用 LLM）

```bash
python -m unittest discover -s data_generation/tests -p 'test_*.py' -q

python data_generation/run_data_generation.py \
  --domain ecology \
  --episode-count 100 \
  --nodes-per-context 1000 \
  --seed 20261002 \
  --judge-mode none \
  --output-dir data_generation/output_ecology_smoke_100
```

若需单独检查三条链，分别加 `--scenario-family regional_drought`、`resource_pulse` 或 `invasive_colonization`，并分别使用新目录。`quality_report.json` 应有 `passed=true`，结构、文本对齐、QA 失败数为 0，`administratively_truncated_episode_count=0`；检查 `candidate_status_counts` 的 `cancelled` 与 `right_censored`。`judge-mode none` 时 `semantic_judge_call_count=0` 是预期，不可据此声称通过 LLM 审核。

本地回归 121 项测试通过；相同配置的 100 条千节点试跑通过，三场景分别为干旱 32、资源脉冲 33、入侵 35 条，共 5403 个事件，结构/文本/QA 失败和行政截断均为 0，候选取消 292、右删失 104。审计文件较大（该轮 `risk_sets.jsonl` 约 81 MiB、`candidates.jsonl` 约 53 MiB）；展示事件流可优先查看 `events.jsonl`、`event_relations.jsonl` 和少量 episode 的文本，不必将全部审计记录一起输入模型。

抽查 `events.jsonl`、`event_relations.jsonl`、`candidates.jsonl`：捕食者变化前应有同斑块草食种群跨阈值；失败的入侵建立不能同时成功；沿廊道扩散的两端必须确实是相邻斑块。资源变化字段可用领域说明中的公式复算。大图节点很多，并不意味着每次 episode 必须动用所有节点。

## 2. 服务器 LLM 小批量

```bash
python data_generation/run_data_generation.py \
  --domain ecology \
  --episode-count 20 \
  --nodes-per-context 1000 \
  --seed 20261003 \
  --judge-mode llm \
  --judge-model deepseek-v4-flash \
  --judge-base-url https://api.deepseek.com \
  --judge-workers 4 \
  --semantic-correction-mode llm \
  --output-dir data_generation/output_ecology_llm_20
```

验收 `quality_report.json` 的 `passed=true`、`semantic_judge_mode=llm`、`semantic_judge_call_count>0`，并检查 `judge_results.jsonl` 与 `semantic_corrections.jsonl` 的驳回原因、协议重试、文本矫正情况。矫正次数为 0 可以正常；Judge 调用数为 0 则不能算完成 LLM 测试。20 条确认后再换目录扩至 100 条。语义矫正只可修改文本投影，不能改动事件结构和 QA 标签。

## 3. 共享前缀续写

```bash
python data_generation/run_continuations.py \
  --domain ecology \
  --scenario-family invasive_colonization \
  --nodes-per-context 1000 \
  --prefix-events 2 \
  --continuation-count 5 \
  --seed 20261004 \
  --output-dir data_generation/output_ecology_continuations_5
```

共享前缀应完全相同，后续分支可出现建立、失败、右删失或不同扩散路径。它们是同前缀条件随机续写，不是因果反事实；相同前缀组不可拆入不同训练/测试分割。
