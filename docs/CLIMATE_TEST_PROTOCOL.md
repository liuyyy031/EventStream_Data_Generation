# Climate：服务器上传与验收

在服务器 STReasoner 仓库根目录执行；沿用已有 `.env`，无需反复 `export`。
每轮使用新的输出目录，先验收程序约束，再检查实际 LLM 审核。

## 上传文件

```text
data_generation/domain_packages/climate/__init__.py
data_generation/domain_packages/climate/physics.py
data_generation/domain_packages/climate/package.py
data_generation/domain_packages/climate/domain_spec.json
data_generation/domain_packages/climate/mechanism_specs.json
data_generation/domain_packages/climate/config/default.json
data_generation/domain_packages/climate/README.md
data_generation/domain_packages/registry.py
data_generation/tests/test_climate.py
data_generation/tests/test_distributed_systems.py
data_generation/README.md
data_generation/docs/CLIMATE_TEST_PROTOCOL.md
```

依赖此前 `data_generation/generation_core/mechanisms.py` 中的 `typed_binding_and_state` 策略；本轮未修改公共文件，服务器未同步时一起上传。
注册表会导入全部已实现领域，包括 Agriculture，因此旧领域目录与公共组件应已同步。
Climate 占位空文件已替换为实现，服务器残留 `.gitkeep` 不影响运行。
不要上传 `.env`、`__pycache__/`、`artifacts/` 和本地输出。

## 1. 回归与普通生成

```bash
python -m unittest discover -s data_generation/tests -p 'test_*.py' -q

python data_generation/run_data_generation.py \
  --domain climate \
  --episode-count 100 \
  --nodes-per-context 1000 \
  --seed 20261011 \
  --judge-mode none \
  --output-dir data_generation/output_climate_smoke_100
```

可分别添加 `--scenario-family warming_pulse`、`cooling_pulse`、`combined_warming` 或 `opposing_forcings`，并使用不同的新目录。
1000 是 245 单元、490 驱动源、245 观测实体、20 服务实体的总预算，不是 1000 个同时活跃的真实气候网格。
默认观察窗 30 Julian year，内部秒制。正常运行快并不代表调用了 LLM；`none` 明确不调用。

检查 `quality_report.json` 中 `passed=true`，结构、文本、QA 失败与行政截断为 0。
取消候选和窗外右删失可以是正确结果。默认窗末的样本仍可产生，而其后到达的报告应右删失，不能硬补链条。

重点抽查 `events.jsonl`、`event_relations.jsonl`、`candidates.jsonl`：

- 温度变化满足 `C dT/dt=F−λT` 与累计能量预算，不能在强迫开始/结束时瞬间跳温。
- 正负强迫相抵只能抵消当前输入，不能立即抹去之前积累的温度异常。
- 阈值不可达、仅能渐近到达或持续时间不足时，不强制生成越界/持续确认事件。
- 叠加事件的父事件必须当前实际生效、绑定同一单元；关系是数学推导而非真实气候因果归因。
- 年度采样是一个时点的低频模型快照，不是整年平均，也不是一年长度的物理模拟分箱。
- 报告保留样本快照，通报基于最新可用报告；不得冒充发布时刻的当前状态或持续异常确认。

### 可选：混合观察窗

```bash
python data_generation/run_data_generation.py \
  --domain climate \
  --episode-count 100 \
  --nodes-per-context 1000 \
  --seed 20261014 \
  --time-unit year \
  --duration-min 12 \
  --duration-max 30 \
  --judge-mode none \
  --output-dir data_generation/output_climate_variable_horizon_100
```

先抽观察时长，事件数随其变化；这不是指定或补齐事件数量。
默认背景支持上限 30 年，不可只把 `--duration-max` 改成超过 30 年；若需要更长窗口，应同时修改配置的 `domain_config.duration_seconds` 支持范围。
默认固定窗的数据多数为采样/报告，长度较集中；公共 `max_causal_depth` 可被日程关系主导，不是物理因果深度。

## 2. 服务器 LLM 小批量

```bash
python data_generation/run_data_generation.py \
  --domain climate \
  --episode-count 20 \
  --nodes-per-context 1000 \
  --seed 20261012 \
  --judge-mode llm \
  --judge-model deepseek-v4-flash \
  --judge-base-url https://api.deepseek.com \
  --judge-workers 4 \
  --semantic-correction-mode llm \
  --output-dir data_generation/output_climate_llm_20
```

检查 `quality_report.json`、`judge_results.jsonl`、`semantic_corrections.jsonl`。
确认 `semantic_judge_mode=llm`、`semantic_judge_call_count>0`，且存在真实模型审核返回。
矫正 0 次可以正常。重点关注是否误把虚构基准叫作实测 WMO 正常值、把点样本叫作年平均、把 ±0.7 K 阈值叫作巴黎协定目标，或把温度响应叫作某次真实气候事件的归因证据。
结构/能量错误不能只改文字掩盖，语义矫正不得改动事件、物理值、时间或 QA 真值。
20 条通过后换新目录扩至 100 条。

## 3. 共享前缀续生成

```bash
python data_generation/run_continuations.py \
  --domain climate \
  --scenario-family opposing_forcings \
  --nodes-per-context 1000 \
  --prefix-events 2 \
  --continuation-count 5 \
  --max-events-per-episode 2048 \
  --seed 20261013 \
  --output-dir data_generation/output_climate_continuations_5
```

检查 `continuation_report.json` 全通过，共享前缀事件逐项一致。
已实现的强迫标记、温度、累计能量与连续异常历史不能重置。由前缀确定的物理时钟和观测日程保留；允许的随机未来时钟按公共条件分布续生成。
不强制分支差异，也不称为因果反事实。同组所有分支保持在同一数据划分；该命令不调用 LLM。

## 本地验收记录

Climate 专项 14 项测试、完整回归 171 项通过；专项覆盖四情景 80 次跨种子检查、解析解、能量预算、时间分割不变性、千节点绑定、弱强迫不越界、真实叠加/抵消、强迫结束后惯性、持续不足取消、采样日程、乱序快照、篡改拒绝、短窗、续生成和导出。

100 条、每背景 1000 节点检查通过，生成 6675 个事件、600 条 QA、各 600 条 instruction/chat 样本；结构/文本/QA 失败与行政截断均为 0。
情景分布：增暖 32、冷却 26、同号叠加 15、异号 27；65 个取消、200 个右删失候选正常。
5 条共享前缀续生成全部通过；额外 5 条 12–30 年随机观察窗检查通过，生成 207 个事件。
默认 100 条完整审计/训练文件约 149 MB，不随代码上传，分析可先取报告和小批样本。

本地未调用 LLM，服务器语义审核仍需第 2 步。若本机默认 Python 因缺时区数据库使既有导入测试报 `Asia/Shanghai` 缺失，应使用带时区数据库的 Python 环境，不改写时间含义。
