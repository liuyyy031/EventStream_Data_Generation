# Agriculture：服务器上传与验收

在服务器 STReasoner 仓库根目录执行，沿用已有 `.env`，不用反复 `export`。
每轮使用新的输出目录。先通过程序验收，再检查真实 LLM 返回。

## 上传文件

```text
data_generation/domain_packages/agriculture/__init__.py
data_generation/domain_packages/agriculture/physics.py
data_generation/domain_packages/agriculture/package.py
data_generation/domain_packages/agriculture/domain_spec.json
data_generation/domain_packages/agriculture/mechanism_specs.json
data_generation/domain_packages/agriculture/config/default.json
data_generation/domain_packages/agriculture/README.md
data_generation/domain_packages/registry.py
data_generation/tests/test_agriculture.py
data_generation/tests/test_distributed_systems.py
data_generation/README.md
data_generation/docs/AGRICULTURE_TEST_PROTOCOL.md
```

服务器还需要此前的 `data_generation/generation_core/mechanisms.py`（支持 `typed_binding_and_state`）；该公共文件本轮未修改，未同步时一起上传。
注册表会导入所有已实现领域，包括 Hydrology，旧领域目录及公共运行组件应已同步。
农业旧 `.gitkeep` 占位已替换为实现；服务器残留该空文件不影响运行。
不要上传 `.env`、`__pycache__/`、`artifacts/` 和本地输出。

## 1. 回归与不调用 LLM 的生成检查

```bash
python -m unittest discover -s data_generation/tests -p 'test_*.py' -q

python data_generation/run_data_generation.py \
  --domain agriculture \
  --episode-count 100 \
  --nodes-per-context 1000 \
  --seed 20261008 \
  --judge-mode none \
  --output-dir data_generation/output_agriculture_smoke_100
```

可添加 `--scenario-family rainfed_drydown`、`irrigation_response` 或 `shared_supply_interruption` 检查单一情景，并换新目录。
1000 是六类实体总数，默认 242 个田块；不代表 1000 个同时活跃田块。
默认观察窗 45 day，事件上限 2048；行政截断不能作为正常完成。

检查 `quality_report.json` 的 `passed`、结构/文本/QA 失败数、行政截断和单位。
取消与右删失可以是正确结果；不能为了完整链条强制补写窗外事件。
`--judge-mode none` 不调用 LLM，运行快是正常的，不可据此声称通过 LLM 审核。

重点抽查 `events.jsonl`、`event_relations.jsonl`、`candidates.jsonl`：

- 水量账本满足 `Dr=Dr_initial+ET−net_irrigation−effective_rain+deep_percolation`；雨超出根区可用空间不能消失。
- 缺水边界由 RAW 和当前状态推导，不能把刚到边界写成作物死亡或已减产。
- 积温事件是热时间代理，不应写成实测开花/成熟；温度低于基温时不能强制推进。
- 灌溉时长等于计划体积/流量，供水中断只有已交付部分；重启按当前水分重新计算。
- 报告保留旧采样快照，不冒充报告时的当前读数；旧样本和已经恢复的状态不能触发过期控制。
- 共享供水不是共同因果证据：中断、延期、重启都要有实际参与的父事件。

## 2. 服务器 LLM 审核与矫正

```bash
python data_generation/run_data_generation.py \
  --domain agriculture \
  --episode-count 20 \
  --nodes-per-context 1000 \
  --seed 20261009 \
  --judge-mode llm \
  --judge-model deepseek-v4-flash \
  --judge-base-url https://api.deepseek.com \
  --judge-workers 4 \
  --semantic-correction-mode llm \
  --output-dir data_generation/output_agriculture_llm_20
```

检查 `quality_report.json`、`judge_results.jsonl`、`semantic_corrections.jsonl`。
必须确认 `semantic_judge_mode=llm`、`semantic_judge_call_count>0` 及真实审核返回，才算调用了 API。
矫正 0 次可以正常。重点关注模型有没有误把热时间代理当作真实生育期、RAW 阈值当作已发生作物损伤、有效雨当成总降雨、计划目标当成精确结束水分。
如果是结构或水量错误，不能靠改文字掩盖。语义矫正不能改事件、时间、物理量和 QA 标签。
20 条审核确认后，换新目录把 `--episode-count` 改为 100 再检查。

## 3. 共享前缀续生成

```bash
python data_generation/run_continuations.py \
  --domain agriculture \
  --scenario-family shared_supply_interruption \
  --nodes-per-context 1000 \
  --prefix-events 2 \
  --continuation-count 5 \
  --max-events-per-episode 2048 \
  --seed 20261010 \
  --output-dir data_generation/output_agriculture_continuations_5
```

检查 `continuation_report.json` 全部通过，共享前缀事件逐项一致。
水分、热量和已执行补水不能在分支起点重置；由已有状态确定的物理时钟不能随意重新抽样。
随机故障/维修/通信等尚未发生的时钟按公共组件的条件分布续生成。不强制分支不同，也不称为因果反事实。
同一前缀组的全部分支必须处于同一数据划分。该命令不调用 LLM。

## 本地验收记录

专项测试包含配置、解析水量、时间分割不变性、三情景跨种子、千节点绑定、积温时钟、有效雨渗漏、供水多父工作流、部分交付、请求释放、乱序快照、篡改拒绝、右删失、续生成和 QA/训练导出。
本轮农业专项 13 项、完整回归 157 项通过。100 条、每背景 1000 节点的本地生成检查通过，产生 6723 个事件、600 条 QA 和各 600 条 instruction/chat 训练样本；结构、文本、QA 失败与行政截断均为 0。
三情景分别 37/31/32 条；270 个取消候选、164 个右删失候选是正常状态结果。5 条共享前缀续生成均通过。
上述 100 条的本地完整审计/训练输出约 169 MB，包含重复的 QA 输入、文本对齐和风险集记录；不应随代码上传，分析时可先只取报告与小批量样本。
本地不调用 LLM；服务器语义质量仍需运行上述第 2 步检查。
若本机默认 Python 缺时区库导致既有真实时间表导入测试报告 `Asia/Shanghai` 缺失，应使用带时区数据库的环境，而不是修改生成的时间含义。
