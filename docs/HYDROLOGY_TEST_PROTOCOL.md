# Hydrology：服务器上传与验收

所有命令在服务器 STReasoner 仓库根目录执行。服务器已有 `.env`，无需逐次 `export`；仍显式指定模型和接口地址。
每轮使用新的输出目录，避免混合不同版本数据。

## 上传文件

```text
data_generation/domain_packages/hydrology/__init__.py
data_generation/domain_packages/hydrology/package.py
data_generation/domain_packages/hydrology/domain_spec.json
data_generation/domain_packages/hydrology/mechanism_specs.json
data_generation/domain_packages/hydrology/config/default.json
data_generation/domain_packages/hydrology/README.md
data_generation/domain_packages/registry.py
data_generation/generation_core/mechanisms.py
data_generation/tests/test_hydrology.py
data_generation/tests/test_distributed_systems.py
data_generation/README.md
data_generation/docs/HYDROLOGY_TEST_PROTOCOL.md
```

`generation_core/mechanisms.py` 本轮没有修改，列出是因为本领域依赖此前增加的 `typed_binding_and_state` 证据策略，服务器未同步时应一起上传。
其余已实现领域（包括 Public Health）与公共组件应已同步；注册表会导入这些领域，缺少旧目录会导致导入失败。
不要上传 `.env`、`__pycache__/`、`artifacts/` 或本地输出目录。服务器残留 `.gitkeep` 不影响运行。

## 1. 本地机制、约束与导出同构测试

```bash
python -m unittest discover -s data_generation/tests -p 'test_*.py' -q

python data_generation/run_data_generation.py \
  --domain hydrology \
  --episode-count 100 \
  --nodes-per-context 1000 \
  --seed 20261005 \
  --judge-mode none \
  --output-dir data_generation/output_hydrology_smoke_100
```

可分别加 `--scenario-family isolated_storm`、`tributary_confluence` 或 `repeated_storms` 单独检查情景，使用不同的新目录。
1000 是四类实体总预算，不是 1000 个测站；背景图不要求每个节点都进入这次事件流。

验收 `quality_report.json`：`passed=true`，结构、文本对齐、QA 失败与行政截断均为 0。候选取消、窗口右删失都可以正常，出现 0 次也不说明机制失效。
这条命令不调用 LLM；不能拿它证明已通过 LLM 审核。

重点抽查 `events.jsonl`、`event_relations.jsonl`、`candidates.jsonl`：

- 初始损失未填满或雨强不超过损失率时不能强制产流。
- 产流体积等于流量乘脉冲宽度；传播到下一河段后体积、流量与宽度不变，滞后等于上游河段长度除以波速。
- 多父阈值事件只能引用当前正值且时间重叠的实际到达贡献，不能引用静态连线或已结束的脉冲。
- 报告保留采样快照；乱序较旧报告不得覆盖更新的报告，也不能重启过时的预警。
- 两场短时间连续降雨保留初始损失消耗历史，不能在第二场雨自动重置为干燥初态。

## 2. 服务器 LLM 小批量

```bash
python data_generation/run_data_generation.py \
  --domain hydrology \
  --episode-count 20 \
  --nodes-per-context 1000 \
  --seed 20261006 \
  --judge-mode llm \
  --judge-model deepseek-v4-flash \
  --judge-base-url https://api.deepseek.com \
  --judge-workers 4 \
  --semantic-correction-mode llm \
  --output-dir data_generation/output_hydrology_llm_20
```

检查 `quality_report.json`、`judge_results.jsonl` 和 `semantic_corrections.jsonl`。
确认 `semantic_judge_mode=llm`、`semantic_judge_call_count>0` 并且有真实模型审核返回，才算实际调用。
矫正次数为 0 可以正常。重点检查模型是否误把流量阈值越过写成洪灾、把 Lag routing 写成带衰减、把共线写成共同因果，或把历史样本误写成报告时的当前流量。
发现结构性问题不能只改写文字掩盖；语义矫正不得修改事件、物理量、时间与 QA 标签。
20 条审核确认后可换新目录扩至 100 条。

## 3. 共享前缀续生成

```bash
python data_generation/run_continuations.py \
  --domain hydrology \
  --scenario-family tributary_confluence \
  --nodes-per-context 1000 \
  --prefix-events 2 \
  --continuation-count 5 \
  --max-events-per-episode 2048 \
  --seed 20261007 \
  --output-dir data_generation/output_hydrology_continuations_5
```

检查 `continuation_report.json`：全部分支通过，共享前缀完全一致。
已经由前缀确定的降雨标记与物理时钟不能随便改；仍未发生的随机天气/通信时钟在条件分布下续生成，可产生不同未来。
不强制分支差异，也不称为因果反事实；同一前缀组必须放在同一个数据划分中。
此命令不调用 LLM。

## 本地验收记录

专项 12 项测试通过，包含三情景 60 次跨种子检查、水量公式、千节点稀疏图、出口体积守恒、不产流、连续降雨损失记忆、实际支流叠加、迟到与乱序报告、篡改拒绝和续生成。
带时区数据库的 Python 环境下，全套 144 项回归通过；本机默认 Python 缺少时区数据库时，既有真实时间表导入测试可能报 `Asia/Shanghai` 缺失，与 Hydrology 生成逻辑无关。

最终代码按第 1 节同参数生成 100 条、4 张千实体图，共 3635 个事件、600 条 QA；孤立降雨 40、支流汇合 29、重复降雨 31 条。
结构、文本、QA 失败和行政截断均为 0；事件数 3～132，27 场降雨没有产生正径流，6 个超阈值事件具有多个实际活动贡献。
本批候选取消 1 次、右删失 0 次。24 小时窗足以覆盖这一批有限情景并不表示删失机制不存在：短窗回归专门检查了右删失。
5 个共享前缀分支均通过，事件数为 40、48、40、56、56；事件数相同也不意味着所有后续报告时刻相同。
本批 `risk_sets.jsonl` 约 20.4 MiB、`events.jsonl` 约 8.7 MiB；风险集是审计数据，不必全部一起输入 LLM。

上述本地试跑未调用 LLM。服务器 LLM 语义验收仍需用户执行，本地确定性验收不能替代它，也不能证明先验参数具有实测统计真实性。
