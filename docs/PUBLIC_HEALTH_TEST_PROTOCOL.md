# Public Health：服务器上传与测试

从服务器 STReasoner 仓库根目录执行。服务器已有 `.env` 时无需逐次 `export`。使用新的输出目录，避免混合不同代码版本的数据。

## 上传文件

```text
data_generation/domain_packages/public_health/__init__.py
data_generation/domain_packages/public_health/package.py
data_generation/domain_packages/public_health/domain_spec.json
data_generation/domain_packages/public_health/mechanism_specs.json
data_generation/domain_packages/public_health/config/default.json
data_generation/domain_packages/public_health/README.md
data_generation/domain_packages/registry.py
data_generation/generation_core/mechanisms.py
data_generation/tests/test_public_health.py
data_generation/tests/test_distributed_systems.py
data_generation/README.md
data_generation/docs/PUBLIC_HEALTH_TEST_PROTOCOL.md
```

其中 `generation_core/mechanisms.py` 本轮未改，但 Public Health 与 Ecology 都依赖此前增加的 `typed_binding_and_state` 证据策略；未同步 Ecology 的服务器应一并上传。
假定其它已实现领域和公共调度器、验证器、QA/训练导出、Judge/矫正代码已经同步。不要上传 `.env`、`artifacts/`、输出目录或 `__pycache__/`。
本地已移除占位 `.gitkeep`；若服务器仍有这个占位文件，不影响运行。

## 1. 不调用 LLM 的机制自测

```bash
python -m unittest discover -s data_generation/tests -p 'test_*.py' -q

python data_generation/run_data_generation.py \
  --domain public_health \
  --episode-count 100 \
  --nodes-per-context 1000 \
  --seed 20260930 \
  --judge-mode none \
  --output-dir data_generation/output_public_health_smoke_100
```

若需单独检查三种情景，分别加 `--scenario-family household_cluster`、`workplace_cluster` 或 `surveillance_response`，并分别使用新目录。
这里 1000 是人员与场所/机构的总实体数，不是 1000 名人员。

检查 `quality_report.json`：`passed=true`，结构、文本对齐和 QA 失败数均为 0，行政截断为 0；窗口结束产生右删失、状态变化产生取消都可以正常。
检查感染事件来源必须仍具有传染性，目标此前必须易感，隔离或检疫中不能新发生有效接触传播。
同一人最多一次成功感染；多人竞争不构成多个实现父事件。阳性/阴性必须对齐同一标本，结果使用采样时冻结的状态，阴性不改变感染状态。
风险通知只描述共享场所名单的预防性动作，不能写成已确认接触或感染。
`judge-mode none` 不调用 LLM，不能据此声称已通过语义模型审核。

## 2. 服务器 LLM 小规模测试

```bash
python data_generation/run_data_generation.py \
  --domain public_health \
  --episode-count 20 \
  --nodes-per-context 1000 \
  --seed 20261001 \
  --judge-mode llm \
  --judge-model deepseek-v4-flash \
  --judge-base-url https://api.deepseek.com \
  --judge-workers 4 \
  --semantic-correction-mode llm \
  --output-dir data_generation/output_public_health_llm_20
```

检查 `quality_report.json`、`judge_results.jsonl`、`semantic_corrections.jsonl`。必须有 `semantic_judge_mode=llm`、`semantic_judge_call_count>0` 及真实返回的 Judge 记录，才算实际调用。
语义矫正次数为 0 可以正常，但仍应查看审核理由是否区分生物状态、标本检测、行政隔离与风险通知。
若发生不自然的医学措辞，应先判断是文本问题还是生成机制问题；矫正不能通过改写文字掩盖机制错误。
20 条确认后可将条数改为 100，使用新输出目录再检查。

## 3. 同一前缀的随机续生成

```bash
python data_generation/run_continuations.py \
  --domain public_health \
  --scenario-family workplace_cluster \
  --nodes-per-context 1000 \
  --prefix-events 2 \
  --continuation-count 5 \
  --max-events-per-episode 2048 \
  --seed 20261002 \
  --output-dir data_generation/output_public_health_continuations_5
```

检查 `continuation_report.json` 与任一分支的事件/关系文件：前缀完全一致，后续传播、症状、检测或窗口删失可不同，但每个分支都应通过约束。
这是条件随机续生成，不是因果反事实；同一前缀的分支应保留在同一个训练/验证/测试划分。
此命令不调用 LLM。

## 本地验收记录

本地 Public Health 专项 11 项测试通过，包含三场景 36 次跨种子检查、千节点类型绑定、多人竞争来源、晚到阳性、阴性不改变感染状态、限制取消传播、篡改拒绝和续生成。
带时区数据库的 Python 环境下，全套 132 项回归通过。Windows 默认 Python 若缺少 `tzdata`，既有真实时间表导入测试会因找不到 `Asia/Shanghai` 报错；这不是 Public Health 生成失败，Linux 通常由系统时区数据库提供该信息。

按第 1 节同样的参数本地生成 100 条、4 个千实体上下文，共 6806 个事件与 600 条 QA；家庭情景 37、工作场所情景 33、筛查响应 30 条。
结构、文本对齐、QA 失败和行政截断均为 0；候选取消 1378、右删失 368。每条事件数为 3～214，未强制填满图节点。
同前缀的 5 个千节点续生成分支全部通过，事件数分别为 128、173、55、79、125，前缀保持相同。
审计风险集文件本轮约 106.5 MiB。给 LLM 的实际输入可使用局部事件及文本/关系，不必把整份风险集审计一起输入；它保留供生成可追溯检查。

上述试跑均明确未调用 LLM。服务器 LLM 测试由用户执行，本地结果不能替代服务器语义验收，更不能证明疾病参数已拟合或所有公共卫生事件均被覆盖。
