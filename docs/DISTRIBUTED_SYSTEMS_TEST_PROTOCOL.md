# 分布式系统领域：上传与测试

## 本轮变更与服务器同步清单

本轮只补全原占位领域 distributed_systems，没有添加未经定义的新领域。交通、医疗的领域规则保持原样；公共调度器只增加一个可选领域验收钩子，不能跳过公共验收。

在本地 STReasoner 根目录下，将以下路径按原相对位置上传到服务器 STReasoner 根目录。不要上传本地 .env、__pycache__ 或 artifacts 测试输出。

完整上传这个目录（6 个源代码/配置/说明文件）：

```text
data_generation/domain_packages/distributed_systems/
  __init__.py
  package.py
  domain_spec.json
  mechanism_specs.json
  config/default.json
  README.md
```

同时同步这些文件：

```text
data_generation/domain_packages/registry.py
data_generation/run_data_generation.py
data_generation/run_continuations.py
data_generation/generation_core/scheduler.py
data_generation/tests/test_distributed_systems.py
data_generation/tests/test_generation_core.py
data_generation/README.md
data_generation/docs/DISTRIBUTED_SYSTEMS_TEST_PROTOCOL.md
```

依赖此前已上传的通用时间模型、稀疏拓扑、候选竞争、续写、语义矫正、QA 导出等组件。如果服务器不是上一轮最新版本，应先同步最新 generation_core 整个目录，再运行测试；不能只上传新领域文件。

## 1. 单元回归

以下命令均在服务器仓库根目录执行，不在本机调用模型：

```bash
python -m unittest discover -s data_generation/tests -p "test_*.py" -q
python data_generation/run_data_generation.py --help
```

本轮本地结果：99 项测试通过，其中新增领域测试 15 项。覆盖角色类型、部署一致性、请求标识、超时/迟到响应隔离、重试退避、精确同时刻响应优先、动态恢复时钟、共享前缀续写、文本对齐和 CLI 的 QA/训练导出。

## 2. 无 LLM 的确定性与生成检查

```bash
python data_generation/run_data_generation.py \
  --domain distributed_systems \
  --episode-count 100 \
  --nodes-per-context 1000 \
  --seed 20260929 \
  --judge-mode none \
  --output-dir data_generation/output_distributed_systems_smoke_100
```

本轮本地同参数验收结果：100 个 episode，4 个上下文各 1000 个节点，2319 个事件，600 组 QA；结构、文本对齐、QA 失败数全部为 0，无事件数上限截断。实际有 172 次超时、149 次重试发送、172 个迟到响应、23 次重试耗尽。LLM 调用次数为 0；这是机制/协议验收，不是 LLM 语义审核的通过记录。

每个 episode 是有限并行 RPC 工作流；不要求 1000 个实体全部参与。调用图出度有差异，而非固定规则格网。节点数表示总实体数，非服务数或事件数。

## 3. 服务器 LLM 小规模正式检查

继续使用服务器已有 .env，无需每次 export。沿用当前兼容客户端读取的密钥变量，显式指定模型和服务地址：

```bash
python data_generation/run_data_generation.py \
  --domain distributed_systems \
  --episode-count 20 \
  --nodes-per-context 1000 \
  --seed 20260929 \
  --judge-mode llm \
  --judge-model deepseek-v4-flash \
  --judge-base-url https://api.deepseek.com \
  --judge-workers 4 \
  --semantic-correction-mode llm \
  --output-dir data_generation/output_distributed_systems_llm_20
```

通过后可将 episode-count 改为 100，并换一个输出目录。LLM 负责审核，必要时矫正文句；结构和时间仍由机制生成。正常数据可能无需矫正，因此 correction_call_count=0 本身不是问题，但 judge_call_count 不应为 0。

优先回传 quality_report.json、judge_results.jsonl 和 semantic_corrections.jsonl。若存在结构驳回，再按相关 episode 抽取 events.jsonl、event_relations.jsonl 和 candidates.jsonl 分析，避免将所有大文件一次读入。

验收要求：

- passed=true，accepted_episode_count 等于请求数量，failed_episode_indices 为空。
- validation_failure_count、text_alignment_failure_count、qa_validation_failure_count 均为 0。
- semantic_judge_mode=llm，semantic_judge_call_count 大于 0，有对应逐条审核记录。
- 检查重试/驳回/矫正原因，而不是只看最终 passed。特别核实是否误将“迟到响应仍出现”判断成结构错误。
- 不因短观察窗而强行补齐恢复或终局；右删失与事件上限截断要分开解释。

## 4. 同前缀多种续写

```bash
python data_generation/run_continuations.py \
  --domain distributed_systems \
  --scenario-family slow_dependency \
  --nodes-per-context 1000 \
  --prefix-events 4 \
  --continuation-count 5 \
  --seed 20260929 \
  --output-dir data_generation/output_distributed_systems_continuations_5
```

本地相同参数的五个分支全部通过，事件数分别为 25、25、20、20、25。此命令不调用 LLM。它保留相同前缀以及已确定的超时/退避预约，对未完成的连续随机时钟进行条件续写；不是因果反事实。五个分支必须放在同一个训练/验证/测试分组，避免共享前缀泄漏。续写目录已存在时应换新目录。

## 5. 需要保留的边界

当前保证的是所声明规则与生成记录相互一致，不宣称真实系统分布或所有互联网事件类型均已覆盖。数值仍是合成先验，不为了可执行性声称经过真实数据拟合。

本包使用 actor-local observed_at。生成数据是离线整体视角，不等同于日志采集器在线已知事件；在线采集器任务需要按 recorded_at 另建可见性视图。

完整机制、节点与模型公式见 [领域说明](../domain_packages/distributed_systems/README.md)。
