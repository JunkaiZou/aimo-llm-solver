# 技术细节深度剖析

## 1. Harmony 协议与工具调用

### 背景

GPT-OSS 系列模型不使用标准的 OpenAI `function_call` 格式，而是通过 **Harmony 协议** 表达工具调用意图。

Harmony 是一个 token 级别的协议，通过特殊标记序列表示：
- 工具调用的触发
- 工具的接收者
- 返回结果的处理

### Harmony Conversation 的构建

```python
# 伪代码结构
conversation = {
    "system_prompt": "你是一个数学竞赛解题专家...",
    "messages": [
        {
            "role": "user",
            "content": "<image>Solve this math problem: ...",
            "tools": {
                "jupyter_python": {
                    "description": "在 Jupyter 中执行 Python 代码",
                    "parameters": {...}
                }
            }
        }
    ]
}
```

### 工具执行流程

```
用户问题输入
    ↓
LLM 生成响应（可能包含工具调用）
    ↓
Harmony 解析器识别 <|action_start|>... <|action_end|>
    ↓
提取工具类型（如 "jupyter_python"）和代码
    ↓
发送代码到对应 Jupyter kernel
    ↓
kernel 执行并返回 stdout/stderr
    ↓
将结果包装为 tool_message 回填给 LLM
    ↓
LLM 继续推理或输出最终答案
```

### 持久 Jupyter 核心池的实现

**为什么需要持久池？**
- 每次启动 Python 需要 ~1-2 秒
- 16 个 kernel × 8 个 attempt × 50 题 = 6,400+ 次启动，累积数分钟浪费
- 状态隔离：每条推理路径独占一个 kernel，避免变量污染

**初始化**：
```python
kernel_pool = []
for i in range(16):
    kernel = KernelManager()
    kernel.start()
    # 预导入常用库
    kernel.execute('import numpy as np')
    kernel.execute('import sympy as sp')
    kernel.execute('import math')
    kernel.execute('from itertools import *')
    kernel_pool.append(kernel)
```

**工具调用**：
```python
# 为每条 attempt 分配独占 kernel
kernel = kernel_pool[attempt_id % 16]

# 执行用户代码
result = kernel.execute(user_code)

# 代码错误处理：只保留与用户代码相关的错误信息
if result.has_error:
    stderr = clean_traceback(result.stderr)
    # 回填给 LLM
    return {"error": stderr}
```

**清理**：
```python
# 单题结束后重置 kernel
kernel.execute('%reset -f')
kernel.execute('import numpy as np; import sympy as sp; ...')
```

## 2. 熵加权投票的数学细节

### Token-level Logprobs 的收集

在 vLLM 流式输出中，每个 token 附带其 top-k logprobs：

```python
# 伪代码
for output in vllm_stream_response():
    token = output.text
    logprobs = output.logprobs  # {token_id: logprob, ...}
    # 计算该 token 的不确定性
    entropy = -sum(exp(lp) * lp for lp in logprobs.values())
    token_entropies.append(entropy)
```

### 综合置信度指标的计算

最终的置信度函数由 5 个分量组成：

#### 分量 1：平均熵 (avg_entropy)
```
H_avg = mean(entropies)  # 越低越好
```

#### 分量 2：位置加权熵 (position_weighted_entropy)
```
越接近最终答案的 token 权重越高
H_pos = sum(w_i * h_i)  where w_i = (i / N)^2

直观：最后输出的 \boxed{42} 的不确定性比中间推理的重要
```

#### 分量 3：熵的变异性 (entropy_std_dev)
```
σ_h = std(entropies)

惩罚那些置信度大幅波动的推理链
如果某条路径"忽而很确定，忽而很不确定"，说明推理不稳定
```

#### 分量 4：高熵占比 (high_entropy_ratio)
```
high_ent_ratio = count(h_i > threshold) / N

惩罚长期处于高不确定状态的路径
完全胡乱生成会导致整体高熵
```

#### 分量 5：低熵连续段奖励 (low_entropy_streak)
```
奖励那些存在持续低熵段的路径
代表"模型某个阶段非常确定"
```

### 综合置信度的组合

```python
composite_score = (
    1.0 * f_normalize(1 / avg_entropy) +           # 平均熵倒数
    0.5 * f_normalize(1 / position_weighted_entropy) +
    0.3 * f_normalize(1 / entropy_std_dev) +       # 倒数：std 越小越好
    0.2 * f_normalize(1 / high_ent_ratio) +        # 倒数：比例越小越好
    0.4 * f_normalize(low_entropy_streak_length)   # 长度越长越好
)

# 最终权重
weight = 1.0 / (0.1 + composite_score)  # 加 0.1 避免除以零
```

### 投票机制

```python
answer_scores = defaultdict(float)

for attempt_id, (answer, confidence) in enumerate(valid_answers):
    answer_scores[answer] += 1.0 / confidence
    
    # Early stop: 任一答案的累积票数达到 4
    if sum(1 for a in valid_answers if a[0] == answer) >= 4:
        break

# 最终答案：分数最高的
final_answer = max(answer_scores, key=answer_scores.get)
```

## 3. 动态时间预算算法

### 初始化

```python
TOTAL_BUDGET = 17400  # 秒
MIN_PER_PROBLEM = 270
MAX_PER_PROBLEM = 900
num_problems = 50

# 初始分配
time_per_problem = TOTAL_BUDGET / num_problems  # ~348 秒/题

# 记录
problem_times = []  # 各题实际用时
```

### 单题预算计算

处理第 i 题时：

```python
def get_time_budget(problem_id, problem_times):
    # 已用总时
    used = sum(problem_times)
    
    # 剩余题数（包括当前题）
    remaining_problems = num_problems - problem_id
    
    # 基础预留（为后续题保留最小时间）
    reserved = remaining_problems * MIN_PER_PROBLEM
    
    # 当前题可用时间上界
    max_budget = min(
        MAX_PER_PROBLEM,
        TOTAL_BUDGET - used - (remaining_problems - 1) * MIN_PER_PROBLEM
    )
    
    return max(MIN_PER_PROBLEM, max_budget)
```

### Early Stop 的时间效应

```
标准 8 路径推理：~300 秒
+ Early Stop @4 票：节省 ~30-40% 尾部时间
   → 通常在第 5-6 路就能累积 4 票
   → 减少为 ~180-200 秒
```

## 4. 离线依赖打包

### 为什么需要离线打包？

Kaggle 正式 rerun 环境：
- 不能访问互联网
- 需要在不能 pip install 的环境下运行
- 必须预先打包所有依赖

### 打包流程

```bash
# 1. 在开发环境中生成 wheels
pip wheel -r requirements.txt -w wheels/

# 2. 压缩
tar -czf wheels.tar.gz wheels/

# 3. 上传至 Kaggle 数据集
# 在 Notebook 中作为输入数据集导入

# 4. 正式 Notebook 中：安装
import tarfile
with tarfile.open('/kaggle/input/wheels/wheels.tar.gz') as tar:
    tar.extractall('wheels')

os.system('pip install --no-index --find-links wheels/ unsloth trl vllm ...')
```

### 依赖列表

关键依赖及其用途：

| 依赖 | 版本 | 用途 |
|------|------|------|
| torch | 2.x | GPU 计算 |
| transformers | 4.42+ | 模型加载 |
| vllm | latest | 推理服务 |
| unsloth | - | LoRA 优化 |
| openai | - | API 接口 |
| openai_harmony | - | Harmony 协议解析 |
| sympy | latest | 符号计算 |
| numpy | latest | 数值计算 |
| jupyter | - | IPython kernel |
| ipython | - | 交互式执行 |

## 5. FP8 KV Cache 与 Prefix Caching

### FP8 KV Cache

**为什么用 FP8？**
- K, V cache 占用显存量大（随序列长度线性增长）
- FP32 → FP8 压缩比 4:1，显存节省 75%
- 对推理准确度影响微小（<0.5%）

**vLLM 参数**：
```python
vllm_engine_args = {
    "kv_cache_dtype": "fp8_e4m3",  # E4M3 格式（指数 4 位，尾数 3 位）
    "max_model_len": 65536,
}
```

### Prefix Caching

**概念**：
- 多条并行推理路径共享相同的系统提示和题目前缀
- 可以直接复用先前计算的 KV cache，无需重新计算

**收益**：
```
不用 Prefix Caching：
  Attempt 1: 计算 [系统提示] + [题目] + [推理]
  Attempt 2: 重新计算 [系统提示] + [题目] + [推理]
  ...
  总开销 ≈ 8 × (系统提示长度 + 题目长度 + 平均推理长度)

用 Prefix Caching：
  Attempt 1: 计算 [系统提示] + [题目] + [推理]
  Attempt 2: 复用 [系统提示] + [题目] KV，仅计算 [推理]
  ...
  总开销 ≈ (系统提示长度 + 题目长度) + 8 × 平均推理长度
  节省 ≈ 7 × (系统提示长度 + 题目长度) ≈ 7% 总时间
```

## 6. 答案抽取的鲁棒性

### 抽取规则的设计

```python
def extract_answer(raw_output: str) -> Optional[int]:
    # 规则 1: \boxed{N} 格式（标准数学习题答案）
    boxed_match = re.search(r'\\boxed\{(\d+)\}', raw_output)
    if boxed_match:
        answer_str = boxed_match.group(1)
        answer = int(answer_str.replace(',', ''))  # 移除千位符
        if 0 <= answer <= 99999:
            return answer
    
    # 规则 2: "final answer is N" 格式
    final_match = re.search(r'final answer is[:\s]*(\d+)', raw_output, re.IGNORECASE)
    if final_match:
        answer = int(final_match.group(1).replace(',', ''))
        if 0 <= answer <= 99999:
            return answer
    
    # 规则 3: "answer: N" 或 "= N" 等变体
    answer_match = re.search(r'answer[:\s]*(\d+)|=\s*(\d+)', raw_output, re.IGNORECASE)
    if answer_match:
        answer_str = answer_match.group(1) or answer_match.group(2)
        answer = int(answer_str.replace(',', ''))
        if 0 <= answer <= 99999:
            return answer
    
    # 规则 4: 最后一个连续数字段
    number_matches = re.findall(r'\b(\d+)\b', raw_output)
    if number_matches:
        answer = int(number_matches[-1])
        if 0 <= answer <= 99999:
            return answer
    
    return None  # 无有效答案
```

### 特殊情况处理

**情况 1：答案是中文数字**
```python
# 如果模型输出"四十二"而非"42"
chinese_numerals = {'零': 0, '一': 1, ..., '十': 10, '百': 100, ...}
# 需要转换为阿拉伯数字
```

**情况 2：模型输出范围**
```python
# 模型可能输出"答案在 40 到 45 之间"
# 需要取区间中点或最可能值（需要额外启发式）
```

**情况 3：输出多个数字**
```python
# 在 Early Stop 中，可能已有足够答案
# 优先级：
# 1. \boxed{} 中的数字
# 2. 最后明确陈述的数字
# 3. 最后一个数字
```

## 7. 对可复现性的深度分析

### 浮点非确定性的来源

在 GPU 上，相同的浮点计算在不同条件下可能产生微小不同的结果：

```python
# 示例：矩阵乘法
# 由于累加顺序不同，结果可能在最后几位不同
result_run1 = matrix_mult_gpu(A, B)
result_run2 = matrix_mult_gpu(A, B)
# 可能 result_run1 != result_run2，差异 < 1e-6

# 这在推理中被放大：
# logit 微小差异 → token 概率微小差异
# → seed 相同但其他条件不同时，采样可能不同
```

### 强制确定性的代价

```python
# CUDA 确定性操作通常比非确定性操作慢
# 性能损失：5-15% 取决于模型

torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
# 这会禁用很多高效但非确定性的内核
```

### 混合策略

本方案采用的策略：

```python
# 1. 设置所有 seed
set_seed(seed)

# 2. 对关键操作启用确定性
torch.use_deterministic_algorithms(True, allow_tf32=False)

# 3. 但不禁用 TF32（性能影响过大）
# 而是依赖固定 seed + 环境变量
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 4. 接受最后 1-2 位浮点差异
# （在大答案空间中不影响最终答案）
```

---

**下一步**: 详见 `engineering-insights.md` 了解工程最佳实践。
