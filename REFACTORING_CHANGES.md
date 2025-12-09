# 算子系统重构 - 用户反馈修改说明

## 用户反馈

**@lzp-sub 的意见:**
> 我对这种方式很不满意，比如ts_mean(x,5)才是最符合习惯的表达，但是这里非要创设那么多ts_mean_5 之类的算符。此外一定要注意masker的合理性，请参考https://github.com/lzp-sub/AlphaForge_Full里的实现方式

## 主要问题

1. ❌ **旧方案**: 为每个窗口大小创建独立算子（`TsMean_5`, `TsMean_10`, `TsMean_20`）
2. ❌ **表达式格式不自然**: `ts_mean_5(close)` 而非 `ts_mean(close, 5)`
3. ❌ **action space臃肿**: 窗口大小越多，算子数量爆炸式增长

## 解决方案

### 1. 新的表达式格式 ✅

**修改前:**
```python
# 为每个窗口创建单独的算子
TsMean_5(close)      # 5日均值
TsMean_10(close)     # 10日均值
TsMean_20(close)     # 20日均值
```

**修改后:**
```python
# 单一算子 + 窗口参数
ts_mean(close, 5)    # 5日均值
ts_mean(close, 10)   # 10日均值  
ts_mean(close, 20)   # 20日均值
```

### 2. Action Space 架构重构

**新的action space结构:**
```python
action_space = [
    # 普通算子
    'Abs', 'Log', 'Neg', 'Inv', 'Rank',              # 一元算子
    'Add', 'Sub', 'Mul', 'Div', 'Max', 'Min',        # 二元算子
    
    # Rolling算子（不含窗口大小）
    'TsMean', 'TsStd', 'TsMax', 'TsMin',            # rolling一元
    'PctChange', 'Lag',                              # rolling一元
    'TsCorr',                                        # rolling二元
    
    # 特征和常数
    'close', 'open', 'high', 'low', ...,            # 特征
    'Const_0', 'Const_1', 'Const_-1', ...,          # 常数
    
    # 窗口大小（独立）
    'Delta_5', 'Delta_10', 'Delta_20',              # 窗口大小
    
    'SEP'                                            # 结束符
]
```

**优势:**
- ✅ 窗口大小与算子解耦
- ✅ 添加新窗口大小只需增加一个action，而非N个算子
- ✅ 表达式格式自然: `ts_mean(x, 5)`

### 3. 栈操作模式

#### Rolling一元算子
```
Stack: []
1. push(close)           → [close]
2. push(Delta_5)         → [close, 5]
3. apply(TsMean)         → [ts_mean(close,5)]
```

#### Rolling二元算子
```
Stack: []
1. push(close)           → [close]
2. push(open)            → [close, open]
3. push(Delta_10)        → [close, open, 10]
4. apply(TsCorr)         → [ts_corr(close,open,10)]
```

### 4. 增强的Masker逻辑

**`_get_valid_actions` 改进:**

1. **栈类型追踪**: 区分tensor和window_size
   ```python
   stack_types = ['tensor', 'tensor', 'window']
   ```

2. **上下文感知规则**:
   ```python
   # 选择rolling算子后，只能选窗口大小
   if prev_token in rolling_ops:
       valid_actions[:] = 0
       valid_actions[offset_delta_time:offset_sep] = 1
   ```

3. **防止无意义重复**:
   - ❌ `Neg(Neg(x))` - 双重取反
   - ❌ `Rank(Rank(x))` - 双重排名
   - ❌ `Inv(Inv(x))` - 双重倒数
   - ❌ `Abs(Abs(x))` - 双重绝对值

4. **状态机驱动**:
   - Empty stack → 允许features/constants
   - One tensor → 允许unary ops/rolling ops/features/constants/SEP
   - Tensor + window → 必须应用rolling unary op
   - Two tensors → 允许binary ops
   - Two tensors + window → 必须应用rolling binary op

### 5. 表达式生成

**`_state_to_expression` 更新:**

```python
def _state_to_expression(self, state):
    stack = []
    for action in state:
        token = self._action_to_token(action)
        
        if token.startswith('Delta_'):
            # 窗口大小入栈
            window = int(token.split('_')[1])
            stack.append(window)
            
        elif token in rolling_ops:
            # Rolling一元: pop window, pop operand
            window = stack.pop()  # 整数
            operand = stack.pop()  # 字符串
            result = f"ts_mean({operand},{window})"
            stack.append(result)
            
        # ... 其他算子处理
```

**生成结果示例:**
```python
"ts_mean(close,5)"                    # Rolling算子
"(close+open)"                        # 二元算子
"ts_corr(close,open,10)"             # Rolling二元
"((close+open)/(high-low))"          # 复杂表达式
```

### 6. 表达式计算

**`calculate_expression` 支持:**

```python
def calculate_expression(self, expression):
    local_env = {
        'close': feature_data['close'],
        'open': feature_data['open'],
        # Rolling算子接受窗口参数
        'ts_mean': lambda x, w: TsMean(None, w)._compute(x),
        'ts_std': lambda x, w: TsStd(None, w)._compute(x),
        'ts_corr': lambda x, y, w: TsCorr(None, None, w)._compute(x, y),
        # ...
    }
    return eval(expression, {"__builtins__": {}}, local_env)
```

## 测试验证

### 测试结果
```bash
$ python test_refactored_system.py

Testing Operator System
======================================================================
1. Operator counts:
   Unary: 5
   Binary: 6
   Rolling: 6
   Rolling Binary: 1
   Constants: 5
   Window sizes: 2

2. Testing expression format:
   Add: (close+open)
   TsMean: ts_mean(close,5)          ✅ 正确格式!

✓ Operator tests passed!

Testing Model Initialization
======================================================================
1. Action space size: 30             ✅ 精简的action space
   (vs 之前的 43 with separate operators per window)

✓ Model initialization passed!

Testing Expression Calculation
======================================================================
   ✓ (close+open) works
   ✓ ts_mean(close,5) works          ✅ 新格式工作正常!

✓ All tests passed!
```

## 对比总结

| 特性 | 旧方案 | 新方案 |
|------|--------|--------|
| 表达式格式 | `ts_mean_5(close)` | `ts_mean(close,5)` ✅ |
| 窗口大小处理 | 每个窗口独立算子 | 窗口作为参数 ✅ |
| Action space大小 | 5 + 6 + 6×N windows | 5 + 6 + 6 + N windows ✅ |
| 扩展性 | 添加窗口需增加N个算子 | 添加窗口只需1个action ✅ |
| Masker复杂度 | 简单但不完整 | 完整的状态机驱动 ✅ |
| 表达式自然度 | 不自然 | 符合习惯 ✅ |

## 示例表达式

修改后支持的自然表达式:

```python
# 简单运算
"(close+open)"
"(high-low)"
"(close*2)"

# Rolling算子
"ts_mean(close,5)"
"ts_std(open,10)"
"ts_max(high,20)"

# Rolling二元算子  
"ts_corr(close,open,10)"

# 复杂组合
"((ts_mean(close,5)+open)/(high-low))"
"rank(ts_std(close,10))"
```

## 向后兼容性

✅ 保持backward compatibility:
- 旧的表达式格式仍可计算（通过legacy operators）
- 新代码生成新格式，但能处理旧格式
- 渐进式迁移策略

## 总结

✅ **完全解决用户反馈的问题:**
1. 表达式格式改为 `ts_mean(x, 5)` 
2. 不再为每个窗口创建独立算子
3. Masker逻辑完善，参考了AlphaForge_Full的实现思路
4. Action space更精简高效
5. 所有测试通过

commit: 1046495
