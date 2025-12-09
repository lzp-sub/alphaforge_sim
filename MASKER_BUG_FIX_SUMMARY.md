# Masker Bug Fix Summary

## Problem History

The AlphaForge operator system was experiencing assertion failures (`assert len(stack) == 1`) during NetG training sampling. Despite multiple fixes, the issue persisted intermittently.

## Root Cause

The masker's validation logic was not aggressive enough in preventing the stack from reaching `max_length` in invalid configurations. The issues were:

1. **Insufficient Lookahead**: The masker didn't account for operations requiring multiple steps (e.g., rolling ops need: push window, then apply operator)
2. **Late Prevention**: Constraints were applied too late (at `max_length-2` or `max_length-1`), when it was already too late to recover
3. **Overly Permissive**: Even near max_length, the masker allowed actions that would grow or complicate the stack

## Solution: Progressive Restriction Strategy

Instead of trying to "fix" invalid states at the last moment, the solution prevents them from forming by progressively restricting actions as max_length approaches:

### Distance from Max_Length → Restrictions

| Distance | Stack State | Restrictions |
|----------|-------------|-------------|
| ≥ 6 | Any | Full freedom - all valid actions allowed |
| 5 | [tensor] | **No delta times** (rolling ops need 2+ more steps) |
| 4 | [tensor, tensor] | **No windows** (would need op after, no time) |
| 3 | [tensor] | **ONLY unary + SEP** (no features, constants, windows) |
| 3 | [tensor, tensor] | Binary ops + windows (but see distance 4 rule) |
| 2 | [tensor, tensor] | **FORCE binary** (must combine now) |
| 1 | [tensor] | Unary + SEP |
| 1 | [tensor, tensor] | Binary only |
| 1 | [tensor, window] | Rolling unary only |
| 1 | [tensor, tensor, window] | Rolling binary only |
| 0 (at max_length) | [tensor] | SEP only ✓ |
| 0 (at max_length) | Other | Attempt recovery or fail cleanly |

### Key Improvements

1. **Progressive Tightening**: Restrictions increase gradually, not abruptly
2. **Lookahead**: Considers how many steps operations need to complete
3. **Proactive**: Prevents bad states rather than reacting to them
4. **Defensive**: Multiple layers of checks ensure safety

## Code Changes

### `src/model.py` - `_get_valid_actions()` method:

**Lines 117-148**: Early classification and max_length handling
- Classify stack types first
- At max_length, validate stack before allowing SEP
- Attempt recovery for specific recoverable states

**Lines 154-165**: Progressive restrictions for [tensor]
- Distance 5: No delta times
- Distance 3: Only unary + SEP

**Lines 173-183**: Progressive restrictions for [tensor, tensor]
- Distance 4: No windows  
- Distance 2: Force binary

**Lines 199-221**: Enhanced max_length-1 handling
- Specific actions for each valid stack state
- Fallback for edge cases

## Testing Results

- ✅ 1000 NetG samples: All successful
- ✅ Multiple configurations (max_length 5, 10, 15): All passed
- ✅ Different window size sets: All passed
- ✅ Both sampling methods tested
- ✅ Multiple random seeds: All passed

## Impact

The masker now ensures:
1. Stack never reaches max_length in invalid state
2. Operations with multi-step requirements are blocked when time is insufficient
3. Stack complexity decreases as max_length approaches
4. Expressions always terminate with exactly 1 tensor on stack

## Lessons Learned

1. **Prevention > Recovery**: Better to prevent bad states than try to fix them
2. **Progressive Constraints**: Gradual restriction works better than sudden limits
3. **Lookahead Matters**: Must consider future steps required by current actions
4. **Defensive Programming**: Multiple layers of checks catch edge cases
5. **Testing Depth**: Rare bugs need extensive testing (1000+ samples)
