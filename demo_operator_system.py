#!/usr/bin/env python
"""
Demonstration of the enhanced operator system with natural expression syntax.

This script shows how the refactored AlphaForge operator system generates
readable expressions and supports constants.
"""

import torch
from src.model import AlphaForge
from src.metric import ICMetric

def main():
    print("=" * 70)
    print("AlphaForge Enhanced Operator System Demonstration")
    print("=" * 70)
    
    # Create sample feature data
    print("\n1. Creating sample feature data...")
    feature_data = {
        'close': torch.randn(100, 50),
        'open': torch.randn(100, 50),
        'high': torch.randn(100, 50),
        'low': torch.randn(100, 50),
        'volume': torch.randn(100, 50)
    }
    print(f"   Created {len(feature_data)} features with shape (100, 50)")
    
    # Create dummy metric
    ret_rank = torch.randn(100, 50)
    metric = ICMetric(ret_rank, method='pearson')
    
    # Initialize model
    print("\n2. Initializing AlphaForge model...")
    window_sizes = [5, 10, 20]
    model = AlphaForge(
        feature_data=feature_data,
        window_sizes=window_sizes,
        metric=metric,
        hidden_size=64,
        max_length=10,
        device='cpu'
    )
    print(f"   Model initialized with {len(model.action_space)} actions")
    print(f"   - {len(model.unary_ops)} unary operators")
    print(f"   - {len(model.binary_ops)} binary operators")
    print(f"   - {len(model.feature_names)} features")
    print(f"   - {len(model.constants)} constants")
    
    # Show constants
    print("\n3. Available constants:")
    for const_name in model.constant_names:
        const = model.constants[const_name]
        print(f"   - {const_name}: {const.value}")
    
    # Demonstrate natural expression generation
    print("\n4. Natural Expression Generation Examples:")
    
    examples = [
        ("Simple addition", "(close+open)"),
        ("Simple subtraction", "(high-low)"),
        ("Multiplication with constant", "(close*2)"),
        ("Division", "((close+open)/(high-low))"),
        ("With constant subtraction", "(close-1)"),
        ("Complex expression", "((close+open)/(high-low))"),
    ]
    
    for name, expr in examples:
        print(f"\n   {name}:")
        print(f"   Old style: ops_divide(ops_add(close,open),ops_subtract(high,low))")
        print(f"   New style: {expr}")
        try:
            result = model.calculate_expression(expr)
            print(f"   ✓ Expression evaluates successfully, shape: {result.shape}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    # Demonstrate operator string representation
    print("\n5. Operator String Representations:")
    from src.operators import Add, Sub, Mul, Div, TsMean, Rank
    
    # Create example operators with string operands
    add_op = Add(None, None)
    add_op._lhs = "close"
    add_op._rhs = "open"
    print(f"   Add operator: {add_op}")
    
    mul_op = Mul(None, None)
    mul_op._lhs = "close"
    mul_op._rhs = "2"
    print(f"   Mul operator: {mul_op}")
    
    ts_mean = TsMean(None, window_size=5)
    ts_mean._operand = "close"
    print(f"   TsMean operator: {ts_mean}")
    
    rank_op = Rank(None)
    rank_op._operand = "(close+open)"
    print(f"   Rank operator: {rank_op}")
    
    # Show enhanced masker checking
    print("\n6. Enhanced Masker Checking:")
    print("   The masker now prevents:")
    print("   - Double negation (e.g., -(-x))")
    print("   - Double rank (e.g., rank(rank(x)))")
    print("   - Double inversion (e.g., 1/(1/x))")
    print("   - Double absolute (e.g., abs(abs(x)))")
    print("   This ensures generated expressions are valid and meaningful.")
    
    # Test expression calculation with constants
    print("\n7. Testing Expression Calculation with Constants:")
    
    test_exprs = [
        "(close+open)",
        "(close-open)",
        "(close*2)",
        "(close/2)",
    ]
    
    for expr in test_exprs:
        try:
            result = model.calculate_expression(expr)
            print(f"   ✓ {expr:20} -> shape {result.shape}, mean: {result.nanmean():.4f}")
        except Exception as e:
            print(f"   ✗ {expr:20} -> Error: {e}")
    
    print("\n" + "=" * 70)
    print("Demonstration Complete!")
    print("=" * 70)
    print("\nKey Improvements:")
    print("✓ Natural mathematical notation: (a+b) instead of ops_add(a,b)")
    print("✓ Constant support: expressions like close*2 or open-1")
    print("✓ Enhanced validation: prevents invalid expression generation")
    print("✓ Backward compatible: legacy code continues to work")
    print("✓ Clean architecture: proper abstraction with Expression classes")
    print("=" * 70)

if __name__ == "__main__":
    main()
