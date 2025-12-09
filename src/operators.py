import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Union


def _nanmax(tensor, dim=None, keepdim=False):
    min_value = torch.finfo(tensor.dtype).min
    output = tensor.nan_to_num(min_value).max(dim=dim, keepdim=keepdim)[0]
    output = torch.where(output == min_value, torch.nan, output)
    return output


def _nanmin(tensor, dim=None, keepdim=False):
    max_value = torch.finfo(tensor.dtype).max
    output = tensor.nan_to_num(max_value).min(dim=dim, keepdim=keepdim)[0]
    output = torch.where(output == max_value, torch.nan, output)
    return output


def _nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output


def _nanstd(tensor, dim=None, keepdim=False):
    output = _nanvar(tensor, dim=dim, keepdim=keepdim)
    output = output.sqrt()
    return output


# ===================== Class-based Operator System =====================

class Expression(ABC):
    """Base class for all expressions including operators and constants"""
    
    @abstractmethod
    def evaluate(self, feature_data: dict) -> torch.Tensor:
        """Evaluate the expression given feature data"""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """Return string representation of the expression"""
        pass
    
    def __add__(self, other):
        from .operators import Add
        return Add(self, other)
    
    def __sub__(self, other):
        from .operators import Sub
        return Sub(self, other)
    
    def __mul__(self, other):
        from .operators import Mul
        return Mul(self, other)
    
    def __truediv__(self, other):
        from .operators import Div
        return Div(self, other)


class Constant(Expression):
    """Constant value as an expression"""
    
    def __init__(self, value: Union[int, float]):
        self.value = value
    
    def evaluate(self, feature_data: dict) -> torch.Tensor:
        # Get shape from any feature in the data
        sample_key = next(iter(feature_data.keys()))
        sample_tensor = feature_data[sample_key]
        return torch.full_like(sample_tensor, self.value)
    
    def __str__(self) -> str:
        if isinstance(self.value, float) and self.value.is_integer():
            return str(int(self.value))
        return str(self.value)


class Operator(Expression):
    """Abstract base class for operators"""
    
    @abstractmethod
    def evaluate(self, feature_data: dict) -> torch.Tensor:
        pass


class UnaryOperator(Operator):
    """Base class for unary operators"""
    
    def __init__(self, operand=None):
        self._operand = operand
    
    @abstractmethod
    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the operation on the input tensor"""
        pass
    
    def evaluate(self, feature_data: dict) -> torch.Tensor:
        x = self._operand.evaluate(feature_data) if isinstance(self._operand, Expression) else self._operand
        return self._compute(x)
    
    def __str__(self) -> str:
        name = type(self).__name__.lower()
        return f"{name}({self._operand})"


class BinaryOperator(Operator):
    """Base class for binary operators"""
    
    def __init__(self, lhs=None, rhs=None):
        self._lhs = lhs
        self._rhs = rhs
    
    @abstractmethod
    def _compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the operation on two input tensors"""
        pass
    
    def evaluate(self, feature_data: dict) -> torch.Tensor:
        x = self._lhs.evaluate(feature_data) if isinstance(self._lhs, Expression) else self._lhs
        y = self._rhs.evaluate(feature_data) if isinstance(self._rhs, Expression) else self._rhs
        return self._compute(x, y)
    
    def __str__(self) -> str:
        name = type(self).__name__
        if name == 'Add':
            return f"({self._lhs}+{self._rhs})"
        elif name == 'Sub':
            return f"({self._lhs}-{self._rhs})"
        elif name == 'Mul':
            return f"({self._lhs}*{self._rhs})"
        elif name == 'Div':
            return f"({self._lhs}/{self._rhs})"
        else:
            return f"{name.lower()}({self._lhs},{self._rhs})"


class RollingOperator(UnaryOperator):
    """Base class for rolling window operators"""
    
    def __init__(self, operand=None, window_size: int = None):
        super().__init__(operand)
        self.window_size = window_size
    
    def __str__(self) -> str:
        name = type(self).__name__.lower()
        if name.startswith('ts'):
            func_name = 'ts_' + name[2:]
        else:
            func_name = name
        if self.window_size is not None:
            return f"{func_name}({self._operand},{self.window_size})"
        else:
            return f"{func_name}({self._operand})"


# ===================== Unary Operators =====================

class Abs(UnaryOperator):
    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)
    
    def __str__(self) -> str:
        return f"abs({self._operand})"


class Log(UnaryOperator):
    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x)
    
    def __str__(self) -> str:
        return f"log({self._operand})"


class Neg(UnaryOperator):
    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        return -x
    
    def __str__(self) -> str:
        return f"(-{self._operand})"


class Inv(UnaryOperator):
    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        return 1 / x
    
    def __str__(self) -> str:
        return f"(1/{self._operand})"


class Rank(UnaryOperator):
    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.full_like(x, torch.nan)
        nan_mask = torch.isnan(x)
        valid_mask = ~nan_mask
        
        x_temp = x.clone()
        x_temp[nan_mask] = float('inf')
        
        ranks = torch.argsort(torch.argsort(x_temp, dim=1), dim=1) + 1
        ranks = ranks / valid_mask.sum(dim=1, keepdim=True)
        result[valid_mask] = ranks[valid_mask]
        return result
    
    def __str__(self) -> str:
        return f"rank({self._operand})"


# ===================== Binary Operators =====================

class Add(BinaryOperator):
    def _compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class Sub(BinaryOperator):
    def _compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x - y


class Mul(BinaryOperator):
    def _compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y


class Div(BinaryOperator):
    def _compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.where(y != 0, x / y, torch.nan)


class Max(BinaryOperator):
    def _compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.max(x, y)


class Min(BinaryOperator):
    def _compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.min(x, y)


# ===================== Rolling Operators =====================

class TsMean(RollingOperator):
    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        if self.window_size is None:
            raise ValueError("window_size must be set for TsMean")
        x_pad = F.pad(x, (0, 0, self.window_size - 1, 0), mode='constant', value=torch.nan)
        return x_pad.unfold(dimension=0, size=self.window_size, step=1).nanmean(dim=2)


class TsStd(RollingOperator):
    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        if self.window_size is None:
            raise ValueError("window_size must be set for TsStd")
        x_pad = F.pad(x, (0, 0, self.window_size - 1, 0), mode='constant', value=torch.nan)
        unfolded = x_pad.unfold(0, self.window_size, 1)
        return _nanstd(unfolded, dim=2)


class TsMax(RollingOperator):
    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        if self.window_size is None:
            raise ValueError("window_size must be set for TsMax")
        x_pad = F.pad(x, (0, 0, self.window_size - 1, 0), mode='constant', value=torch.nan)
        unfolded = x_pad.unfold(dimension=0, size=self.window_size, step=1)
        return _nanmax(unfolded, dim=2)


class TsMin(RollingOperator):
    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        if self.window_size is None:
            raise ValueError("window_size must be set for TsMin")
        x_pad = F.pad(x, (0, 0, self.window_size - 1, 0), mode='constant', value=torch.nan)
        unfolded = x_pad.unfold(dimension=0, size=self.window_size, step=1)
        return _nanmin(unfolded, dim=2)


class PctChange(RollingOperator):
    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        if self.window_size is None:
            raise ValueError("window_size must be set for PctChange")
        x_lag = self._lag(x, self.window_size)
        result = torch.where(x_lag != 0, (x - x_lag) / x_lag, torch.nan)
        return result
    
    @staticmethod
    def _lag(x: torch.Tensor, window_size: int) -> torch.Tensor:
        result = torch.full_like(x, torch.nan)
        if window_size < x.size(0):
            result[window_size:] = x[:-window_size]
        return result


class Lag(RollingOperator):
    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        if self.window_size is None:
            raise ValueError("window_size must be set for Lag")
        result = torch.full_like(x, torch.nan)
        if self.window_size < x.size(0):
            result[self.window_size:] = x[:-self.window_size]
        return result


class TsCorr(BinaryOperator):
    """Time-series correlation with window size"""
    
    def __init__(self, lhs=None, rhs=None, window_size: int = None):
        super().__init__(lhs, rhs)
        self.window_size = window_size
    
    def _compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.window_size is None:
            raise ValueError("window_size must be set for TsCorr")
        x_pad = F.pad(x, (0, 0, self.window_size - 1, 0), mode='constant', value=torch.nan)
        y_pad = F.pad(y, (0, 0, self.window_size - 1, 0), mode='constant', value=torch.nan)
        x_unfolded = x_pad.unfold(dimension=0, size=self.window_size, step=1)
        y_unfolded = y_pad.unfold(dimension=0, size=self.window_size, step=1)
        
        x_mean = x_unfolded.nanmean(dim=2, keepdim=True)
        y_mean = y_unfolded.nanmean(dim=2, keepdim=True)
        
        cov = ((x_unfolded - x_mean) * (y_unfolded - y_mean)).nanmean(dim=2)
        x_std = _nanstd(x_unfolded, dim=2)
        y_std = _nanstd(y_unfolded, dim=2)
        
        corr = torch.where((x_std > 0) & (y_std > 0), cov / (x_std * y_std), torch.nan)
        return corr
    
    def __str__(self) -> str:
        if self.window_size is not None:
            return f"ts_corr({self._lhs},{self._rhs},{self.window_size})"
        else:
            return f"ts_corr({self._lhs},{self._rhs})"


# ===================== Legacy Function-based Operators (for backward compatibility) =====================

def ops_abs(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x)


def ops_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x)


def ops_neg(x: torch.Tensor) -> torch.Tensor:
    return -x


def ops_inv(x: torch.Tensor) -> torch.Tensor:
    return 1 / x


def ops_rank(x: torch.Tensor) -> torch.Tensor:
    result = torch.full_like(x, torch.nan)
    nan_mask = torch.isnan(x)
    valid_mask = ~nan_mask
    
    x_temp = x.clone()
    x_temp[nan_mask] = float('inf')
    
    ranks = torch.argsort(torch.argsort(x_temp, dim=1), dim=1) + 1
    ranks = ranks / valid_mask.sum(dim=1, keepdim=True)
    result[valid_mask] = ranks[valid_mask]
    return result


def ops_rolling_mean(x: torch.Tensor, window_size: int) -> torch.Tensor:
    x_pad = F.pad(x, (0, 0, window_size - 1, 0), mode='constant', value=torch.nan)
    return x_pad.unfold(dimension=0, size=window_size, step=1).nanmean(dim=2)


def ops_rolling_std(x: torch.Tensor, window_size: int) -> torch.Tensor:
    x_pad = F.pad(x, (0, 0, window_size - 1, 0), mode='constant', value=torch.nan)
    unfolded = x_pad.unfold(0, window_size, 1)
    return _nanstd(unfolded, dim=2)


def ops_rolling_max(x: torch.Tensor, window_size: int) -> torch.Tensor:
    x_pad = F.pad(x, (0, 0, window_size - 1, 0), mode='constant', value=torch.nan)
    unfolded = x_pad.unfold(dimension=0, size=window_size, step=1)
    return _nanmax(unfolded, dim=2)


def ops_rolling_min(x: torch.Tensor, window_size: int) -> torch.Tensor:
    x_pad = F.pad(x, (0, 0, window_size - 1, 0), mode='constant', value=torch.nan)
    unfolded = x_pad.unfold(dimension=0, size=window_size, step=1)
    return _nanmin(unfolded, dim=2)


def ops_pct_change(x: torch.Tensor, window_size: int) -> torch.Tensor:
    x_lag = ops_lag(x, window_size)
    result = torch.where(x_lag != 0, (x - x_lag) / x_lag, torch.nan)
    return result


def ops_lag(x: torch.Tensor, window_size: int) -> torch.Tensor:
    result = torch.full_like(x, torch.nan)
    if window_size < x.size(0):
        result[window_size:] = x[:-window_size]
    return result


def ops_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


def ops_subtract(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x - y


def ops_multiply(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x * y


def ops_divide(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.where(y != 0, x / y, torch.nan)


def ops_max(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.max(x, y)


def ops_min(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.min(x, y)


def ops_roll_corr(x: torch.Tensor, y: torch.Tensor, window_size: int) -> torch.Tensor:
    x_pad = F.pad(x, (0, 0, window_size - 1, 0), mode='constant', value=torch.nan)
    y_pad = F.pad(y, (0, 0, window_size - 1, 0), mode='constant', value=torch.nan)
    x_unfolded = x_pad.unfold(dimension=0, size=window_size, step=1)
    y_unfolded = y_pad.unfold(dimension=0, size=window_size, step=1)
    
    x_mean = x_unfolded.nanmean(dim=2, keepdim=True)
    y_mean = y_unfolded.nanmean(dim=2, keepdim=True)
    
    cov = ((x_unfolded - x_mean) * (y_unfolded - y_mean)).nanmean(dim=2)
    x_std = _nanstd(x_unfolded, dim=2)
    y_std = _nanstd(y_unfolded, dim=2)
    
    corr = torch.where((x_std > 0) & (y_std > 0), cov / (x_std * y_std), torch.nan)
    return corr


def generate_operators(window_sizes, include_constants=True):
    """Generate operator dictionaries with class instances
    
    Args:
        window_sizes: List of window sizes for rolling operators
        include_constants: Whether to include constant values in the action space
    
    Returns:
        unary_ops: Dict mapping operator names to operator classes
        binary_ops: Dict mapping operator names to operator classes  
        rolling_ops: Dict mapping rolling operator names to operator classes
        constants: Dict mapping constant names to Constant instances (if include_constants=True)
        delta_times: List of window sizes
    """
    unary_ops = {
        'Abs': Abs,
        'Log': Log,
        'Neg': Neg,
        'Inv': Inv,
        'Rank': Rank
    }
    
    binary_ops = {
        'Add': Add,
        'Sub': Sub,
        'Mul': Mul,
        'Div': Div,
        'Max': Max,
        'Min': Min
    }
    
    # Rolling operators that take window size as parameter
    rolling_ops = {
        'TsMean': TsMean,
        'TsStd': TsStd,
        'TsMax': TsMax,
        'TsMin': TsMin,
        'PctChange': PctChange,
        'Lag': Lag,
    }
    
    # Rolling binary operators
    rolling_binary_ops = {
        'TsCorr': TsCorr,
    }
    
    constants = {}
    if include_constants:
        constants = {
            'Const_0': Constant(0),
            'Const_1': Constant(1),
            'Const_-1': Constant(-1),
            'Const_0.5': Constant(0.5),
            'Const_2': Constant(2),
        }

    return unary_ops, binary_ops, rolling_ops, rolling_binary_ops, constants, window_sizes