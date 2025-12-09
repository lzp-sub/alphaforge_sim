import warnings
warnings.filterwarnings('ignore')

import torch
import joblib
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from loguru import logger
from copy import deepcopy
from torch.nn import functional as F
from typing import List, Dict, Callable
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.categorical import Categorical

from .net import NetP, NetG
from .operators import generate_operators, ops_rank, _nanstd


class AlphaForge:
    def __init__(
        self,
        feature_data: Dict[str, torch.Tensor],
        window_sizes: List[int],
        metric: Callable,
        hidden_size=64,
        max_length=20,
        device='cuda',
        ):

        self.feature_data = feature_data
        self.metric = metric
        self.max_length = max_length
        self.device = device
        self.window_sizes = window_sizes

        self.sep = ['SEP']
        self.unary_ops, self.binary_ops, self.rolling_ops, self.rolling_binary_ops, self.constants, self.delta_times = generate_operators(window_sizes, include_constants=True)
        
        self.unary_op_names = list(self.unary_ops.keys())
        self.binary_op_names = list(self.binary_ops.keys())
        self.rolling_op_names = list(self.rolling_ops.keys())
        self.rolling_binary_op_names = list(self.rolling_binary_ops.keys())
        self.constant_names = list(self.constants.keys())
        self.feature_names = list(self.feature_data.keys())
        self.delta_time_names = [f'Delta_{dt}' for dt in self.delta_times]
        
        # Action space: unary_ops + binary_ops + rolling_ops + rolling_binary_ops + features + constants + delta_times + SEP
        self.action_space = (self.unary_op_names + self.binary_op_names + 
                           self.rolling_op_names + self.rolling_binary_op_names +
                           self.feature_names + self.constant_names + 
                           self.delta_time_names + self.sep)
        self.action_size = len(self.action_space)

        # Calculate offsets
        self.offset_unary = 0
        self.offset_binary = self.offset_unary + len(self.unary_ops)
        self.offset_rolling = self.offset_binary + len(self.binary_ops)
        self.offset_rolling_binary = self.offset_rolling + len(self.rolling_ops)
        self.offset_feature = self.offset_rolling_binary + len(self.rolling_binary_ops)
        self.offset_constant = self.offset_feature + len(self.feature_data)
        self.offset_delta_time = self.offset_constant + len(self.constants)
        self.offset_sep = self.offset_delta_time + len(self.delta_times)

        self.netp = NetP(self.action_size, hidden_size, dropout=0.1).to(self.device)
        self.netg = NetG(self.action_size, hidden_size, self.max_length + 1).to(self.device)

        self.alpha_pool = {}

    def _reset_net(self):
        self.netp.reset_params()
        self.netg.reset_params()

    def _action_to_token(self, action: int) -> str: 
        if action < self.offset_unary or action > self.offset_sep:
            raise ValueError(f"Invalid action: {action}")
        elif action < self.offset_binary:
            return self.unary_op_names[action - self.offset_unary]
        elif action < self.offset_rolling:
            return self.binary_op_names[action - self.offset_binary]
        elif action < self.offset_rolling_binary:
            return self.rolling_op_names[action - self.offset_rolling]
        elif action < self.offset_feature:
            return self.rolling_binary_op_names[action - self.offset_rolling_binary]
        elif action < self.offset_constant:
            return self.feature_names[action - self.offset_feature]
        elif action < self.offset_delta_time:
            return self.constant_names[action - self.offset_constant]
        elif action < self.offset_sep:
            delta_idx = action - self.offset_delta_time
            return f'Delta_{self.delta_times[delta_idx]}'
        elif action == self.offset_sep:
            return 'SEP'
        else:
            raise ValueError(f"Invalid action: {action}")

    def _get_valid_actions(self, stack, state):
        """Get valid actions based on current stack state
        
        Stack elements can be:
        - Tensors (features, constants, computed results)
        - Window sizes (integers)
        
        Rules for rolling operators:
        - Rolling unary needs: tensor, window (in that order on stack), then operator
        - Rolling binary needs: tensor, tensor, window (in that order on stack), then operator
        
        So when we want to use rolling operators:
        - First push operand(s)
        - Then push window size
        - Then apply rolling operator
        """
        valid_actions = torch.zeros(self.action_size, dtype=torch.bool, device=self.device)

        if len(state) == self.max_length:  # only SEP is allowed
            valid_actions[self.offset_sep] = 1
            return valid_actions

        # Classify stack elements
        stack_types = []
        for elem in stack:
            if isinstance(elem, int):
                stack_types.append('window')
            elif isinstance(elem, torch.Tensor):
                stack_types.append('tensor')
            else:
                stack_types.append('unknown')

        # Decision based on stack state
        if len(stack) == 0:
            # Empty stack: allow features and constants only
            valid_actions[self.offset_feature:self.offset_delta_time] = 1
            
        elif len(stack) == 1:
            if stack_types[0] == 'tensor':
                # One tensor: can do several things
                # - Apply unary op to it
                # - Add another tensor/constant/feature for binary op
                # - Add a window size for rolling unary op
                # - Finish with SEP
                valid_actions[self.offset_unary:self.offset_binary] = 1  # unary ops only (not binary/rolling yet)
                valid_actions[self.offset_feature:self.offset_sep] = 1  # features, constants, delta times, SEP
                
            elif stack_types[0] == 'window':
                # Just a window on stack - invalid, need tensor first
                # This shouldn't happen, but set no valid actions
                pass
                
        elif len(stack) == 2:
            if stack_types == ['tensor', 'tensor']:
                # Two tensors: can apply binary ops or unary to either, or add window for rolling binary
                valid_actions[self.offset_unary:self.offset_rolling] = 1  # unary and binary ops
                valid_actions[self.offset_delta_time:self.offset_sep] = 1  # can add window for rolling binary
                
            elif stack_types == ['tensor', 'window']:
                # Tensor + window: can ONLY apply rolling unary op now
                valid_actions[self.offset_rolling:self.offset_rolling_binary] = 1
                
            elif stack_types == ['window', 'tensor']:
                # Wrong order - invalid
                pass
                
        elif len(stack) == 3:
            if stack_types == ['tensor', 'tensor', 'window']:
                # Two tensors + window: can ONLY apply rolling binary op
                valid_actions[self.offset_rolling_binary:self.offset_feature] = 1
            else:
                # Other combinations shouldn't happen or are invalid
                pass
                
        # Special handling at max_length - 1
        if len(state) == self.max_length - 1:
            # Must finish in one move - need exactly 1 tensor result
            valid_actions[:] = 0
            if len(stack) == 1 and stack_types[0] == 'tensor':
                # Can apply unary op to finish
                valid_actions[self.offset_unary:self.offset_binary] = 1
            elif len(stack) == 2:
                if stack_types == ['tensor', 'tensor']:
                    # Can apply binary op to finish
                    valid_actions[self.offset_binary:self.offset_rolling] = 1
                elif stack_types == ['tensor', 'window']:
                    # Can apply rolling unary op to finish
                    valid_actions[self.offset_rolling:self.offset_rolling_binary] = 1
            elif len(stack) == 3:
                if stack_types == ['tensor', 'tensor', 'window']:
                    # Can apply rolling binary op to finish
                    valid_actions[self.offset_rolling_binary:self.offset_feature] = 1
        
        # Enhanced masker: prevent meaningless double operations
        if state.shape[0] > 0:
            last_action = state[-1].argmax().item()
            last_token = self._action_to_token(last_action)
            
            # Prevent double negation
            if 'Neg' in last_token:
                for i, name in enumerate(self.unary_op_names):
                    if 'Neg' in name:
                        valid_actions[self.offset_unary + i] = 0
            
            # Prevent double rank
            if 'Rank' in last_token:
                for i, name in enumerate(self.unary_op_names):
                    if 'Rank' in name:
                        valid_actions[self.offset_unary + i] = 0
            
            # Prevent double inversion
            if 'Inv' in last_token:
                for i, name in enumerate(self.unary_op_names):
                    if 'Inv' in name:
                        valid_actions[self.offset_unary + i] = 0
            
            # Prevent double absolute
            if 'Abs' in last_token:
                for i, name in enumerate(self.unary_op_names):
                    if 'Abs' in name:
                        valid_actions[self.offset_unary + i] = 0

        return valid_actions

    def _step_action(self, stack: List, action: int):
        """Execute an action on the stack
        
        Actions can be:
        - Unary operators: pop 1 tensor, apply op, push result
        - Binary operators: pop 2 tensors, apply op, push result  
        - Rolling unary operators: pop 1 window + 1 tensor, apply op, push result
        - Rolling binary operators: pop 1 window + 2 tensors, apply op, push result
        - Features: push feature tensor
        - Constants: push constant tensor
        - Delta times: push window size (integer)
        - SEP: do nothing
        """
        if action < self.offset_unary or action > self.offset_sep:
            raise ValueError(f"Invalid action: {action}")

        token = self._action_to_token(action)
        
        if action < self.offset_binary:  # Unary operators
            operand = stack.pop()
            if not isinstance(operand, torch.Tensor):
                raise ValueError("Unary operator requires tensor operand")
            operator_class = self.unary_ops[token]
            operator = operator_class(operand)
            result = operator._compute(operand)
            stack.append(result)
            
        elif action < self.offset_rolling:  # Binary operators
            rhs = stack.pop()
            lhs = stack.pop()
            if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
                raise ValueError("Binary operator requires two tensor operands")
            operator_class = self.binary_ops[token]
            operator = operator_class(lhs, rhs)
            result = operator._compute(lhs, rhs)
            stack.append(result)
            
        elif action < self.offset_rolling_binary:  # Rolling unary operators
            window_size = stack.pop()
            operand = stack.pop()
            if not isinstance(operand, torch.Tensor) or not isinstance(window_size, int):
                raise ValueError("Rolling unary operator requires tensor and window size")
            operator_class = self.rolling_ops[token]
            operator = operator_class(operand, window_size)
            result = operator._compute(operand)
            stack.append(result)
            
        elif action < self.offset_feature:  # Rolling binary operators
            window_size = stack.pop()
            rhs = stack.pop()
            lhs = stack.pop()
            if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor) or not isinstance(window_size, int):
                raise ValueError("Rolling binary operator requires two tensors and window size")
            operator_class = self.rolling_binary_ops[token]
            operator = operator_class(lhs, rhs, window_size)
            result = operator._compute(lhs, rhs)
            stack.append(result)
            
        elif action < self.offset_constant:  # Features
            stack.append(self.feature_data[token])
            
        elif action < self.offset_delta_time:  # Constants
            constant = self.constants[token]
            result = constant.evaluate(self.feature_data)
            stack.append(result)
            
        elif action < self.offset_sep:  # Delta times (window sizes)
            delta_idx = action - self.offset_delta_time
            window_size = self.delta_times[delta_idx]
            stack.append(window_size)
            
        elif action == self.offset_sep:  # SEP
            pass

    def _sample_factor(self):
        stack = []
        state = torch.zeros(self.max_length + 1, self.action_size, device=self.device, dtype=torch.float32) # +1 for SEP

        for t in range(self.max_length + 1):
            valid_actions = self._get_valid_actions(stack, state[:t])
            masked = torch.where(valid_actions, 1.0, -1e5)
            action = Categorical(logits=masked).sample().item()

            state[t, action] = 1.0
            self._step_action(stack, action)
            
            if action == self.offset_sep:
                break
        
        assert len(stack) == 1
        factor = stack.pop()
        metric = self.metric(factor)
        return state, metric
    
    def _sample_factor_from_logits(self, logits):
        stack, one_hot = [], []
        state = torch.zeros(self.max_length + 1, self.action_size, device=self.device, dtype=torch.float32) # +1 for SEP

        for t in range(self.max_length + 1):
            valid_actions = self._get_valid_actions(stack, state[:t])
            masked = torch.where(valid_actions, logits[t], -1e5)
            one_hot.append(masked)

            with torch.no_grad():
                action = masked.argmax().item()

            state[t, action] = 1.0
            self._step_action(stack, action)
            
            if action == self.offset_sep:
                break

        assert len(stack) == 1
        factor = stack.pop()
        metric = self.metric(factor)

        one_hot = torch.stack(one_hot, dim=0)
        one_hot = F.gumbel_softmax(one_hot, hard=True)
        one_hot = F.pad(one_hot, (0, 0, 0, self.max_length + 1 - one_hot.shape[0]), value=0)

        return state, metric, one_hot
    
    def _state_to_expression(self, state) -> str:
        """Convert state to expression string with natural syntax
        
        Builds expression string from state by simulating stack operations.
        Stack elements can be strings (sub-expressions) or integers (window sizes).
        """
        stack = []
        for t in range(state.shape[0]):
            action = state[t].argmax().item()
            token = self._action_to_token(action)

            if token in self.feature_names:
                # Feature
                stack.append(token)
                
            elif token in self.constant_names:
                # Constant
                const = self.constants[token]
                stack.append(str(const))
                
            elif token.startswith('Delta_'):
                # Window size
                window = int(token.split('_')[1])
                stack.append(window)
                
            elif token in self.unary_op_names:
                # Unary operator
                if len(stack) == 0:
                    break
                operand = stack.pop()
                operator_class = self.unary_ops[token]
                temp_op = operator_class(None)
                temp_op._operand = operand
                stack.append(str(temp_op))
                
            elif token in self.binary_op_names:
                # Binary operator
                if len(stack) < 2:
                    break
                rhs = stack.pop()
                lhs = stack.pop()
                operator_class = self.binary_ops[token]
                temp_op = operator_class(None, None)
                temp_op._lhs = lhs
                temp_op._rhs = rhs
                stack.append(str(temp_op))
                
            elif token in self.rolling_op_names:
                # Rolling unary operator: needs window then operand
                if len(stack) < 2:
                    break
                window = stack.pop()
                operand = stack.pop()
                if not isinstance(window, int):
                    break
                operator_class = self.rolling_ops[token]
                temp_op = operator_class(None, window)
                temp_op._operand = operand
                stack.append(str(temp_op))
                
            elif token in self.rolling_binary_op_names:
                # Rolling binary operator: needs window then two operands
                if len(stack) < 3:
                    break
                window = stack.pop()
                rhs = stack.pop()
                lhs = stack.pop()
                if not isinstance(window, int):
                    break
                operator_class = self.rolling_binary_ops[token]
                temp_op = operator_class(None, None, window)
                temp_op._lhs = lhs
                temp_op._rhs = rhs
                stack.append(str(temp_op))
                
            elif token == 'SEP':
                break
                
        return stack[0] if len(stack) > 0 else ""        

    def _evaluate_factor(self, expr, corr_threshold) -> bool:
        factor = self.calculate_expression(expr)
        x_s = (factor - factor.nanmean(dim=1, keepdim=True)) / _nanstd(factor, dim=1, keepdim=True)
        for k, v in self.alpha_pool.items():
            alpha = self.calculate_expression(k)
            y_s = (alpha - alpha.nanmean(dim=1, keepdim=True)) / _nanstd(alpha, dim=1, keepdim=True)
            corr = (x_s * y_s).nanmean(dim=1).nanmean()
            corr = torch.abs(corr)
            if corr > corr_threshold:
                return False
        return True

    def _train_netp(self, sampled_matrix, sampled_metrics, batch_size, num_epochs, learning_rate, early_stopping):
        optimizer = torch.optim.Adam(self.netp.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        X, y = torch.stack(sampled_matrix, dim=0), torch.stack(sampled_metrics, dim=0)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        
        best_val_loss = np.inf
        patience_counter = 0
        best_model = deepcopy(self.netp)

        for epoch in range(num_epochs):
            self.netp.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = self.netp(x_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            self.netp.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    y_pred = self.netp(x_batch)
                    loss = criterion(y_pred, y_batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            # logging
            logger.info(
                f'NetP Epochs: {epoch + 1}, '
                f'Train Loss: {train_loss}, '
                f'Val Loss: {val_loss}'
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model = deepcopy(self.netp)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    logger.info(f'Early stopping at epoch {epoch + 1}')
                    break

        self.netp = deepcopy(best_model)

    def _train_netg(self, sampled_states, sampled_metrics, batch_size, num_epochs, learning_rate, early_stopping, ic_threshold, corr_threshold):
        optimizer = torch.optim.Adam(self.netg.parameters(), lr=learning_rate)
        best_metric = -np.inf
        patience_counter = 0
        
        self.netp.eval()
        self.netg.train()

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            z_1 = torch.randn(batch_size, self.action_size, device=self.device)
            z_2 = torch.randn(batch_size, self.action_size, device=self.device)
            logits_1, logits_2 = self.netg(z_1), self.netg(z_2)
            
            batch_metric = 0.0
            one_hots_1, one_hots_2 = [], []

            for i in tqdm(range(batch_size), desc='Sampling factors from NetG'):
                state, metric, one_hot = self._sample_factor_from_logits(logits_1[i])
                sampled_states.append(state)
                sampled_metrics.append(metric)
                one_hots_1.append(one_hot)
                batch_metric += metric.item()

                if metric > ic_threshold:
                    expr = self._state_to_expression(state)
                    if expr not in self.alpha_pool and self._evaluate_factor(expr, corr_threshold):
                        self.alpha_pool[expr] = metric.item()

                state, metric, one_hot = self._sample_factor_from_logits(logits_2[i])
                one_hots_2.append(one_hot)

            one_hots_1, one_hots_2 = torch.stack(one_hots_1, dim=0), torch.stack(one_hots_2, dim=0)
            pred_1, pred_2 = self.netp(one_hots_1), self.netp(one_hots_2)

            similarity = torch.sum(one_hots_1 * one_hots_2, dim=-1).sum(dim=-1) / (self.max_length + 1)
            similarity = torch.relu(similarity - corr_threshold) ** 2
            similarity_loss = similarity.mean()
            pred_loss = 1 - pred_1.mean()

            loss = pred_loss + similarity_loss
            loss.backward()
            optimizer.step()

            batch_metric /= batch_size
            logger.info(
                f'NetG Epochs: {epoch + 1}, '
                f'Loss: {loss.item()}, '
                f'Metric: {batch_metric}'
            )

            if batch_metric > best_metric:
                best_metric = batch_metric
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    logger.info(f'Early stopping at epoch {epoch + 1}')
                    break

    def train_model(
        self,
        max_factors=100,
        init_sample_size=4096,
        max_sample_size=20000,
        batch_size_p=128,
        num_epochs_p=1000,
        learning_rate_p=1e-4,
        early_stopping_p=20,
        batch_size_g=128,
        num_epochs_g=200,
        learning_rate_g=1e-3,
        early_stopping_g=10,
        ic_threshold=0.05,
        corr_threshold=0.7,
        result_dir='./tmp/'
        ):

        sampled_states, sampled_metrics = [], []

        for i in tqdm(range(init_sample_size), desc='Sampling initial factors'):
            state, metric = self._sample_factor()
            sampled_states.append(state)
            sampled_metrics.append(metric)

        while len(self.alpha_pool) < max_factors:
            self._reset_net()
            self._train_netp(sampled_states, sampled_metrics, batch_size_p, num_epochs_p, learning_rate_p, early_stopping_p)
            self._train_netg(sampled_states, sampled_metrics, batch_size_g, num_epochs_g, learning_rate_g, early_stopping_g, ic_threshold, corr_threshold)
            logger.info(f'Current number of alphas in pool: {len(self.alpha_pool)}')
            joblib.dump(self.alpha_pool, result_dir + 'aff_alpha_pool.pkl')

            if len(sampled_states) > max_sample_size:
                sample_idx = np.random.permutation(len(sampled_states))[:max_sample_size]
                sampled_states = [sampled_states[i] for i in sample_idx]
                sampled_metrics = [sampled_metrics[i] for i in sample_idx]

        return self.alpha_pool

    def calculate_expression(self, expression, feature_data=None, rank=False) -> torch.Tensor:
        """Calculate expression result from string representation
        
        Supports both new mathematical notation (a+b, a*b, ts_mean(x,5)) and legacy formats
        """
        if feature_data is None:
            feature_data = self.feature_data

        # Import necessary classes
        from .operators import (Abs, Log, Neg, Inv, Rank, Add, Sub, Mul, Div, Max, Min,
                               TsMean, TsStd, TsMax, TsMin, PctChange, Lag, TsCorr, Constant)
        
        # Get a sample tensor shape for creating constant tensors
        sample_tensor = next(iter(feature_data.values()))
        
        # Helper to convert scalar to tensor if needed
        def to_tensor(x):
            if isinstance(x, (int, float)):
                return torch.full_like(sample_tensor, float(x))
            return x
        
        # Build local environment for eval
        local_env = {}
        
        # Add feature data
        for feature_name in self.feature_names:
            local_env[feature_name] = feature_data[feature_name]
        
        # Add operator classes with direct computation and scalar conversion
        local_env['abs'] = lambda x: Abs(None)._compute(to_tensor(x))
        local_env['log'] = lambda x: Log(None)._compute(to_tensor(x))
        local_env['neg'] = lambda x: Neg(None)._compute(to_tensor(x))
        local_env['inv'] = lambda x: Inv(None)._compute(to_tensor(x))
        local_env['rank'] = lambda x: Rank(None)._compute(to_tensor(x))
        local_env['max'] = lambda x, y: Max(None, None)._compute(to_tensor(x), to_tensor(y))
        local_env['min'] = lambda x, y: Min(None, None)._compute(to_tensor(x), to_tensor(y))
        
        # Add rolling operators with window size parameter and scalar conversion
        local_env['ts_mean'] = lambda x, w: TsMean(None, w)._compute(to_tensor(x))
        local_env['ts_std'] = lambda x, w: TsStd(None, w)._compute(to_tensor(x))
        local_env['ts_max'] = lambda x, w: TsMax(None, w)._compute(to_tensor(x))
        local_env['ts_min'] = lambda x, w: TsMin(None, w)._compute(to_tensor(x))
        local_env['pctchange'] = lambda x, w: PctChange(None, w)._compute(to_tensor(x))
        local_env['lag'] = lambda x, w: Lag(None, w)._compute(to_tensor(x))
        local_env['ts_corr'] = lambda x, y, w: TsCorr(None, None, w)._compute(to_tensor(x), to_tensor(y))
        
        # Evaluate the expression
        result = eval(expression, {"__builtins__": {}}, local_env)
        return ops_rank(result) if rank else result