# rl_agent/ppo_agent.py: Implementation of the Proximal Policy Optimization (PPO) algorithm. 

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPOAgent:
    def __init__(self, state_dim=259, action_dim=3, lr=3e-5, gamma=0.99, eps_clip=0.2):
        # TST features (256) + portfolio info (3: cash_ratio, stock_ratio, portfolio_value_normalized)
        self.tst_dim = 256
        self.portfolio_dim = 3
        self.total_state_dim = state_dim
        
        # Learning rate scheduling parameters
        self.initial_lr = lr
        self.current_lr = lr
        self.min_lr = lr * 0.01  # 최소 학습률 (더 낮게)
        self.max_lr = lr * 100    # 최대 학습률 (더 높게)
        self.lr_decay_factor = 0.8   # 20% 감소 (더 적극적)
        self.lr_increase_factor = 1.5  # 50% 증가 (더 적극적)
        
        # Loss history for adaptive scheduling
        self.policy_loss_history = []
        self.value_loss_history = []
        self.loss_ratio_history = []
        
        # Dynamic loss weighting
        self.initial_value_weight = 0.5
        self.current_value_weight = 0.5
        self.min_value_weight = 0.1
        self.max_value_weight = 2.0
        
        # 정책 네트워크 - 포트폴리오 정보 포함
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

        # 이전 정책 네트워크 (PPO의 ratio 계산용)
        self.old_policy = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.old_policy.load_state_dict(self.policy.state_dict())

        # 가치 함수 네트워크 추가
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.optimizer = optim.AdamW(list(self.policy.parameters()) + list(self.value_net.parameters()), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip

    def _prepare_state(self, state_input):
        """
        Prepare state for neural network input.
        Handles both training (TST only) and inference (TST + portfolio) scenarios.
        """
        # Determine device from model parameters
        device = next(self.policy.parameters()).device
        
        if isinstance(state_input, dict):
            # Inference mode: extract from state dictionary
            tst_state = state_input["tst_rl_state"]  # (256,)
            
            # Extract portfolio information
            cash = state_input.get("cash", 10000.0)
            shares = state_input.get("shares", 0.0)
            price = state_input.get("current_price", state_input.get("price", 100.0))
            
            # Calculate portfolio metrics
            portfolio_value = cash + shares * price
            cash_ratio = cash / portfolio_value if portfolio_value > 0 else 1.0
            stock_ratio = (shares * price) / portfolio_value if portfolio_value > 0 else 0.0
            portfolio_value_normalized = np.log(portfolio_value / 10000.0)  # Log-normalized relative to initial cash
            
            # Combine TST features with portfolio info
            portfolio_info = np.array([cash_ratio, stock_ratio, portfolio_value_normalized])
            full_state = np.concatenate([tst_state, portfolio_info])
            
        elif isinstance(state_input, np.ndarray):
            if state_input.shape[0] == self.tst_dim:
                # Training mode: TST features only, add dummy portfolio info
                dummy_portfolio = np.array([1.0, 0.0, 0.0])  # 100% cash, 0% stock, baseline portfolio
                full_state = np.concatenate([state_input, dummy_portfolio])
            else:
                # Already includes portfolio info
                full_state = state_input
        else:
            raise ValueError(f"Unsupported state input type: {type(state_input)}")
        
        return torch.FloatTensor(full_state).to(device)

    def select_action(self, state):
        state_tensor = self._prepare_state(state)
        probs = self.old_policy(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate_value(self, state):
        state_tensor = self._prepare_state(state)
        value = self.value_net(state_tensor)
        return value.squeeze().item()

    def update_policy(self, transitions):
        states = transitions['states']  # Already 259-dim from train.py
        actions = transitions['actions']
        returns = transitions['returns']
        advantages = transitions['advantages']
        old_log_probs = transitions['log_probs']

        # Normalize advantages to prevent gradient explosion
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # Additional clipping for extreme advantages
            advantages = torch.clamp(advantages, -5.0, 5.0)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            # Additional clipping for extreme returns
            returns = torch.clamp(returns, -5.0, 5.0)

        # 정책 네트워크 forward - states are already 259-dim
        new_probs = self.policy(states)
        new_dist = torch.distributions.Categorical(new_probs)
        new_log_probs = new_dist.log_prob(actions)

        # ratio 계산
        ratio = (new_log_probs - old_log_probs).exp()
        
        # Clamp ratio to prevent extreme values
        ratio = torch.clamp(ratio, 0.1, 10.0)

        # surrogate loss 계산 (clip 적용)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        # value loss (mean squared error) - states are already 259-dim
        value_estimates = self.value_net(states).squeeze()
        value_loss = nn.functional.mse_loss(value_estimates, returns)

        # Adaptive learning rate and loss weight adjustment
        lr_changed, lr_reason, loss_ratio = self._adjust_learning_rate(policy_loss.item(), value_loss.item())
        current_weight = self._adjust_loss_weights(policy_loss.item(), value_loss.item())

        # 총 loss: 정책 + 가치 (동적 가중치 적용)
        total_loss = policy_loss + current_weight * value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.value_net.parameters()), 
            max_norm=0.5
        )
        
        self.optimizer.step()

        # old policy 업데이트
        self.old_policy.load_state_dict(self.policy.state_dict())

        # Return detailed loss information for monitoring
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'mean_advantage': advantages.mean().item() if len(advantages) > 1 else 0.0,
            'mean_return': returns.mean().item() if len(returns) > 1 else 0.0,
            'current_lr': self.current_lr,
            'value_weight': current_weight,
            'loss_ratio': loss_ratio,
            'lr_changed': lr_changed,
            'lr_reason': lr_reason
        }

    def predict_action(self, state_vector):
        action, _ = self.select_action(state_vector)  # 전체 state_vector 전달
        return {
            "action": ["BUY", "SELL", "HOLD"][action],
            "reason": "PPO policy output",
            "target_price": state_vector.get("current_price", 0)
        }

    def _adjust_learning_rate(self, policy_loss, value_loss):
        """
        Adaptively adjust learning rate based on loss patterns.
        """
        # Calculate loss ratio (value_loss / policy_loss)
        if policy_loss > 1e-8:  # Avoid division by zero
            loss_ratio = value_loss / policy_loss
        else:
            loss_ratio = 1000.0  # Very high ratio if policy loss is very small
        
        self.loss_ratio_history.append(loss_ratio)
        self.policy_loss_history.append(policy_loss)
        self.value_loss_history.append(value_loss)
        
        # Keep only recent history (last 5 updates for faster response)
        if len(self.loss_ratio_history) > 5:
            self.loss_ratio_history.pop(0)
            self.policy_loss_history.pop(0)
            self.value_loss_history.pop(0)
        
        # Adaptive learning rate adjustment (start after just 2 updates)
        if len(self.loss_ratio_history) >= 2:
            recent_ratio = self.loss_ratio_history[-1]  # Use latest ratio for faster response
            
            # More realistic thresholds based on actual training patterns
            if recent_ratio > 100:  # Value loss >> Policy loss (severe imbalance)
                new_lr = min(self.current_lr * self.lr_increase_factor, self.max_lr)
                adjustment_reason = f"Value loss too high (ratio: {recent_ratio:.1f})"
            elif recent_ratio > 20:  # Moderate imbalance
                new_lr = min(self.current_lr * 1.2, self.max_lr)
                adjustment_reason = f"Value loss high (ratio: {recent_ratio:.1f})"
            elif recent_ratio < 0.1:  # Policy loss >> Value loss
                new_lr = max(self.current_lr * self.lr_decay_factor, self.min_lr)
                adjustment_reason = f"Policy converging too fast (ratio: {recent_ratio:.1f})"
            elif recent_ratio < 1:  # Policy slightly higher
                new_lr = max(self.current_lr * 0.9, self.min_lr)
                adjustment_reason = f"Policy ahead (ratio: {recent_ratio:.1f})"
            elif 5 <= recent_ratio <= 20:  # Good balance
                new_lr = min(self.current_lr * 1.05, self.max_lr)
                adjustment_reason = f"Balanced learning (ratio: {recent_ratio:.1f})"
            else:
                new_lr = self.current_lr
                adjustment_reason = f"No change (ratio: {recent_ratio:.1f})"
            
            # Update learning rate with lower threshold (2% instead of 5%)
            if abs(new_lr - self.current_lr) / self.current_lr > 0.02:  # 2% threshold
                old_lr = self.current_lr
                self.current_lr = new_lr
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.current_lr
                print(f"    📈 LR: {old_lr:.2e} → {self.current_lr:.2e} ({adjustment_reason})")
                return True, adjustment_reason, recent_ratio
        
        return False, "Insufficient history", loss_ratio if len(self.loss_ratio_history) > 0 else 0.0

    def _adjust_loss_weights(self, policy_loss, value_loss):
        """
        Dynamically adjust the weight of value loss in total loss.
        """
        if policy_loss > 1e-8:
            loss_ratio = value_loss / policy_loss
            
            # If value loss is much higher, increase its weight
            if loss_ratio > 100:
                self.current_value_weight = min(self.current_value_weight * 1.1, self.max_value_weight)
            # If policy loss is higher, decrease value weight
            elif loss_ratio < 0.1:
                self.current_value_weight = max(self.current_value_weight * 0.9, self.min_value_weight)
            # Gradual adjustment towards balanced learning
            else:
                target_weight = max(0.5, min(2.0, loss_ratio / 10))
                self.current_value_weight = 0.9 * self.current_value_weight + 0.1 * target_weight
                self.current_value_weight = max(self.min_value_weight, min(self.max_value_weight, self.current_value_weight))
        
        return self.current_value_weight

