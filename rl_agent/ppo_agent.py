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

        # 총 loss: 정책 + 가치 (value loss 가중치 감소)
        total_loss = policy_loss + 0.1 * value_loss

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
            'mean_return': returns.mean().item() if len(returns) > 1 else 0.0
        }

    def predict_action(self, state_vector):
        action, _ = self.select_action(state_vector)  # 전체 state_vector 전달
        return {
            "action": ["BUY", "SELL", "HOLD"][action],
            "reason": "PPO policy output",
            "target_price": state_vector.get("current_price", 0)
        }

