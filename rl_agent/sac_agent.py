# rl_agent/sac_agent.py: Implementation of the Soft Actor-Critic (SAC) algorithm.
 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.net(state)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

class SACAgent:
    def __init__(self, state_dim=260, action_dim=3, lr=3e-4, gamma=0.99, tau=0.005):
        # TST features (256) + portfolio info (4: cash_ratio, stock_ratio, portfolio_value_normalized, has_shares_flag)
        self.tst_dim = 256
        self.portfolio_dim = 4
        self.total_state_dim = state_dim
        
        # Learning rate scheduling parameters
        self.initial_lr = lr
        self.current_lr = lr
        self.min_lr = lr * 0.01  # 최소 학습률 (더 낮게)
        self.max_lr = lr * 30    # 최대 학습률 (더 높게, SAC는 PPO보다 보수적)
        self.lr_decay_factor = 0.8   # 20% 감소 (더 적극적)
        self.lr_increase_factor = 1.3  # 30% 증가 (더 적극적)
        
        # Loss history for adaptive scheduling
        self.policy_loss_history = []
        self.value_loss_history = []
        self.q_loss_history = []
        self.loss_ratio_history = []
        
        # Dynamic loss weighting for SAC
        self.initial_value_weight = 1.0
        self.current_value_weight = 1.0
        self.min_value_weight = 0.5
        self.max_value_weight = 3.0
        
        # Policy Network (Actor)
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

        # Q-Networks (Critics)
        self.q_net1 = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        self.q_net2 = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        # Value Network
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Target Value Network
        self.target_value_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.target_value_net.load_state_dict(self.value_net.state_dict())

        # Optimizers
        self.policy_optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr)
        self.q1_optimizer = optim.AdamW(self.q_net1.parameters(), lr=lr)
        self.q2_optimizer = optim.AdamW(self.q_net2.parameters(), lr=lr)
        self.value_optimizer = optim.AdamW(self.value_net.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau

    def _prepare_state(self, state_input):
        """
        Prepare state for neural network input.
        Handles both training (TST only) and inference (TST + portfolio) scenarios.
        """
        # Determine device from model parameters
        device = next(self.policy_net.parameters()).device
        
        if isinstance(state_input, dict):
            # Inference mode: extract from state dictionary
            tst_state = state_input["tst_rl_state"]  # (256,)
            
            # Extract portfolio information
            cash = state_input.get("cash", 10000.0)
            shares = state_input.get("shares", 0.0)
            price = state_input.get("current_price", state_input.get("price", 100.0))
            has_shares_flag = 1.0 if shares > 0 else 0.0
            
            # Calculate portfolio metrics
            portfolio_value = cash + shares * price
            cash_ratio = cash / portfolio_value if portfolio_value > 0 else 1.0
            stock_ratio = (shares * price) / portfolio_value if portfolio_value > 0 else 0.0
            portfolio_value_normalized = np.log(portfolio_value / 10000.0 + 1e-9)
            
            # Combine TST features with portfolio info
            portfolio_info = np.array([cash_ratio, stock_ratio, portfolio_value_normalized, has_shares_flag])
            full_state = np.concatenate([tst_state, portfolio_info])
            
        elif isinstance(state_input, np.ndarray):
            if state_input.shape[0] == self.tst_dim:
                # Training mode: TST features only, add dummy portfolio info
                dummy_portfolio = np.array([1.0, 0.0, 0.0, 0.0])
                full_state = np.concatenate([state_input, dummy_portfolio])
            else:
                # Already includes portfolio info
                full_state = state_input
        else:
            raise ValueError(f"Unsupported state input type: {type(state_input)}")
        
        return torch.FloatTensor(full_state).to(device)

    def select_action(self, state):
        state_tensor = self._prepare_state(state)
        probs = self.policy_net(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate_value(self, state):
        state_tensor = self._prepare_state(state)
        value = self.value_net(state_tensor)
        return value.squeeze().item()

    def _update(self, transitions):
        states = transitions['states']  # Already 259-dim from train.py
        actions = transitions['actions'].unsqueeze(1)
        rewards = transitions['returns'].unsqueeze(1)
        next_states = transitions['next_states']  # Already 259-dim from train.py
        dones = transitions['dones'].unsqueeze(1)
        
        # SAC temperature parameter
        alpha = 0.2

        # Current Q-values
        q1_values = self.q_net1(states).gather(1, actions)
        q2_values = self.q_net2(states).gather(1, actions)

        with torch.no_grad():
            next_action_probs = self.policy_net(next_states)
            next_dist = torch.distributions.Categorical(next_action_probs)
            next_actions = next_dist.sample()
            next_log_probs = next_dist.log_prob(next_actions).unsqueeze(1)
            
            target_values = self.target_value_net(next_states)
            q_target = rewards + self.gamma * (1 - dones) * (target_values - alpha * next_log_probs)

        # Q-network losses
        q1_loss = F.mse_loss(q1_values, q_target)
        q2_loss = F.mse_loss(q2_values, q_target)

        # Update Q-networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update value network
        with torch.no_grad():
            action_probs = self.policy_net(states)
            dist = torch.distributions.Categorical(action_probs)
            actions_sample = dist.sample()
            log_probs = dist.log_prob(actions_sample).unsqueeze(1)
            
            q1_values_new = self.q_net1(states).gather(1, actions_sample.unsqueeze(1))
            q2_values_new = self.q_net2(states).gather(1, actions_sample.unsqueeze(1))
            q_min = torch.min(q1_values_new, q2_values_new)
            
            value_target = q_min - alpha * log_probs

        current_values = self.value_net(states)
        value_loss = F.mse_loss(current_values, value_target)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update policy network
        action_probs = self.policy_net(states)
        dist = torch.distributions.Categorical(action_probs)
        actions_sample = dist.sample()
        log_probs = dist.log_prob(actions_sample).unsqueeze(1)
        
        q1_values_policy = self.q_net1(states).gather(1, actions_sample.unsqueeze(1))
        q2_values_policy = self.q_net2(states).gather(1, actions_sample.unsqueeze(1))
        q_min_policy = torch.min(q1_values_policy, q2_values_policy)
        
        policy_loss = (alpha * log_probs - q_min_policy).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft update target network
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Calculate average Q loss for adaptive scheduling
        q_loss_avg = (q1_loss.item() + q2_loss.item()) / 2
        
        # Adaptive learning rate and loss weight adjustment
        lr_changed, lr_reason, loss_ratio = self._adjust_learning_rate(
            policy_loss.item(), value_loss.item(), q_loss_avg
        )
        current_weight = self._adjust_loss_weights(
            policy_loss.item(), value_loss.item(), q_loss_avg
        )

        return {
            'total_loss': (q1_loss + q2_loss + value_loss + policy_loss).item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'q_loss_avg': q_loss_avg,
            'current_lr': self.current_lr,
            'value_weight': current_weight,
            'loss_ratio': loss_ratio,
            'lr_changed': lr_changed,
            'lr_reason': lr_reason
        }

    def update_policy(self, transitions):
        """
        Update policy using the existing update method and return loss for consistency with PPO
        """
        # Call the existing update method
        loss_info = self._update(transitions)
        return loss_info

    def predict_action(self, state_vector):
        with torch.no_grad():
            state_tensor = self._prepare_state(state_vector)
            
            # Get action probabilities from the policy network
            action_probs = self.policy_net(state_tensor) # Output shape: (action_dim,)

            # For deterministic inference, choose the action with the highest probability
            action_idx = torch.argmax(action_probs, dim=-1).item() # Returns index 0, 1, or 2

        # Consistent mapping with PPO and TSTEnv: 0:HOLD, 1:BUY, 2:SELL
        action_map = ["HOLD", "BUY", "SELL"]
        
        return {
            "action": action_map[action_idx],
            "reason": f"SAC policy output {action_probs.cpu().numpy()}",
            "target_price": state_vector.get("current_price", 0)
        }

    def _adjust_learning_rate(self, policy_loss, value_loss, q_loss_avg):
        """
        Adaptively adjust learning rate based on loss patterns for SAC.
        """
        # Calculate combined loss ratio (use absolute value for policy loss since SAC can be negative)
        total_critic_loss = value_loss + q_loss_avg
        abs_policy_loss = abs(policy_loss)  # SAC policy loss can be negative
        
        if abs_policy_loss > 1e-8:
            loss_ratio = total_critic_loss / abs_policy_loss
        else:
            loss_ratio = 1000.0
        
        self.loss_ratio_history.append(loss_ratio)
        self.policy_loss_history.append(policy_loss)
        self.value_loss_history.append(value_loss)
        self.q_loss_history.append(q_loss_avg)
        
        # Keep only recent history (last 5 updates for faster response)
        if len(self.loss_ratio_history) > 5:
            self.loss_ratio_history.pop(0)
            self.policy_loss_history.pop(0)
            self.value_loss_history.pop(0)
            self.q_loss_history.pop(0)
        
        # Adaptive learning rate adjustment (start after just 2 updates)
        if len(self.loss_ratio_history) >= 2:
            recent_ratio = self.loss_ratio_history[-1]  # Use latest ratio for faster response
            
            # SAC-specific adjustment logic with more realistic thresholds
            if recent_ratio > 50:  # Critic losses >> Policy loss (severe imbalance)
                new_lr = min(self.current_lr * self.lr_increase_factor, self.max_lr)
                adjustment_reason = f"Critic losses too high (ratio: {recent_ratio:.1f})"
            elif recent_ratio > 10:  # Moderate imbalance
                new_lr = min(self.current_lr * 1.15, self.max_lr)
                adjustment_reason = f"Critic losses high (ratio: {recent_ratio:.1f})"
            elif recent_ratio < 0.2:  # Policy loss >> Critic losses
                new_lr = max(self.current_lr * self.lr_decay_factor, self.min_lr)
                adjustment_reason = f"Policy converging too fast (ratio: {recent_ratio:.1f})"
            elif recent_ratio < 1:  # Policy slightly higher
                new_lr = max(self.current_lr * 0.9, self.min_lr)
                adjustment_reason = f"Policy ahead (ratio: {recent_ratio:.1f})"
            elif 3 <= recent_ratio <= 10:  # Good balance for SAC
                new_lr = min(self.current_lr * 1.03, self.max_lr)  # Very gradual increase
                adjustment_reason = f"Balanced learning (ratio: {recent_ratio:.1f})"
            else:
                new_lr = self.current_lr
                adjustment_reason = f"No change (ratio: {recent_ratio:.1f})"
            
            # Update learning rate with lower threshold (2% instead of 3%)
            if abs(new_lr - self.current_lr) / self.current_lr > 0.02:  # 2% threshold
                old_lr = self.current_lr
                self.current_lr = new_lr
                # Update all optimizers
                for optimizer in [self.policy_optimizer, self.q1_optimizer, self.q2_optimizer, self.value_optimizer]:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = self.current_lr
                print(f"    📈 LR: {old_lr:.2e} → {self.current_lr:.2e} ({adjustment_reason})")
                return True, adjustment_reason, recent_ratio
        
        return False, "Insufficient history", loss_ratio if len(self.loss_ratio_history) > 0 else 0.0

    def _adjust_loss_weights(self, policy_loss, value_loss, q_loss_avg):
        """
        Dynamically adjust the weight of value loss in SAC (less aggressive than PPO).
        """
        total_critic_loss = value_loss + q_loss_avg
        abs_policy_loss = abs(policy_loss)  # SAC policy loss can be negative
        
        if abs_policy_loss > 1e-8:
            loss_ratio = total_critic_loss / abs_policy_loss
            
            # SAC uses more conservative weight adjustment
            if loss_ratio > 50:
                self.current_value_weight = min(self.current_value_weight * 1.05, self.max_value_weight)
            elif loss_ratio < 0.2:
                self.current_value_weight = max(self.current_value_weight * 0.95, self.min_value_weight)
            else:
                # Gradual adjustment towards target
                target_weight = max(0.8, min(2.0, loss_ratio / 15))
                self.current_value_weight = 0.95 * self.current_value_weight + 0.05 * target_weight
                self.current_value_weight = max(self.min_value_weight, min(self.max_value_weight, self.current_value_weight))
        
        return self.current_value_weight