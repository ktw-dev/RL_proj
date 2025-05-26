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
    def __init__(self, state_dim=259, action_dim=3, lr=3e-4, gamma=0.99, tau=0.005):
        # TST features (256) + portfolio info (3: cash_ratio, stock_ratio, portfolio_value_normalized)
        self.tst_dim = 256
        self.portfolio_dim = 3
        self.total_state_dim = state_dim
        
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

        return {
            'total_loss': (q1_loss + q2_loss + value_loss + policy_loss).item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item()
        }

    def update_policy(self, transitions):
        """
        Update policy using the existing update method and return loss for consistency with PPO
        """
        # Call the existing update method
        loss_info = self._update(transitions)
        return loss_info

    def predict_action(self, state_vector):
        action, _ = self.select_action(state_vector)  # 전체 state_vector 전달
        return {
            "action": ["BUY", "SELL", "HOLD"][action],
            "reason": "SAC policy output",
            "target_price": state_vector.get("current_price", 0)
        }