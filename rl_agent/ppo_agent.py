# rl_agent/ppo_agent.py: Implementation of the Proximal Policy Optimization (PPO) algorithm.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PPOAgent:
    def __init__(self, state_dim=260, action_dim=3, lr=3e-4, gamma=0.99, 
                 k_epochs=4, eps_clip=0.2, entropy_coeff=0.01, value_loss_coeff=0.5, max_grad_norm=0.5):
        
        self.tst_dim = 256  # Assuming TST features are 256
        # cash_ratio, stock_ratio, portfolio_value_normalized, has_shares_flag
        self.portfolio_dim = 4 
        self.total_state_dim = state_dim # Should be tst_dim + portfolio_dim

        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm

        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        # Old Actor Network (for PPO ratio calculation)
        self.actor_old = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        self.actor_old.load_state_dict(self.actor.state_dict())

        # Optimizer for both actor and critic parameters
        self.optimizer = optim.AdamW(
            list(self.actor.parameters()) + list(self.critic.parameters()), 
            lr=lr
        )
        
        self.policy_loss_history = []
        self.value_loss_history = []


    def _prepare_state(self, state_input):
        """
        Prepare state for neural network input.
        Handles both training (TST only) and inference (TST + portfolio) scenarios.
        """
        device = next(self.actor.parameters()).device
        
        if isinstance(state_input, dict):
            tst_state = state_input["tst_rl_state"]
            cash = state_input.get("cash", 10000.0)
            shares = state_input.get("shares", 0.0)
            price = state_input.get("current_price", state_input.get("price", 100.0))
            has_shares_flag = 1.0 if shares > 0 else 0.0 # Get has_shares_flag from shares
            
            portfolio_value = cash + shares * price
            cash_ratio = cash / portfolio_value if portfolio_value > 0 else 1.0
            stock_ratio = (shares * price) / portfolio_value if portfolio_value > 0 else 0.0
            # Normalize portfolio value (e.g., log-normalized relative to initial cash)
            portfolio_value_normalized = np.log(portfolio_value / 10000.0 + 1e-9) # Added small epsilon
            
            portfolio_info = np.array([cash_ratio, stock_ratio, portfolio_value_normalized, has_shares_flag])
            full_state = np.concatenate([tst_state, portfolio_info])
            
        elif isinstance(state_input, np.ndarray):
            if state_input.shape[0] == self.tst_dim:
                # Training: TST features only, add dummy portfolio info (neutral/initial state)
                dummy_portfolio = np.array([1.0, 0.0, 0.0, 0.0]) # cash_ratio, stock_ratio, portfolio_val_norm, has_shares_flag(0 for no shares)
                full_state = np.concatenate([state_input, dummy_portfolio])
            elif state_input.shape[0] == self.total_state_dim:
                full_state = state_input
            else:
                raise ValueError(f"Unexpected state shape: {state_input.shape}, expected {self.tst_dim} or {self.total_state_dim}")
        else:
            raise ValueError(f"Unsupported state input type: {type(state_input)}")
        
        return torch.FloatTensor(full_state).to(device)

    def select_action(self, state):
        """
        Selects an action using the old actor network (for collecting trajectories).
        """
        with torch.no_grad():
            state_tensor = self._prepare_state(state)
            action_probs = self.actor_old(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        return action.item(), action_logprob

    def get_value(self, state):
        """
        Gets the state value from the critic network.
        """
        with torch.no_grad():
            state_tensor = self._prepare_state(state)
            value = self.critic(state_tensor)
        return value.squeeze().item()

    def update_policy(self, transitions):
        """
        Update policy for K epochs on the given batch of transitions.
        """
        # Device from model parameters
        device = next(self.actor.parameters()).device

        # Extract transitions and move to device if not already
        # train.py already sends tensors on the correct device
        states = transitions['states'].to(device)
        actions = transitions['actions'].to(device)
        old_log_probs = transitions['log_probs'].to(device).detach() # Detach, as these are fixed targets
        returns = transitions['returns'].to(device) # Targets for value function
        advantages = transitions['advantages'].to(device)

        # Normalize advantages
        if len(advantages) > 1:
             advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else: # Handle single element case to avoid NaN std
            advantages = torch.zeros_like(advantages)


        for _ in range(self.k_epochs):
            # Evaluate old actions and values :
            # নতুন actor নেটওয়ার্ক থেকে লগ প্রবাবিলিটি, স্টেট ভ্যালু এবং এনট্রপি পান
            action_probs = self.actor(states)
            dist = torch.distributions.Categorical(action_probs)
            
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            state_values = self.critic(states).squeeze()

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(new_log_probs - old_log_probs)

            # Finding Surrogate Loss:
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(state_values, returns)
            
            loss = policy_loss + self.value_loss_coeff * value_loss - self.entropy_coeff * entropy

            # Take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()), 
                self.max_grad_norm
            )
            self.optimizer.step()
            
        # Copy new weights into old policy:
        self.actor_old.load_state_dict(self.actor.state_dict())

        # Store loss history (optional, can be removed if not used for adaptive logic)
        self.policy_loss_history.append(policy_loss.item())
        self.value_loss_history.append(value_loss.item())

        return {
            'total_loss': loss.item(), # loss from the last PPO epoch
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'current_lr': self.optimizer.param_groups[0]['lr'] # No adaptive LR for now
        }

    def predict_action(self, state_vector):
        """
        Predicts an action for inference using the current actor network.
        Also calculates a dynamic target_price based on the action and TST predictions.
        """
        with torch.no_grad():
            state_tensor = self._prepare_state(state_vector)
            # For prediction, use the main actor, not actor_old
            action_probs = self.actor(state_tensor) 
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            action_idx = action.item() # 0:HOLD, 1:BUY, 2:SELL

        current_price = state_vector.get("current_price", 0)
        # Default to current_price if TST prediction is not available
        predicted_future_close = state_vector.get("predicted_tst_next_day_close", current_price) 

        BUY_OFFSET_PERCENT = 0.0005  # 0.05%
        SELL_OFFSET_PERCENT = 0.0005 # 0.05%

        if action_idx == 1: # BUY
            target_price = current_price * (1 + BUY_OFFSET_PERCENT)
        elif action_idx == 2: # SELL
            target_price = predicted_future_close * (1 - SELL_OFFSET_PERCENT)
        else: # HOLD or any other case
            target_price = current_price
        
        action_map = ["HOLD", "BUY", "SELL"]
        
        return {
            "action": action_map[action_idx],
            "reason": f"PPO policy output (action_probs: {action_probs.cpu().numpy()})", # Added probs for more info
            "target_price": target_price 
        }

    # Removed _adjust_learning_rate and _adjust_loss_weights for a standard PPO
