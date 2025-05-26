# rl_agent/environment.py: Defines the trading environment for the RL agent.
 
import numpy as np

class TSTEnv:
    def __init__(self, rl_states, prices, initial_cash=10000.0, trading_fee=0.001):
        self.rl_states = rl_states  # shape: (T, 256)
        self.prices = prices        # shape: (T,)
        self.initial_cash = initial_cash
        self.trading_fee = trading_fee
        self.reset()

    def reset(self):
        self.index = 0
        self.cash = self.initial_cash
        self.shares = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        return {
            "rl_state": self.rl_states[self.index],  # np.array shape: (256,)
            "tst_rl_state": self.rl_states[self.index],  # For compatibility with agent interface
            "cash": self.cash,
            "shares": self.shares,
            "price": self.prices[self.index],
            "current_price": self.prices[self.index],  # For compatibility
        }

    def step(self, action):
        # action: 0 = HOLD, 1 = BUY, 2 = SELL
        price = self.prices[self.index]
        
        # Prevent extreme price values that cause overflow
        price = np.clip(price, 0.01, 10000.0)  # Reasonable price range
        
        # Calculate portfolio value safely
        try:
            prev_portfolio_value = self.cash + self.shares * price
            # Prevent overflow in portfolio calculation
            if prev_portfolio_value > 1e10:  # 10 billion limit
                prev_portfolio_value = 1e10
        except (OverflowError, ValueError):
            prev_portfolio_value = self.initial_cash

        if action == 1:  # BUY
            try:
                cost_per_share = price * (1 + self.trading_fee)
                if cost_per_share > 0 and self.cash > cost_per_share:
                    max_shares = int(self.cash // cost_per_share)
                    if max_shares > 0:
                        # Limit maximum shares to prevent overflow
                        max_shares = min(max_shares, 1000000)  # 1M shares limit
                        cost = max_shares * cost_per_share
                        self.cash = max(0, self.cash - cost)
                        self.shares += max_shares
            except (OverflowError, ValueError, ZeroDivisionError):
                pass  # Skip action if calculation fails

        elif action == 2:  # SELL
            if self.shares > 0:
                try:
                    revenue = self.shares * price * (1 - self.trading_fee)
                    # Prevent overflow in revenue calculation
                    if revenue < 1e10:  # 10 billion limit
                        self.cash += revenue
                        self.shares = 0
                    else:
                        self.cash = 1e10
                        self.shares = 0
                except (OverflowError, ValueError):
                    self.shares = 0  # Emergency sell

        self.index += 1
        if self.index >= len(self.prices):
            self.index = len(self.prices) - 1
            self.done = True

        # Calculate new portfolio value safely
        try:
            new_price = np.clip(self.prices[self.index], 0.01, 10000.0)
            new_portfolio_value = self.cash + self.shares * new_price
            # Prevent overflow
            if new_portfolio_value > 1e10:
                new_portfolio_value = 1e10
        except (OverflowError, ValueError):
            new_portfolio_value = prev_portfolio_value
        
        # Calculate normalized reward (percentage change)
        try:
            if prev_portfolio_value > 0:
                reward = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value
            else:
                reward = 0.0
        except (OverflowError, ValueError, ZeroDivisionError):
            reward = 0.0
        
        # Clip reward to prevent extreme values
        reward = np.clip(reward, -0.1, 0.1)  # Limit to ±10% per step

        return self._get_state(), reward, self.done, {}

    def get_portfolio_value(self):
        price = self.prices[self.index]
        return self.cash + self.shares * price
