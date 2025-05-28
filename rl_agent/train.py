# rl_agent/train.py: Script for training the RL agent. 

import torch
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to sys.path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# Import with relative paths to avoid module issues
try:
    from rl_agent.environment import TSTEnv
    from rl_agent.ppo_agent import PPOAgent
    from rl_agent.sac_agent import SACAgent
except ImportError:
    # Fallback to direct imports if running from rl_agent directory
    from environment import TSTEnv
    from ppo_agent import PPOAgent
    from sac_agent import SACAgent

# Import TST prediction functionality
try:
    from tst_model.predict import (
        load_latest_model, 
        prepare_data_for_prediction, 
        create_prediction_sequences,
        DEFAULT_MODEL_CONFIG
    )
    from tst_model.model import TSTModel
    TST_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TST model imports failed: {e}")
    print("TST integration will not be available.")
    TST_AVAILABLE = False

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages

def check_gpu_availability():
    """Check and display GPU information."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        
        print(f"🚀 GPU Available: {gpu_count} device(s)")
        print(f"   Current GPU: {gpu_name}")
        print(f"   GPU Memory: {gpu_memory:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("⚠️  GPU not available. Training will use CPU.")
        return False

def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def generate_rl_states_from_tst(
    ticker: str = None,
    model_dir: str = None,
    data_path: str = None,
    context_length: int = 60,
    device: torch.device = None
):
    """
    Generate RL states using trained TST model for agent training.
    
    Args:
        ticker (str): Target ticker (if None, process all available)
        model_dir (str): Directory containing trained TST models
        data_path (str): Path to historical data CSV
        context_length (int): Context length for TST model
        device (torch.device): Device for computation
        
    Returns:
        dict: Contains RL states, prices, and metadata for each ticker
    """
    if not TST_AVAILABLE:
        raise ImportError("TST model components not available. Cannot generate RL states.")
    
    # Set default paths
    if model_dir is None:
        model_dir = os.path.join(PROJECT_ROOT, 'tst_model_output')
    if data_path is None:
        data_path = os.path.join(PROJECT_ROOT, 'all_tickers_historical_features.csv')
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"=== Generating RL States from TST Model ===")
    print(f"Target ticker: {ticker or 'All tickers'}")
    print(f"Model directory: {model_dir}")
    print(f"Data path: {data_path}")
    print(f"Device: {device}")
    
    # Prepare data
    data_info = prepare_data_for_prediction(
        data_path, 
        target_ticker=ticker,
        context_length=context_length
    )
    
    # Update model config with actual input size
    model_config = DEFAULT_MODEL_CONFIG.copy()
    model_config['input_size'] = len(data_info['feature_columns'])
    model_config['context_length'] = context_length
    
    # Load trained TST model
    model, model_path = load_latest_model(model_dir, model_config, device)
    print(f"Loaded TST model: {model_path}")
    
    # Create sequences for all available data (not just latest)
    rl_states_data = {}
    
    for ticker_name, group in data_info['scaled_data'].groupby(level='Ticker'):
        ticker_data = group.values  # Shape: (T, n_features)
        ticker_dates = group.index.get_level_values('Date')
        
        if len(ticker_data) < context_length + 1:  # Need at least context + 1 for training
            print(f"Warning: {ticker_name} insufficient data. Skipping.")
            continue
        
        # Generate RL states for all possible windows
        rl_states = []
        prices = []
        dates = []
        
        # Get price data from raw data (try different column names)
        ticker_raw = data_info['raw_data'].xs(ticker_name, level='Ticker')
        price_column = None
        
        # Try different price column names
        for col_name in ['close', 'Close', 'CLOSE', 'adj_close', 'Adj Close']:
            if col_name in ticker_raw.columns:
                price_column = col_name
                break
        
        if price_column:
            price_series = ticker_raw[price_column].values
            print(f"Using {price_column} prices for {ticker_name}")
        else:
            # Fallback: use synthetic price data
            print(f"Warning: No price column found for {ticker_name}. Available columns: {list(ticker_raw.columns[:10])}...")
            print(f"Using synthetic prices.")
            price_series = np.random.uniform(100, 200, len(ticker_data))
        
        model.eval()
        with torch.no_grad():
            # Generate RL states for sliding windows
            for i in range(context_length, len(ticker_data)):
                # Get context window
                context_window = ticker_data[i-context_length:i]  # Shape: (context_length, n_features)
                context_tensor = torch.FloatTensor(context_window).unsqueeze(0).to(device)  # (1, context_length, n_features)
                
                # Generate RL state
                rl_state = model(past_values=context_tensor)  # Shape: (1, rl_state_size)
                rl_state = rl_state.cpu().numpy().squeeze()  # Shape: (rl_state_size,)
                
                rl_states.append(rl_state)
                prices.append(price_series[i] if i < len(price_series) else price_series[-1])
                dates.append(ticker_dates[i])
        
        rl_states_data[ticker_name] = {
            'rl_states': np.array(rl_states),  # Shape: (T-context_length, rl_state_size)
            'prices': np.array(prices),        # Shape: (T-context_length,)
            'dates': dates,
            'scaler': data_info['scalers'][ticker_name]
        }
        
        print(f"Generated {len(rl_states)} RL states for {ticker_name}")
    
    return rl_states_data

def train_rl_agent_with_tst(
    agent_type: str = "PPO",
    ticker: str = None,
    model_dir: str = None,
    data_path: str = None,
    epochs: int = 10,
    gamma: float = 0.99,
    use_gae: bool = True,
    lam: float = 0.95,
    save_agent: bool = True,
    output_dir: str = None,
    multi_ticker: bool = True,
    max_tickers: int = 20,
    max_samples: int = 50000
):
    """
    Complete workflow: TST model → RL states → Agent training
    
    Args:
        agent_type (str): "PPO" or "SAC"
        ticker (str): Target ticker (if None and multi_ticker=False, use first available)
        model_dir (str): TST model directory
        data_path (str): Historical data path
        epochs (int): Training epochs
        gamma (float): Discount factor
        use_gae (bool): Use Generalized Advantage Estimation
        lam (float): GAE lambda parameter
        save_agent (bool): Save trained agent
        output_dir (str): Output directory for saved agent
        multi_ticker (bool): Train on multiple tickers for better generalization
        max_tickers (int): Maximum number of tickers to use (None = all available)
        
    Returns:
        dict: Contains trained agent and training metadata
    """
    print(f"=== Training {agent_type} Agent with TST-generated RL States ===")
    print(f"Multi-ticker training: {multi_ticker}")
    
    # Generate RL states from TST model
    rl_states_data = generate_rl_states_from_tst(
        ticker=ticker if not multi_ticker else None,  # If multi_ticker, get all tickers
        model_dir=model_dir,
        data_path=data_path
    )
    
    if not rl_states_data:
        raise ValueError("No RL states generated. Cannot proceed with training.")
    
    if multi_ticker:
        # Use multiple tickers for training
        available_tickers = list(rl_states_data.keys())
        if max_tickers and len(available_tickers) > max_tickers:
            # Select top N tickers by data size
            ticker_sizes = [(t, len(rl_states_data[t]['rl_states'])) for t in available_tickers]
            ticker_sizes.sort(key=lambda x: x[1], reverse=True)
            selected_tickers = [t[0] for t in ticker_sizes[:max_tickers]]
        else:
            selected_tickers = available_tickers
        
        print(f"Training on {len(selected_tickers)} tickers: {selected_tickers}")
        
        # Combine data from multiple tickers
        combined_rl_states = []
        combined_prices = []
        combined_dates = []
        ticker_info = {}
        
        total_samples = 0
        for ticker_name in selected_tickers:
            ticker_data = rl_states_data[ticker_name]
            rl_states = ticker_data['rl_states']
            prices = ticker_data['prices']
            dates = ticker_data['dates']
            
            combined_rl_states.append(rl_states)
            combined_prices.append(prices)
            combined_dates.extend(dates)
            
            ticker_info[ticker_name] = {
                'samples': len(rl_states),
                'price_range': (prices.min(), prices.max())
            }
            total_samples += len(rl_states)
            
            print(f"  {ticker_name}: {len(rl_states)} samples, price range: ${prices.min():.2f}-${prices.max():.2f}")
        
        # Concatenate all data
        final_rl_states = np.concatenate(combined_rl_states, axis=0)  # Shape: (total_T, 256)
        final_prices = np.concatenate(combined_prices, axis=0)        # Shape: (total_T,)
        
        training_label = f"multi_ticker_{len(selected_tickers)}"
        
    else:
        # Single ticker training (original behavior)
        if ticker and ticker in rl_states_data:
            selected_ticker = ticker
        else:
            selected_ticker = list(rl_states_data.keys())[0]
            print(f"Using single ticker: {selected_ticker}")
        
        ticker_data = rl_states_data[selected_ticker]
        final_rl_states = ticker_data['rl_states']  # Shape: (T, 256)
        final_prices = ticker_data['prices']        # Shape: (T,)
        selected_tickers = [selected_ticker]
        total_samples = len(final_rl_states)
        training_label = selected_ticker
    
    print(f"Total training data: {total_samples} timesteps")
    print(f"RL state shape: {final_rl_states.shape}")
    print(f"Price range: ${final_prices.min():.2f} - ${final_prices.max():.2f}")
    
    # Initialize early stopping parameters
    best_value_loss = float('inf')
    epochs_without_improvement = 0
    patience = 10  # Stop after 10 epochs without improvement in value loss
    
    # Initialize agent
    if agent_type == "PPO":
        agent = PPOAgent(state_dim=260)
    elif agent_type == "SAC":
        agent = SACAgent(state_dim=260)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    
    # Create directory for epoch-wise model saving
    if save_agent:
        if output_dir is None:
            output_dir = os.path.join(PROJECT_ROOT, 'rl_model_output')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp_base = datetime.now().strftime('%Y%m%d_%H%M%S')
        agent_base_filename = f"{agent_type.lower()}_agent_{training_label}"
        
        # Directory for this specific training run
        run_output_dir = os.path.join(output_dir, f"{agent_base_filename}_{timestamp_base}")
        os.makedirs(run_output_dir, exist_ok=True)
        print(f"Saving epoch models to: {run_output_dir}")
    
    # Training loop (moved from train_rl_agent)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting {agent_type} training on {device}")

    env = TSTEnv(final_rl_states, final_prices)

    for epoch in range(epochs):
        state = env.reset()
        done = False
        epoch_losses = []

        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_log_probs = []
        batch_values = []

        while not done:
            if agent_type == "PPO":
                action, log_prob = agent.select_action(state)
                value = agent.get_value(state)
            else:  # SAC
                action, log_prob = agent.select_action(state)
                value = agent.evaluate_value(state)

            next_state, reward, done, _ = env.step(action)

            batch_states.append(state["rl_state"])
            batch_actions.append(action)
            batch_rewards.append(reward)
            batch_log_probs.append(log_prob)
            batch_values.append(value)
            state = next_state

        values_np = [v.item() if isinstance(v, torch.Tensor) else v for v in batch_values]
        if use_gae:
            advantages = compute_gae(batch_rewards, values_np, gamma=gamma, lam=lam)
            returns = [a + v for a, v in zip(advantages, values_np)]
        else:
            returns = compute_returns(batch_rewards, gamma=gamma)
            advantages = [r - v for r, v in zip(returns, values_np)]

        # cash_ratio, stock_ratio, portfolio_val_norm, has_shares_flag (0 for no shares initially)
        dummy_portfolio = np.array([1.0, 0.0, 0.0, 0.0])
        batch_states_260 = np.array([np.concatenate([s, dummy_portfolio]) for s in batch_states])
        
        transitions = {
            'states': torch.FloatTensor(batch_states_260).to(device),
            'actions': torch.LongTensor(batch_actions).to(device),
            'log_probs': torch.stack(batch_log_probs).to(device),
            'returns': torch.FloatTensor(returns).to(device),
            'advantages': torch.FloatTensor(advantages).to(device),
            'next_states': torch.FloatTensor(np.concatenate([batch_states_260[1:], [batch_states_260[-1]]])).to(device),
            'dones': torch.FloatTensor([0.0] * (len(batch_rewards) - 1) + [1.0]).to(device)
        }

        loss_info = agent.update_policy(transitions)
        epoch_losses.append(loss_info)
        
        # --- Output and Saving --- 
        if device.type == 'cuda':
            gpu_memory_used = torch.cuda.memory_allocated(device) / 1024**3
            gpu_memory_cached = torch.cuda.memory_reserved(device) / 1024**3
            gpu_memory_info = f" | GPU: {gpu_memory_used:.2f}GB used, {gpu_memory_cached:.2f}GB cached"
        else:
            gpu_memory_info = ""

        total_loss_val = loss_info['total_loss']
        policy_loss_val = loss_info['policy_loss']
        value_loss_val = loss_info['value_loss']
        current_lr_val = loss_info.get('current_lr', 'N/A')
        value_weight_val = loss_info.get('value_weight', 'N/A')
        loss_ratio_val = loss_info.get('loss_ratio', 'N/A')
        lr_changed_val = loss_info.get('lr_changed', False)
        lr_reason_val = loss_info.get('lr_reason', '')
        
        loss_str = f"Epoch {epoch + 1}/{epochs} | Total: {total_loss_val:.4f} | Policy: {policy_loss_val:.4f} | Value: {value_loss_val:.4f}"
        if 'q_loss_avg' in loss_info:
            loss_str += f" | Q-Avg: {loss_info['q_loss_avg']:.4f}"
        if current_lr_val != 'N/A': loss_str += f" | LR: {current_lr_val:.2e}"
        if value_weight_val != 'N/A': loss_str += f" | VW: {value_weight_val:.2f}"
        if loss_ratio_val != 'N/A': loss_str += f" | Ratio: {loss_ratio_val:.1f}"
        loss_str += gpu_memory_info
        print(loss_str)

        if lr_changed_val:
            print(f"    🔄 LR adjusted: {lr_reason_val} (Ratio: {loss_ratio_val:.1f})")

        # --- Epoch-wise Model Saving --- 
        if save_agent:
            epoch_agent_filename = f"{agent_base_filename}_epoch_{epoch+1}.pt"
            epoch_agent_path = os.path.join(run_output_dir, epoch_agent_filename)
            
            # Determine fallback config based on agent type
            if agent_type == "PPO":
                fallback_config = agent.actor.state_dict()
            else: # SAC
                fallback_config = agent.policy_net.state_dict()
                
            save_data_epoch = {
                'epoch': epoch + 1,
                'config': agent.config if hasattr(agent, 'config') else fallback_config
            }
            if agent_type == "PPO":
                save_data_epoch.update({
                    'actor_state_dict': agent.actor.state_dict(),
                    'critic_state_dict': agent.critic.state_dict()
                })
            else: # SAC
                save_data_epoch.update({
                    'policy_net_state_dict': agent.policy_net.state_dict(),
                    'q_net1_state_dict': agent.q_net1.state_dict(),
                    'q_net2_state_dict': agent.q_net2.state_dict(),
                    'value_net_state_dict': agent.value_net.state_dict(),
                    'target_value_net_state_dict': agent.target_value_net.state_dict()
                })
            torch.save(save_data_epoch, epoch_agent_path)
            # print(f"    💾 Saved epoch model: {epoch_agent_path}")

        # --- Early Stopping Logic --- 
        current_value_loss = value_loss_val
        if current_value_loss < best_value_loss:
            best_value_loss = current_value_loss
            epochs_without_improvement = 0
            if save_agent:
                best_agent_filename = f"{agent_base_filename}_best_model.pt"
                best_agent_path = os.path.join(run_output_dir, best_agent_filename)
                torch.save(save_data_epoch, best_agent_path) # Save current best model
                print(f"    🏆 New best model saved (Value Loss: {best_value_loss:.4f})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"🛑 Early stopping after {epoch + 1} epochs: Value loss did not improve for {patience} epochs.")
            break
            
        # Policy-based early stopping (if policy is too stable and value is not improving)
        if agent_type == "PPO" and abs(policy_loss_val) < 0.001 and current_value_loss > 0.1 and epoch > 20:
            print(f"🛑 Early stopping: PPO Policy likely overfitted (Policy: {policy_loss_val:.4f}, Value: {current_value_loss:.4f})")
            break
        if agent_type == "SAC" and abs(policy_loss_val) < 0.05 and current_value_loss > 0.1 and epoch > 20:
            print(f"🛑 Early stopping: SAC Policy likely overfitted (Policy: {policy_loss_val:.4f}, Value: {current_value_loss:.4f})")
            break

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU memory cleared after training.")
    
    print(f"{agent_type} training process completed.")
    
    return {
        'agent': agent,
        'agent_type': agent_type,
        'training_tickers': selected_tickers,
        'multi_ticker': multi_ticker,
        'training_data_size': total_samples,
        'rl_states_data': rl_states_data,
        'agent_path': None if not save_agent else os.path.join(run_output_dir, f"{agent_base_filename}_best_model.pt"),
        'ticker_info': ticker_info if multi_ticker else None
    }

def train_rl_agent_multi_ticker(
    agent_type: str = "PPO",
    tickers: list = None,
    model_dir: str = None,
    data_path: str = None,
    epochs: int = 10,
    gamma: float = 0.99,
    use_gae: bool = True,
    lam: float = 0.95,
    save_agent: bool = True,
    output_dir: str = None,
    max_tickers: int = 20,
    min_samples_per_ticker: int = 100,
    max_samples: int = 129723
):
    """
    Train RL agent on multiple tickers for better generalization.
    
    Args:
        agent_type (str): "PPO" or "SAC"
        tickers (list): List of specific tickers (if None, use all available)
        model_dir (str): TST model directory
        data_path (str): Historical data path
        epochs (int): Training epochs
        gamma (float): Discount factor
        use_gae (bool): Use Generalized Advantage Estimation
        lam (float): GAE lambda parameter
        save_agent (bool): Save trained agent
        output_dir (str): Output directory for saved agent
        max_tickers (int): Maximum number of tickers to use
        min_samples_per_ticker (int): Minimum samples required per ticker
        
    Returns:
        dict: Contains trained agent and training metadata
    """
    print(f"=== Training {agent_type} Agent on Multiple Tickers ===")
    
    # Generate RL states from TST model for all tickers
    rl_states_data = generate_rl_states_from_tst(
        ticker=None,  # Get all tickers
        model_dir=model_dir,
        data_path=data_path
    )
    
    if not rl_states_data:
        raise ValueError("No RL states generated. Cannot proceed with training.")
    
    # Filter tickers based on criteria
    available_tickers = list(rl_states_data.keys())
    
    # Filter by specific tickers if provided
    if tickers:
        available_tickers = [t for t in available_tickers if t in tickers]
        if not available_tickers:
            raise ValueError(f"None of the specified tickers {tickers} found in data")
    
    # Filter by minimum samples
    valid_tickers = []
    for ticker in available_tickers:
        if len(rl_states_data[ticker]['rl_states']) >= min_samples_per_ticker:
            valid_tickers.append(ticker)
        else:
            print(f"Skipping {ticker}: insufficient data ({len(rl_states_data[ticker]['rl_states'])} < {min_samples_per_ticker})")
    
    if not valid_tickers:
        raise ValueError(f"No tickers have sufficient data (>= {min_samples_per_ticker} samples)")
    
    # Limit number of tickers if specified
    if max_tickers and len(valid_tickers) > max_tickers:
        # Sort by data size and take top N
        ticker_sizes = [(t, len(rl_states_data[t]['rl_states'])) for t in valid_tickers]
        ticker_sizes.sort(key=lambda x: x[1], reverse=True)
        selected_tickers = [t[0] for t in ticker_sizes[:max_tickers]]
    else:
        selected_tickers = valid_tickers
    
    print(f"Selected {len(selected_tickers)} tickers for training: {selected_tickers}")
    
    # Combine data from multiple tickers
    combined_rl_states = []
    combined_prices = []
    ticker_info = {}
    total_samples = 0
    
    for ticker_name in selected_tickers:
        ticker_data = rl_states_data[ticker_name]
        rl_states = ticker_data['rl_states']
        prices = ticker_data['prices']
        
        combined_rl_states.append(rl_states)
        combined_prices.append(prices)
        
        ticker_info[ticker_name] = {
            'samples': len(rl_states),
            'price_range': (float(prices.min()), float(prices.max())),
            'price_mean': float(prices.mean()),
            'price_std': float(prices.std())
        }
        total_samples += len(rl_states)
        
        print(f"  {ticker_name}: {len(rl_states)} samples, "
              f"price range: ${prices.min():.2f}-${prices.max():.2f}, "
              f"mean: ${prices.mean():.2f}")
    
    # Concatenate all data
    final_rl_states = np.concatenate(combined_rl_states, axis=0)  # Shape: (total_T, 256)
    final_prices = np.concatenate(combined_prices, axis=0)        # Shape: (total_T,)
    
    print(f"\nCombined training data:")
    print(f"  Total samples: {total_samples}")
    print(f"  RL state shape: {final_rl_states.shape}")
    print(f"  Price range: ${final_prices.min():.2f} - ${final_prices.max():.2f}")
    print(f"  Price mean: ${final_prices.mean():.2f}")
    
    # Shuffle the combined data for better training
    indices = np.random.permutation(len(final_rl_states))
    final_rl_states = final_rl_states[indices]
    final_prices = final_prices[indices]
    print(f"  Data shuffled for better training")
    
    # Initialize agent (moved here to avoid re-initialization)
    if agent_type == "PPO":
        agent = PPOAgent(state_dim=260)
    elif agent_type == "SAC":
        agent = SACAgent(state_dim=260)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    
    # --- Training Loop moved to here from train_rl_agent_with_tst --- 
    # Initialize early stopping parameters
    best_value_loss = float('inf')
    epochs_without_improvement = 0
    patience = 15  # More patience for multi-ticker training
    
    # Create directory for epoch-wise model saving
    agent_path = None # Initialize agent_path
    if save_agent:
        if output_dir is None:
            output_dir = os.path.join(PROJECT_ROOT, 'rl_model_output')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp_base = datetime.now().strftime('%Y%m%d_%H%M%S')
        agent_base_filename = f"{agent_type.lower()}_agent_multi_{len(selected_tickers)}tickers"
        
        run_output_dir = os.path.join(output_dir, f"{agent_base_filename}_{timestamp_base}")
        os.makedirs(run_output_dir, exist_ok=True)
        print(f"Saving epoch models to: {run_output_dir}")
        agent_path = os.path.join(run_output_dir, f"{agent_base_filename}_best_model.pt") # For return value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nStarting {agent_type} multi-ticker training on {device}")

    if agent_type == "PPO":
        agent.actor = agent.actor.to(device)
        agent.actor_old = agent.actor_old.to(device)
        agent.critic = agent.critic.to(device)
    else:  # SAC
        agent.policy_net = agent.policy_net.to(device)
        agent.q_net1 = agent.q_net1.to(device)
        agent.q_net2 = agent.q_net2.to(device)
        agent.value_net = agent.value_net.to(device)
        agent.target_value_net = agent.target_value_net.to(device)
    print(f"{agent_type} networks moved to {device}")
    
    env = TSTEnv(final_rl_states, final_prices)

    for epoch in range(epochs):
        state = env.reset()
        done = False

        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_log_probs = []
        batch_values = []

        while not done:
            if agent_type == "PPO":
                action, log_prob = agent.select_action(state)
                value = agent.get_value(state)
            else:  # SAC
                action, log_prob = agent.select_action(state)
                value = agent.evaluate_value(state)

            next_state, reward, done, _ = env.step(action)

            batch_states.append(state["rl_state"])
            batch_actions.append(action)
            batch_rewards.append(reward)
            batch_log_probs.append(log_prob)
            batch_values.append(value)
            state = next_state

        values_np = [v.item() if isinstance(v, torch.Tensor) else v for v in batch_values]
        if use_gae:
            advantages = compute_gae(batch_rewards, values_np, gamma=gamma, lam=lam)
            returns = [a + v for a, v in zip(advantages, values_np)]
        else:
            returns = compute_returns(batch_rewards, gamma=gamma)
            advantages = [r - v for r, v in zip(returns, values_np)]

        # cash_ratio, stock_ratio, portfolio_val_norm, has_shares_flag (0 for no shares initially)
        dummy_portfolio = np.array([1.0, 0.0, 0.0, 0.0])
        batch_states_260 = np.array([np.concatenate([s, dummy_portfolio]) for s in batch_states])
        
        transitions = {
            'states': torch.FloatTensor(batch_states_260).to(device),
            'actions': torch.LongTensor(batch_actions).to(device),
            'log_probs': torch.stack(batch_log_probs).to(device),
            'returns': torch.FloatTensor(returns).to(device),
            'advantages': torch.FloatTensor(advantages).to(device),
            'next_states': torch.FloatTensor(np.concatenate([batch_states_260[1:], [batch_states_260[-1]]])).to(device),
            'dones': torch.FloatTensor([0.0] * (len(batch_rewards) - 1) + [1.0]).to(device)
        }

        loss_info = agent.update_policy(transitions)
        
        # --- Output and Saving --- 
        if device.type == 'cuda':
            gpu_memory_used = torch.cuda.memory_allocated(device) / 1024**3
            gpu_memory_cached = torch.cuda.memory_reserved(device) / 1024**3
            gpu_memory_info = f" | GPU: {gpu_memory_used:.2f}GB used, {gpu_memory_cached:.2f}GB cached"
        else:
            gpu_memory_info = ""

        total_loss_val = loss_info['total_loss']
        policy_loss_val = loss_info['policy_loss']
        value_loss_val = loss_info['value_loss']
        current_lr_val = loss_info.get('current_lr', 'N/A')
        value_weight_val = loss_info.get('value_weight', 'N/A')
        loss_ratio_val = loss_info.get('loss_ratio', 'N/A')
        lr_changed_val = loss_info.get('lr_changed', False)
        lr_reason_val = loss_info.get('lr_reason', '')
        
        loss_str = f"Epoch {epoch + 1}/{epochs} | Total: {total_loss_val:.4f} | Policy: {policy_loss_val:.4f} | Value: {value_loss_val:.4f}"
        if 'q_loss_avg' in loss_info:
            loss_str += f" | Q-Avg: {loss_info['q_loss_avg']:.4f}"
        if current_lr_val != 'N/A': loss_str += f" | LR: {current_lr_val:.2e}"
        if value_weight_val != 'N/A': loss_str += f" | VW: {value_weight_val:.2f}"
        if loss_ratio_val != 'N/A': loss_str += f" | Ratio: {loss_ratio_val:.1f}"
        loss_str += gpu_memory_info
        print(loss_str)

        if lr_changed_val:
            print(f"    🔄 LR adjusted: {lr_reason_val} (Ratio: {loss_ratio_val:.1f})")

        # --- Epoch-wise Model Saving --- 
        if save_agent:
            epoch_agent_filename = f"{agent_base_filename}_epoch_{epoch+1}.pt"
            epoch_agent_path = os.path.join(run_output_dir, epoch_agent_filename)
            
            # Determine fallback config based on agent type
            if agent_type == "PPO":
                fallback_config = agent.actor.state_dict()
            else: # SAC
                fallback_config = agent.policy_net.state_dict()
                
            save_data_epoch = {
                'epoch': epoch + 1,
                'config': agent.config if hasattr(agent, 'config') else fallback_config
            }
            if agent_type == "PPO":
                save_data_epoch.update({
                    'actor_state_dict': agent.actor.state_dict(),
                    'critic_state_dict': agent.critic.state_dict()
                })
            else: # SAC
                save_data_epoch.update({
                    'policy_net_state_dict': agent.policy_net.state_dict(),
                    'q_net1_state_dict': agent.q_net1.state_dict(),
                    'q_net2_state_dict': agent.q_net2.state_dict(),
                    'value_net_state_dict': agent.value_net.state_dict(),
                    'target_value_net_state_dict': agent.target_value_net.state_dict()
                })
            torch.save(save_data_epoch, epoch_agent_path)
            # print(f"    💾 Saved epoch model: {epoch_agent_path}")

        # --- Early Stopping Logic --- 
        current_value_loss = value_loss_val
        if current_value_loss < best_value_loss:
            best_value_loss = current_value_loss
            epochs_without_improvement = 0
            if save_agent:
                best_agent_filename = f"{agent_base_filename}_best_model.pt"
                # Path for best model is now agent_path, defined earlier
                torch.save(save_data_epoch, agent_path) 
                print(f"    🏆 New best model saved (Value Loss: {best_value_loss:.4f})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"🛑 Early stopping after {epoch + 1} epochs: Value loss did not improve for {patience} epochs.")
            break
            
        if agent_type == "PPO" and abs(policy_loss_val) < 0.001 and current_value_loss > 0.1 and epoch > 30: # More epochs for multi-ticker
            print(f"🛑 Early stopping: PPO Policy overfitted (Policy: {policy_loss_val:.4f}, Value: {current_value_loss:.4f})")
            break
        if agent_type == "SAC" and abs(policy_loss_val) < 0.05 and current_value_loss > 0.1 and epoch > 30:
            print(f"🛑 Early stopping: SAC Policy overfitted (Policy: {policy_loss_val:.4f}, Value: {current_value_loss:.4f})")
            break

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU memory cleared after training.")
    
    print(f"{agent_type} multi-ticker training process completed.")
    
    return {
        'agent': agent,
        'agent_type': agent_type,
        'training_tickers': selected_tickers,
        'multi_ticker': True,
        'training_data_size': total_samples,
        'ticker_info': ticker_info,
        'rl_states_data': rl_states_data,
        'agent_path': agent_path
    }

def main():
    """Main function with command-line interface for TST-RL training workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RL Agent with TST-generated states')
    parser.add_argument('--agent_type', type=str, choices=['PPO', 'SAC'],
                       help='Type of RL agent to train')
    parser.add_argument('--ticker', type=str, help='Target ticker symbol (e.g., AAPL)')
    parser.add_argument('--model_dir', type=str, 
                       default=os.path.join(PROJECT_ROOT, 'tst_model_output'),
                       help='Directory containing trained TST models')
    parser.add_argument('--data_path', type=str,
                       default=os.path.join(PROJECT_ROOT, 'all_tickers_historical_features.csv'),
                       help='Path to historical data CSV')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--use_gae', action='store_true', default=True,
                       help='Use Generalized Advantage Estimation')
    parser.add_argument('--lam', type=float, default=0.95,
                       help='GAE lambda parameter')
    parser.add_argument('--output_dir', type=str,
                       default=os.path.join(PROJECT_ROOT, 'rl_model_output'),
                       help='Output directory for trained agent')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save the trained agent')
    parser.add_argument('--multi_ticker', action='store_true', default=True,
                       help='Train on multiple tickers for better generalization')
    parser.add_argument('--single_ticker', action='store_true',
                       help='Force single ticker training (overrides --multi_ticker)')
    parser.add_argument('--max_tickers', type=int, default=20,
                       help='Maximum number of tickers to use for multi-ticker training (default: 20, matches available tickers)')
    parser.add_argument('--min_samples', type=int, default=100,
                       help='Minimum samples required per ticker')
    parser.add_argument('--tickers', nargs='+', 
                       help='Specific tickers to use (e.g., --tickers AAPL MSFT GOOGL)')
    parser.add_argument('--max_samples', type=int, default=129723,
                       help='Maximum number of samples to use for training (default: 130000)')
    
    args = parser.parse_args()
    
    # Determine training mode
    use_multi_ticker = args.multi_ticker and not args.single_ticker
    
    print("=== TST-RL Training Workflow ===")
    print(f"Agent type: {args.agent_type}")
    print(f"Training mode: {'Multi-ticker' if use_multi_ticker else 'Single-ticker'}")
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    if use_multi_ticker:
        print(f"Max tickers: {args.max_tickers}")
        print(f"Min samples per ticker: {args.min_samples}")
        if args.tickers:
            print(f"Specific tickers: {args.tickers}")
    else:
        print(f"Target ticker: {args.ticker or 'Auto-select'}")
    print(f"TST model directory: {args.model_dir}")
    print(f"Data path: {args.data_path}")
    print(f"Training epochs: {args.epochs}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        if use_multi_ticker:
            # Run multi-ticker training
            result = train_rl_agent_multi_ticker(
                agent_type=args.agent_type,
                tickers=args.tickers,
                model_dir=args.model_dir,
                data_path=args.data_path,
                epochs=args.epochs,
                gamma=args.gamma,
                use_gae=args.use_gae,
                lam=args.lam,
                save_agent=not args.no_save,
                output_dir=args.output_dir,
                max_tickers=args.max_tickers,
                min_samples_per_ticker=args.min_samples,
                max_samples=args.max_samples
            )
        else:
            # Run single-ticker training (original)
            result = train_rl_agent_with_tst(
                agent_type=args.agent_type,
                ticker=args.ticker,
                model_dir=args.model_dir,
                data_path=args.data_path,
                epochs=args.epochs,
                gamma=args.gamma,
                use_gae=args.use_gae,
                lam=args.lam,
                save_agent=not args.no_save,
                output_dir=args.output_dir,
                max_samples=args.max_samples
            )
        
        print("\n=== Training Complete ===")
        print(f"Agent type: {result['agent_type']}")
        
        if result.get('multi_ticker', False):
            print(f"Training mode: Multi-ticker")
            print(f"Tickers used: {result['training_tickers']}")
            print(f"Number of tickers: {len(result['training_tickers'])}")
            if result.get('ticker_info'):
                print("Ticker details:")
                for ticker, info in result['ticker_info'].items():
                    print(f"  {ticker}: {info['samples']} samples, "
                          f"price range: ${info['price_range'][0]:.2f}-${info['price_range'][1]:.2f}")
        else:
            print(f"Training mode: Single-ticker")
            print(f"Ticker: {result.get('ticker', 'Unknown')}")
        
        print(f"Total training data: {result['training_data_size']} timesteps")
        if result['agent_path']:
            print(f"Agent saved to: {result['agent_path']}")
        
        # Test the trained agent
        print("\n=== Testing Trained Agent ===")
        agent = result['agent']
        
        # Get test RL state from first available ticker
        if result.get('multi_ticker', False):
            test_ticker = result['training_tickers'][0]
        else:
            test_ticker = result.get('ticker') or list(result['rl_states_data'].keys())[0]
        
        test_rl_state = result['rl_states_data'][test_ticker]['rl_states'][0]
        
        # Test predict_action interface
        # state_vector = {"tst_rl_state": test_rl_state}
        # For testing, need to simulate the full state dict including portfolio info for _prepare_state
        state_vector = {
            "tst_rl_state": test_rl_state,
            "cash": 10000.0, # Dummy cash
            "shares": 0,      # Dummy shares (implies has_shares_flag = 0.0 in agent._prepare_state)
            "current_price": result['rl_states_data'][test_ticker]['prices'][0] # Dummy price
        }
        action_result = agent.predict_action(state_vector)
        print(f"Test action (using {test_ticker} RL state): {action_result}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU memory cleared in finally block (e.g., after interrupt or error).")

if __name__ == '__main__':
    main()
