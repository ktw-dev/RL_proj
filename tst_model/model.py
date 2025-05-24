# tst_model/model.py: Defines a simple Time Series Transformer (TST) architecture using PyTorch.
# 05-24-2025 19:25
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TSTModel(nn.Module):
    def __init__(self, config_dict):
        """
        Simple Time Series Transformer model using standard PyTorch components.
        
        Args:
            config_dict (dict): Configuration parameters:
                - input_size: Number of input features (80 for our case)
                - prediction_length: Number of future steps to predict
                - context_length: Number of past steps to use as context
                - d_model: Transformer model dimension
                - n_head: Number of attention heads
                - n_layer: Number of transformer layers
                - rl_state_size: Output dimension for RL agent state
        """
        super().__init__()
        
        self.input_size = config_dict['input_size']
        self.prediction_length = config_dict['prediction_length']
        self.context_length = config_dict['context_length']
        self.d_model = config_dict.get('d_model', 128)
        self.n_head = config_dict.get('n_head', 8)
        self.n_layer = config_dict.get('n_layer', 4)
        self.rl_state_size = config_dict['rl_state_size']
        
        # Input projection: map input features to d_model dimension
        self.input_projection = nn.Linear(self.input_size, self.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model, max_len=self.context_length + self.prediction_length)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dim_feedforward=self.d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.n_layer)
        
        # Prediction head: predict future values
        self.prediction_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model, self.input_size)
        )
        
        # RL state head: extract meaningful state for RL agent
        self.rl_head = nn.Sequential(
            nn.Linear(self.d_model * self.context_length, self.d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model, self.rl_state_size)
        )
        
        self.criterion = nn.MSELoss()
        
    def forward(self, past_values, future_values=None):
        """
        Forward pass of the TST model.
        
        Args:
            past_values: (batch_size, context_length, input_size) - Historical data
            future_values: (batch_size, prediction_length, input_size) - Target future values (for training)
            
        Returns:
            If training: loss tensor
            If inference: RL state tensor (batch_size, rl_state_size)
        """
        batch_size, seq_len, _ = past_values.shape
        
        # Project input to d_model dimension
        # (batch_size, context_length, input_size) -> (batch_size, context_length, d_model)
        x = self.input_projection(past_values)
        
        # Add positional encoding
        # Need to transpose for positional encoding: (context_length, batch_size, d_model)
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # Back to (batch_size, context_length, d_model)
        
        # Pass through transformer encoder
        encoded = self.transformer_encoder(x)  # (batch_size, context_length, d_model)
        
        if self.training and future_values is not None:
            # Training mode: predict future values and compute loss
            # Use the last encoded state to predict future steps
            last_encoded = encoded[:, -1:, :]  # (batch_size, 1, d_model)
            
            # Predict each future step autoregressively
            predictions = []
            current_input = last_encoded
            
            for i in range(self.prediction_length):
                # Predict next step
                pred = self.prediction_head(current_input)  # (batch_size, 1, input_size)
                predictions.append(pred)
                
                # Use prediction as input for next step (teacher forcing alternative)
                # Project back to d_model for next iteration
                current_input = self.input_projection(pred)
            
            # Stack predictions: (batch_size, prediction_length, input_size)
            predicted_values = torch.cat(predictions, dim=1)
            
            # Compute loss
            loss = self.criterion(predicted_values, future_values)
            
            # Return loss in a format compatible with training loop
            class ModelOutput:
                def __init__(self, loss):
                    self.loss = loss
            
            return ModelOutput(loss)
        
        else:
            # Inference mode: return RL state
            # Flatten the encoded sequence for RL head
            # (batch_size, context_length, d_model) -> (batch_size, context_length * d_model)
            flattened = encoded.reshape(batch_size, -1)
            
            # Generate RL state
            rl_state = self.rl_head(flattened)  # (batch_size, rl_state_size)
            
            return rl_state
    
    def predict_future(self, past_values, num_steps=None):
        """
        Predict future values for inference.
        
        Args:
            past_values: (batch_size, context_length, input_size)
            num_steps: Number of steps to predict (default: prediction_length)
            
        Returns:
            predictions: (batch_size, num_steps, input_size)
        """
        if num_steps is None:
            num_steps = self.prediction_length
            
        self.eval()
        with torch.no_grad():
            batch_size = past_values.shape[0]
            
            # Encode past values
            x = self.input_projection(past_values)
            x = x.transpose(0, 1)
            x = self.pos_encoding(x)
            x = x.transpose(0, 1)
            encoded = self.transformer_encoder(x)
            
            # Predict future steps
            predictions = []
            last_encoded = encoded[:, -1:, :]
            current_input = last_encoded
            
            for i in range(num_steps):
                pred = self.prediction_head(current_input)
                predictions.append(pred)
                current_input = self.input_projection(pred)
            
            return torch.cat(predictions, dim=1)

# Example usage and configuration
def create_tst_model(input_size=80, prediction_length=10, context_length=60, rl_state_size=256):
    """Helper function to create a TST model with common parameters."""
    config = {
        'input_size': input_size,
        'prediction_length': prediction_length,
        'context_length': context_length,
        'd_model': 128,
        'n_head': 8,
        'n_layer': 4,
        'rl_state_size': rl_state_size
    }
    return TSTModel(config)

# Test the model
if __name__ == "__main__":
    # Test configuration based on our CSV analysis
    config = {
        'input_size': 80,  # 80 numeric features from CSV
        'prediction_length': 10,
        'context_length': 60,
        'd_model': 128,
        'n_head': 8,
        'n_layer': 4,
        'rl_state_size': 256
    }
    
    model = TSTModel(config)
    
    # Test with dummy data
    batch_size = 4
    past_values = torch.randn(batch_size, config['context_length'], config['input_size'])
    future_values = torch.randn(batch_size, config['prediction_length'], config['input_size'])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test training mode
    model.train()
    output = model(past_values, future_values)
    print(f"Training loss: {output.loss.item():.4f}")
    
    # Test inference mode
    model.eval()
    with torch.no_grad():
        rl_state = model(past_values)
        print(f"RL state shape: {rl_state.shape}")
        
        predictions = model.predict_future(past_values)
        print(f"Predictions shape: {predictions.shape}")
    
    print("TST Model test completed successfully!") 