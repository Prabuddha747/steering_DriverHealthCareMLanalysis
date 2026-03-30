"""
LSTM Autoencoder model for driver stress classification from sequence data.
Encoder-decoder structure with classifier head for stress level prediction.
"""
import torch
import torch.nn as nn


class LSTMAEModel(nn.Module):
    """LSTM Autoencoder + classifier with dropout for regularization."""

    def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout: float = 0.35):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.0)
        self.decoder = nn.LSTM(hidden_size, input_size, batch_first=True, dropout=0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        z = self.dropout(h[-1])
        return self.fc(z)
