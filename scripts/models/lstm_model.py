"""
LSTM model for driver stress classification from sequence data.
"""
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """LSTM for sequence classification with dropout for regularization."""

    def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout: float = 0.35):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.fc(self.dropout(h[-1]))
