import torch
import torch.nn as nn


class TemporalLSTM(nn.Module):
    """Model temporal patterns across video frame features."""

    def __init__(
        self,
        input_size=1280,
        hidden_size=256,
        num_layers=2,
        dropout=0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape: (batch, sequence_length, input_size)

        Returns
        -------
        torch.Tensor
            Authenticity logit.
        """

        lstm_output, _ = self.lstm(x)

        # Use the final time step.
        final_output = lstm_output[:, -1, :]

        logits = self.classifier(final_output)

        return logits