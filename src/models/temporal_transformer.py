
import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    """
    Transformer model for learning temporal patterns
    across video frame features.

    Input:
        [batch, num_frames, 1280]

    Output:
        [batch, 1]
    """

    def __init__(
        self,
        input_size=1280,
        hidden_size=256,
        num_heads=8,
        num_layers=2,
        dropout=0.3,
        max_frames=64,
    ):
        super().__init__()

        # --------------------------------------------------
        # Project EfficientNet features
        #
        # [batch, frames, 1280]
        #          ↓
        # [batch, frames, 256]
        # --------------------------------------------------

        self.feature_projection = nn.Linear(
            input_size,
            hidden_size,
        )

        # --------------------------------------------------
        # Learnable positional embeddings
        #
        # Transformers do not naturally understand
        # frame order, so we explicitly add position.
        # --------------------------------------------------

        self.position_embedding = nn.Parameter(
            torch.randn(1, max_frames, hidden_size)
        )

        # --------------------------------------------------
        # Transformer encoder
        # --------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # --------------------------------------------------
        # Classification head
        # --------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        """
        Forward pass.

        Input:
            x = [batch, frames, 1280]

        Output:
            logits = [batch, 1]
        """

        batch_size, num_frames, _ = x.shape

        # Project features
        x = self.feature_projection(x)

        # Add positional information
        x = x + self.position_embedding[:, :num_frames, :]

        # Transformer temporal modeling
        x = self.transformer(x)

        # Global temporal pooling
        x = x.mean(dim=1)

        # Binary classification
        logits = self.classifier(x)

        return logits

