import torch
import torch.nn as nn
from torchvision import models


class EfficientNetFeatureExtractor(nn.Module):
    """
    EfficientNet-B0 feature extractor.

    Converts aligned face images into 1280-dimensional
    spatial feature vectors.
    """

    def __init__(self):
        super().__init__()

        # Load ImageNet-pretrained EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.DEFAULT

        self.model = models.efficientnet_b0(
            weights=weights
        )

        # Remove the final classification layer
        # so the network outputs feature vectors.
        self.model.classifier = nn.Identity()

        # ImageNet preprocessing associated with
        # the pretrained EfficientNet weights.
        self.preprocess = weights.transforms()

        # Feature extractor is frozen for this baseline.
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.model.eval()

    def forward(self, images):
        """
        Extract spatial features.

        Input:
            [batch, 3, 224, 224]

        Output:
            [batch, 1280]
        """

        with torch.no_grad():
            features = self.model(images)

        return features