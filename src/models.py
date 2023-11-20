import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Transformer(nn.Module):

    def __init__(self, middle_size, input_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.encoder = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.linear2 = nn.Linear(middle_size * 512, 512)

    def forward(self, x):
        x = self.linear1(x)
        output = self.encoder(x)
        reshaped = output.view(x.shape[0], -1)
        return self.linear2(reshaped)


class MM_fusion_model(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a multi-modal fusion model.
        """
        super(MM_fusion_model, self).__init__()
        self.extract_l = Transformer(50, 300)
        self.extract_a = Transformer(375, 5)
        self.extract_v = Transformer(500, 20)
        hidden_size = 512 * 3
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.regression = nn.Linear(hidden_size, 1)

    def forward(self, x_l, x_a, x_v):
        # print(x_l.shape, x_a.shape, x_v.shape)
        # torch.Size([24, 50, 300]) torch.Size([24, 375, 5]) torch.Size([24, 500, 20]
        feature_l = self.extract_l(x_l)
        feature_a = self.extract_a(x_a)
        feature_v = self.extract_v(x_v)
        concat_feature = torch.cat([feature_l, feature_a, feature_v], dim = 1)
        hidden = self.linear(concat_feature)
        output = self.regression(hidden)
        return output


class AttentionMechanism(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        weights = F.softmax(self.attention(x), dim=1)
        return (x * weights).sum(dim=1, keepdim=True)

class MM_coordination_model(nn.Module):
    def __init__(self, hyp_params):
        super(MM_coordination_model, self).__init__()
        self.extract_l = Transformer(50, 300)
        self.extract_a = Transformer(375, 5)
        self.extract_v = Transformer(500, 20)

        # Attention mechanisms for each modality
        self.attention_l = AttentionMechanism(512)
        self.attention_a = AttentionMechanism(512)
        self.attention_v = AttentionMechanism(512)

        # Final layers for decision making
        hidden_size = 512 * 3
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.regression = nn.Linear(hidden_size, 1)

    def forward(self, x_l, x_a, x_v):
        feature_l = self.extract_l(x_l)
        feature_a = self.extract_a(x_a)
        feature_v = self.extract_v(x_v)

        # Apply attention to each modality
        attended_l = self.attention_l(feature_l)
        attended_a = self.attention_a(feature_a)
        attended_v = self.attention_v(feature_v)

        # Coordinate the features
        combined_score = attended_l + attended_a + attended_v
        # print(combined_score.shape, feature_l.shape)
        feature_l = combined_score * feature_l
        feature_a = combined_score * feature_a
        feature_v = combined_score * feature_v
        combined_feature = torch.cat([feature_l, feature_a, feature_v], dim=1)

        # Final decision making
        hidden = self.linear(combined_feature)
        output = self.regression(hidden)
        return output
