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


class CrossModalAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(CrossModalAttention, self).__init__()
        self.attention_layer = nn.Linear(input_dim, attention_dim)

    def forward(self, x1, x2):
        # x1, x2 are inputs from two different modalities
        # Apply attention from x1 to x2
        attention = F.softmax(self.attention_layer(x1), dim=1)
        print(attention.shape, x1.shape, x2.shape)
        attended_x2 = attention.unsqueeze(2) @ x2.unsqueeze(1)
        return attended_x2.sum(dim=1)


class MM_coordination_model(nn.Module):

    def __init__(self, config, text_dim=300, audio_dim=5, video_dim=20, hidden_dim=768):
        super(MM_coordination_model, self).__init__()
        # Linear layers for each modality
        self.text_layer = nn.Linear(text_dim * 50, hidden_dim)
        self.audio_layer = nn.Linear(audio_dim * 375, hidden_dim)
        self.video_layer = nn.Linear(video_dim * 500, hidden_dim)

        # Cross-modal attention layers
        self.audio_video_attention = CrossModalAttention(hidden_dim, hidden_dim)
        self.video_audio_attention = CrossModalAttention(hidden_dim, hidden_dim)
        self.text_audio_attention = CrossModalAttention(hidden_dim, hidden_dim)
        self.text_video_attention = CrossModalAttention(hidden_dim, hidden_dim)

        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.regression = nn.Linear(hidden_dim, 1)

    def forward(self, x_l, x_a=None, x_v=None):
        # Check which modalities are available
        # print(x_l.shape, x_a.shape, x_v.shape)
        audio_available = x_a is not None
        video_available = x_v is not None

        # Process available modalities
        text_features = F.relu(self.text_layer(x_l.view(x_l.shape[0], -1)))
        if audio_available and video_available:
            audio_features = F.relu(self.audio_layer(x_a.view(x_a.shape[0], -1)))
            video_features = F.relu(self.video_layer(x_v.view(x_v.shape[0], -1)))
            # print(audio_features.shape, video_features.shape)

            attended_audio = self.video_audio_attention(video_features, audio_features)
            attended_video = self.audio_video_attention(audio_features, video_features)
            text_attended_audio = self.text_audio_attention(text_features, attended_audio)
            text_attended_video = self.text_video_attention(text_features, attended_video)
            combined_features = text_features + text_attended_audio + text_attended_video
        else:
            combined_features = text_features

        hidden = self.linear(combined_features)
        output = self.regression(hidden)

        return output

