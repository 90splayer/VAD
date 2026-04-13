import torch
import torch.nn as nn

class AttentionMIL(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x):
        attn = torch.softmax(self.attention(x), dim=1)
        scores = self.classifier(x).squeeze(-1)
        return scores, attn
    

  

   # def forward(self, x):

        # x shape: (B, T, D)

        features = self.feature_extractor(x)

        attention = self.attention_layer(features)  # (B, T, 1)
        attention = torch.softmax(attention, dim=1)

        scores = self.classifier(features)  # (B, T, 1)

        # 🔥 YOUR IMPROVEMENT HERE
        scores = scores * attention

        return scores.squeeze(-1), attention