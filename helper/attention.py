import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __ini__(self, dim_in, num_layers=4, num_heads=8):
        super(Attention, self).__init__()
        self.num_layers = num_layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=dim_in, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.fc_layers = nn.ModuleList([
            nn.Linear(dim_in, dim_in)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        residual = x
        for i in range(self.num_layers):
            x, _ = self.attention_layers[i](x, x, x)
            x = F.relu(x)
            x += residual
            residual = x
        
        return x