import torch.nn as nn

class MLPProjection(nn.Module):
    """
    A configurable projection layer module that maps input embeddings to a target dimension.
    """
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.projection(x)
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)  # L2 normalize
        return x
    

# class MLPProjection(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim, nonlin=nn.ReLU(), dropout=0.1):        
#         super().__init__()
#         self.nonlin = nonlin
#         self.dropout = dropout

#         sequence = []
#         units = [input_dim, hidden_dim, output_dim]
#         for u0, u1 in zip(units[:-1], units[1:]):
#             sequence.append(nn.Linear(u0, u1))
#             sequence.append(self.nonlin)
#             sequence.append(nn.Dropout(self.dropout))
#         sequence = sequence[:-2]

#         self.sequential = nn.Sequential(*sequence)

#     def forward(self, X):
#         X = self.sequential(X)
#         return X
