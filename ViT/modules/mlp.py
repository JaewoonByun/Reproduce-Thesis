import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rto=0.5, device='cpu'):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, device=device)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rto)
        self.fc2 = nn.Linear(hidden_dim, out_dim, device=device)
        self.dropout2 = nn.Dropout(dropout_rto)
        
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x