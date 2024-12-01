"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, n_class, device):
        super(GNN, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.fc4 = nn.Linear(hidden_dim_3, n_class)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj, idx):
        
        ############## Task 6

        A_hat = adj + torch.eye(adj.size(0)).to(self.device).to_sparse()
        z_1 = self.fc1(torch.mm(A_hat, x_in))
        z_1 = self.relu(z_1)

        x = self.fc2(torch.mm(A_hat, z_1))

        idx.to(self.device)
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx)+1, x.size(1), device=self.device)
        out = out.scatter_add_(0, idx, x)

        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        
        return F.log_softmax(out, dim=1)
