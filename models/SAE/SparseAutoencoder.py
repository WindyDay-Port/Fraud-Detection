import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAE(nn.Module):
    def __init__(self, sparsity_param=0.08, beta=0.004):
        super(SparseAE, self).__init__()
        self.sparsity_param = sparsity_param # p hat
        self.beta = beta # To adjust the sparsity penalty

        # Starting point consisting of:
        # 1 input layer
        # 2 hidden layer
        # Modification capacity in exchange of interpretability
        
        self.encoder = nn.Sequential(
            nn.Linear(49, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 49),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded

def total_cost_function(decoded, input_data, encoded, beta=0.1, rho=0.05):
    mse_loss = F.mse_loss(decoded, input_data)

    epsilon = 1e-8
    rho_hat = torch.mean(encoded, dim=0)
    kl_divergence = torch.sum(
        rho * torch.log(rho / (rho_hat + epsilon)) + 
        (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + epsilon))
    )
    return mse_loss, kl_divergence, mse_loss + beta * kl_divergence
    