import torch
from typing import Callable, List, Tuple
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from torch.cuda.amp import autocast
from src.config import PRE_RESIDUAL_ADDITION_KEY
import numpy as np
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset

TransformerModel = Tuple[AutoModelForCausalLM, AutoTokenizer]

def load_model(model_name: str,
               device,
               quantization=None) -> TransformerModel:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = None
    if quantization is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name)
    elif quantization == '4bit':
        qc = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=qc, device_map=device)#, torch_dtype=torch.float32)

    elif quantization == '8bit':
        qc = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=qc, device_map=device)#, torch_dtype=torch.float32)
    else:
        raise ValueError("Expected None, 4bit, or 8bit")
    return (model, tokenizer)


def forward_hooked_model(model: TransformerModel, inp: List[str]) -> Tuple[torch.tensor, List[torch.tensor]]:
    model, tokenizer = model
    with torch.no_grad():
        inps = tokenizer(inp, return_tensors='pt').to(model.device)
        r = model.forward(**inps, output_hidden_states=True, use_cache=True)
        
        return r['logits'], r[PRE_RESIDUAL_ADDITION_KEY]

class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, sparsity_param=0.05, beta=2, lamda=0.0001):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.sparsity_param = sparsity_param
        self.beta = beta
        self.lamda = lamda

        # Initialize weights
        nn.init.xavier_normal_(self.encoder.weight)
        nn.init.xavier_normal_(self.decoder.weight)
        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return encoded, decoded

    def kl_divergence(self, p, q):
        q = torch.clamp(q, 1e-5, 1-1e-5)  # Ensure q is within (1e-10, 1 - 1e-10)
        return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))

    def loss_function(self, x, decoded, encoded):
        # print("AA", decoded.shape, x.shape)
        mse_loss = self.loss(decoded, x)
        rho_hat = torch.mean(encoded, dim=0)
        kl_div = self.kl_divergence(self.sparsity_param, rho_hat).sum()
        l2_reg = sum(torch.norm(param) for param in self.parameters())
        loss = mse_loss + self.beta * kl_div + self.lamda * l2_reg
        # print("AAA", kl_div, loss)
        return loss


def train_autoencoder(autoencoder, dataset, num_epochs=50, batch_size=64, learning_rate=1e-4):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    clip_value = 5

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in data_loader:
            optimizer.zero_grad()
            batch = batch[0]
            encoded, decoded = autoencoder(batch)
            loss = autoencoder.loss_function(batch, decoded, encoded)
            if torch.isnan(loss):
                print("Loss became NaN. Stopping training.")
                return
            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), clip_value)
            optimizer.step()
            epoch_loss += loss.item()

        average_loss = epoch_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")
