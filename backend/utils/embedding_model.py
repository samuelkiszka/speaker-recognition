import torch
import torch.nn as nn

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # (seq_len, batch, feat)
        out = self.encoder(x)
        return out.mean(dim=0)

_model = SimpleTransformerEncoder()

def get_embedding(mfcc):
    with torch.no_grad():
        x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
        emb = _model(x)
    return emb.squeeze().numpy()
