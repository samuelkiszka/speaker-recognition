import torch
import torch.nn as nn

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, input_dim=39, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)  # project 39 → 64
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.input_proj(x)        # project input
        x = x.permute(1, 0, 2)        # (seq_len, batch, d_model)
        out = self.encoder(x)
        return out.mean(dim=0)


_model = SimpleTransformerEncoder(input_dim=39, d_model=64)

def get_embedding(mfcc39):
    """
    mfcc39: (frames, 39) numpy array
    """
    with torch.no_grad():
        x = torch.tensor(mfcc39, dtype=torch.float32).unsqueeze(0)  # (1, frames, 39)
        emb = _model(x)
    return emb.squeeze().numpy()

