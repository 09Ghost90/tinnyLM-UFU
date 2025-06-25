import torch
import torch.nn as nn

# vocab_size is the number of tokens model can preview
# dim is the dimensionality of vector representations

class TinyLM(nn.Module):
    def __init__(self, vocab_size=256, dim=64, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, dim) # transform indices integers of tokens to vectors
        self.rnn = nn.GRU(dim, dim, batch_first=True, num_layers=self.num_layers) 
        self.layer_norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, vocab_size)
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)

        x = self.embed(x) # Converte token indices em vetores (embeddings)
        x = self.dropout(x)

        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        x, hidden = self.rnn(x, hidden)
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Output projection
        x = self.fc(x)
        
        return x, hidden
    
    def init_hidden(self, batch_size, device):
      return torch.zeros(self.num_layers, batch_size, self.dim).to(device)

if __name__ == "__main__":
    model = TinyLM()
    dummy_input = torch.randint(0, 256, (1, 10))
    output, hidden = model(dummy_input)
    print("Output shape:", output.shape)
    print("Hidden shape:", hidden.shape)