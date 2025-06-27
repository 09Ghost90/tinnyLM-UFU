import torch
from model import TinyLM

checkpoint = torch.load('weight/best_tiny_lm.pth', map_location='cpu')

# Parametros
vocab_size = checkpoint['vocab_size']
dim = checkpoint['dim']
num_layers = checkpoint['num_layers']

# Instanciar e carregar os pesos já salvos
model = TinyLM(vocab_size=vocab_size, dim=dim, num_layers=num_layers)
model.load_state_dict(torch.load('weight/best_tiny_lm.pth', map_location='cpu')['model_state_dict'])
model.eval()

# Dummy input para exportar o modelo
seq_len = 32
dummy_input = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long) # Batch size 1, sequence length 32)

# Exportar para ONNX
torch.onnx.export(
    model,
    dummy_input,
    "tiny_lm.onnx",
    input_names=["input"],
    output_names=["output", "hidden"],
    dynamic_axes={"input": {1: "seq_len"}},
    opset_version=11
)

print("Modelo exportado para ONNX com sucesso.")
