import torch
import torch.nn as nn
from model import TinyLM

# Hiperparameters
batch_size = 32
seq_len = 32
eval_interval = 100
lr = 1e-3
epochs = 5000
dim_embedding = 128
num_layers = 2
dropout = 0.1
patience = 300
device = "cuda" if torch.cuda.is_available() else "cpu"


# Carregar o dataset
file_path = 'dataset/input.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Vocabulário
characteres = sorted(list(set(text)))
vocab_size = len(characteres)
tamanho_vocabulario = len(characteres)
stoi = {ch: i for i, ch in enumerate(characteres)}
itos = {i: ch for i, ch in enumerate(characteres)}

def codificar(s):
    return [stoi[c] for c in s]

def decodificar(l):
    return ''.join([itos[i] for i in l])

data = torch.tensor(codificar(text), dtype=torch.long)

if len(data) < batch_size:
    raise ValueError("O tamanho do dataset é menor que o tamanho do batch_size. Ajuste o dataset ou o batch_size.")
train_size = int(0.9 * len(data))
train_data = data[:train_size]
val_data = data[train_size:]

# Function to gen batchs of data : This fuction prepares batches of data for training
def get_batch(data, batch_size, seq_len):
    start_indices = []
    max_start = len(data) - seq_len - 1
    for _ in range(batch_size):
        idx = torch.randint(0, max_start, (1,)).item()
        start_indices.append(idx)

    # Lista para armazenar os pares de entrada (x) e saida (y)
    x_list = []
    y_list = []

    # Para cada índice sorteado, gera as sequências de entrada e alvo
    for start in start_indices:
        x_seq = data[start:start + seq_len]
        y_seq = data[start + 1:start + seq_len + 1]

        x_list.append(x_seq)
        y_list.append(y_seq)

    # Empilha todas as seq. em tensores de shape (batch_size, seq_len)
    x_batch = torch.stack(x_list)
    y_batch = torch.stack(y_list)

    return x_batch, y_batch

def evaluate_model(model, data, batch_size, seq_len):
    """Avalia o modelo no conjunto de validação"""
    model.eval()
    total_loss = 0
    num_batches = 10  # Número limitado de batches para validação
    
    with torch.no_grad():
        for _ in range(num_batches):
            try:
                x_batch, y_batch = get_batch(data, batch_size, seq_len)
                logits, _ = model(x_batch)
                loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), y_batch.view(-1))
                total_loss += loss.item()
            except:
                break
    
    return total_loss / num_batches if num_batches > 0 else float('inf')


# Instancia o modelo e otimização
model = TinyLM(vocab_size=vocab_size, dim=128, num_layers=2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
criterion = nn.CrossEntropyLoss()

# Treinamento
best_val_loss = float('inf')
patience = 300
patience_counter = 0
val_loss = float('inf')  # Inicializa val_loss para evitar NameError

for epoch in range(epochs):
    model.train()
    
    try:
        x_batch, y_batch = get_batch(train_data, batch_size, seq_len)
        
        # Forward pass
        logits, _ = model(x_batch)
        loss = criterion(logits.view(-1, vocab_size), y_batch.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Logging
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Validação
        if (epoch + 1) % eval_interval == 0:
            val_loss = evaluate_model(model, val_data, batch_size, seq_len)
            print(f"Validation Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Salvar melhor modelo
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'vocab_size': vocab_size,
                    'dim': 128,
                    'num_layers': 2,
                    'epoch': epoch,
                    'loss': val_loss
                }, "./weight/best_tiny_lm.pth")
            else:
                patience_counter += eval_interval
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
    except Exception as e:
        print(f"Erro no epoch {epoch+1}: {e}")
        break

# Salvar modelo final
# Implementar a verificação se os pesos já foram criados

torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'dim': 128,
    'num_layers': 2,
    'epoch': epochs,
    'loss': val_loss
}, "./weight/tiny_lm_final.pth")

print("Treinamento concluído!")
print(f"Melhor validation loss: {best_val_loss:.4f}")