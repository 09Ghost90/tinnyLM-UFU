import torch
import random
import numpy as np
from model import TinyLM

device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

file_path = 'dataset/input.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

characteres = sorted(list(set(text)))
vocab_size = len(characteres)
stoi = {ch: i for i, ch in enumerate(characteres)}
itos_list = [ch for ch in characteres]

# Carregamento do modelo
try:
    checkpoint = torch.load("weight/best_tiny_lm.pth", map_location=device)
    model = TinyLM(
        vocab_size=checkpoint['vocab_size'], 
        dim=checkpoint['dim'], 
        num_layers=checkpoint['num_layers']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Modelo carregado (epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f})")
    
except FileNotFoundError:
    print("Modelo não encontrado. Execute o treinamento primeiro.")
    exit(1)

def generate_text(model, prompt, max_tokens=100, temperature=0.8, top_k=40):
    """
    Gera texto usando o modelo treinado
    
    Args:
        model: Modelo treinado
        prompt: Texto inicial
        max_tokens: Número máximo de tokens a gerar
        temperature: Controla aleatoriedade (menor = mais determinístico)
        top_k: Considera apenas os k tokens mais prováveis
    """

    # Converte prompt para índices
    input_indices = []
    for c in prompt:
        if c in stoi:
            input_indices.append(stoi[c])
        else:
            print(f"Caractere '{c}' não encontrado no vocabulário, usando espaço")
            input_indices.append(stoi.get(' ', 0))
    
    if not input_indices:
        input_indices = [stoi.get(' ', 0)]
    
    # Tensor de entrada
    input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)
    generated = input_indices.copy()
    
    # Estado oculto inicial
    hidden = model.init_hidden(1, device)
    
    # Processa o prompt primeiro para "aquecer" o estado
    with torch.no_grad():
        if len(input_indices) > 1:
            prompt_output, hidden = model(input_tensor, hidden)
    
    print(f"Gerando texto... (temperatura: {temperature}, top_k: {top_k})")
    
    # Geração token por token
    for i in range(max_tokens):
        with torch.no_grad():
            # Usa apenas o último token para a próxima previsão
            last_token = torch.tensor([[generated[-1]]], dtype=torch.long).to(device)
            output, hidden = model(last_token, hidden)
            
            # Logits do último (e único) token
            next_logits = output[0, -1]
            
            # Aplica temperatura
            next_logits = next_logits / temperature
            
            # Top-k sampling
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_logits, min(top_k, len(next_logits)))
                # Zera logits que não estão no top-k
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits[top_k_indices] = top_k_logits
            
            # Converte para probabilidades
            probs = torch.softmax(next_logits, dim=-1)
            
            # Amostragem
            next_id = torch.multinomial(probs, num_samples=1).item()

            # Adiciona à sequência gerada
            generated.append(next_id)
            
            # Index -> Char
            if 0 <= next_id < vocab_size:
                next_char = itos_list[next_id]
            else:
                next_char = '?'

            print(next_char, end='', flush=True)
            if next_char in ['\n\n', '<END>', '<|endoftext|>']:
                break
    
    print()
    
    result = ''.join([itos_list[i] if i < len(itos_list) else '?' for i in generated])
    return result

def interactive_generation():
    """Modo interativo para geração de texto"""
    print("\n=== Gerador de Texto Interativo ===")
    print("Digite 'quit' para sair")
    
    while True:
        prompt = input("\nPrompt: ")
        if prompt.lower() == 'quit':
            break
        
        if not prompt.strip():
            prompt = "First Citizen:"
        
        # Parâmetros configuráveis
        try:
            max_tokens = int(input("Máximo de tokens (padrão 50): ") or "50")
            temperature = float(input("Temperatura 0.1-2.0 (padrão 0.8): ") or "0.8")
            top_k = int(input("Top-k (padrão 40): ") or "40")
        except ValueError:
            max_tokens = 50
            temperature = 0.8
            top_k = 40
        
        print(f"\nTexto gerado:")
        print("-" * 50)
        print(prompt, end='')
        
        result = generate_text(
            model, 
            prompt, 
            max_tokens=max_tokens, 
            temperature=temperature, 
            top_k=top_k
        )
        
        print("-" * 50)

if __name__ == "__main__":
    # Teste rápido
    prompt = "First Citizen:"
    print(f"Teste com prompt: '{prompt}'")
    print("=" * 60)
    print(prompt, end='')
    
    result = generate_text(model, prompt, max_tokens=100, temperature=0.8, top_k=40)
    print("=" * 60)
    
    interactive_generation()