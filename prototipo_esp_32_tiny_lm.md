# Pipeline de Protótipo de Modelo TinyLM para ESP32

Este documento apresenta o **pipeline** e a **estrutura do projeto** para desenvolver, quantizar e executar um modelo de geração de texto minúsculo (*TinyLM*) no ESP32, considerando **512 KB de RAM**.

---

## 1. Objetivos

- **Modelo compacto**: ≤ 200 KB de pesos originais.
- **Quantização ternária**: 2 bits por peso.
- **Formato binário**: exportar pesos para `.bin` legíveis em C.
- **Inferência em C**: usar flash para armazenar pesos e RAM para buffers mínimos.
- **Geração de texto**: desempenho alvo de \~10–50 tokens/s.

---

## 2. Requisitos de Ambiente

### 2.1 Host (PC)

- **Python 3.10+**
- **PyTorch**, **NumPy**, **tqdm**
- Editor de texto (VSCode, Vim) e Git

### 2.2 Target (ESP32)

- **ESP-IDF toolchain** (CMake, GCC para Xtensa)
- **ESP32** sem PSRAM (512 KB de RAM interna)
- Flash disponível ≥ 1 MB

---

## 3. Pipeline de Desenvolvimento

1. **Preparação do ambiente**

   - Criar ambiente virtual Python
   - Instalar dependências (PyTorch, NumPy, tqdm)

2. **Construção do modelo TinyLM em PyTorch**

   - Definir vocabulário (e.g. 256 tokens)
   - Implementar arquitetura (`Embedding` + `GRU` + `Linear`)

3. **Treinamento rápido**

   - Preparar corpus de exemplo
   - Treinar por 10–20 épocas até gerar texto coerente

4. **Quantização para ternário**

   - Extrair pesos do modelo treinado
   - Mapear float → {-1, 0, 1}
   - Converter para formato binário (2 bits por peso)

5. **Exportação de pesos**

   - Gerar arquivos `.bin` para cada camada
   - Agregar blobs em um único arquivo, se necessário

6. **Desenvolvimento da inferência em C (ESP-IDF)**

   - Estruturar projeto ESP-IDF
   - Implementar leitura de pesos da flash
   - Escrever funções de embed, GRU simplificada e softmax
   - Gerenciar buffers de entrada/saída na RAM disponível

7. **Integração e testes**

   - Flash do firmware no ESP32
   - Enviar prompt pela serial ou rede
   - Receber e exibir tokens gerados

8. **Otimizações finais**

   - Ajustar limiar de ternarização (tamanho × qualidade)
   - Implementar buffer circular para inferência contínua
   - Documentar resultados de velocidade e uso de memória

---

## 4. Detalhamento Inicial de Cada Etapa

### 4.1 Preparação do ambiente

```bash
python -m venv venv_tiny
source venv_tiny/bin/activate
pip install torch numpy tqdm
```

### 4.2 Arquitetura do TinyLM (`model.py`)

```python
import torch.nn as nn

class TinyLM(nn.Module):
    def __init__(self, vocab_size=256, dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.rnn = nn.GRU(dim, dim, batch_first=True)
        self.fc = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.rnn(x)
        return self.fc(x)
```

### 4.3 Treinamento e Amostragem (`train.py`)

- Carregar dados de texto
- Loop de treinamento simples
- Função de amostragem para geração de texto

### 4.4 Quantização (`quantize.py`)

```python
def to_ternary(tensor):
    # Converte tensor float em sinais {-1, 0, 1}
    return torch.sign(tensor).clamp(-1, 1)

# Exemplo: exportar FC para binário
ternary_weights = to_ternary(model.fc.weight).to(torch.int8).numpy()
ternary_weights.tofile("fc_weights.bin")
```

### 4.5 Projeto ESP32 (ESP-IDF)

```bash
idf.py create-project tiny_lm_esp32
cd tiny_lm_esp32
# Colocar blobs em main/weights/
idf.py menuconfig # ajustar flash partition
idf.py build flash monitor
```

- `main/`: código de inferência em C
- `weights/`: arquivos `.bin` de cada camada

---

## 5. Próximos Passos

1. Validar ambiente Python e treinar modelo mínimo
2. Quantizar e exportar pesos
3. Implementar e testar camada de inferência em C

---

*Documento sujeito a atualizações contínuas.*

