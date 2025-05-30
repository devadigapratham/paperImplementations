# 🧠 paperImplementations

A collection of minimal and readable implementations of research papers I read.  
The goal is to better understand complex architectures by translating them into simple, self-contained Python modules.

## 📂 Current Implementation

### `llama4/`
Lightweight reimplementation of core components inspired by the LLaMA 4 architecture:
- `attention.py` – Multi-head attention mechanism
- `feedforward.py` – Feedforward neural network block
- `tokenizer.py` – Basic tokenizer for preprocessing input text

These modules are intended as learning tools and are not optimized for production use.

## 🛠️ Setup
```bash
# Clone the repo
git clone https://github.com/your-username/paperImplementations.git
cd paperImplementations

# (Optional) Create a virtual environment
python3 -m venv venv
source venv/bin/activate
