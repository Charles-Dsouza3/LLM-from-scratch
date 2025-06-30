# 🧠 GPT Chatbot from Scratch (LLM + UI)

This is a fully functional GPT-style language model built from scratch using PyTorch and trained on the OpenWebText dataset. It includes a clean web-based chat interface built with Flask.

## 🚀 Features

- Train a Transformer-based language model from scratch
- Uses OpenWebText dataset (1% subset)
- Implements multi-head attention, feedforward layers, and positional embeddings
- Chat with your model via a sleek HTML interface (Flask + Jinja2)
- Maintains chat history and allows resetting the conversation

### 🏗️ Model Architecture

- Character-level language modeling
- Token + position embeddings
- Multi-head masked self-attention
- LayerNorm + Feedforward network
- Single or multiple transformer blocks
- Cross-entropy loss during training
- Greedy + multinomial sampling for generation

Implemented in `model.py` using only PyTorch modules.

### 📄 Dataset

- Uses [Skylion007/OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) from HuggingFace Datasets
- 1% slice used by default for lightweight training
- Auto-generated character vocabulary stored in `openwebtext/vocab.txt`

## 🛠️ How to Run the Project

## 🔧 Step 1: Train the Model
```bash
python main.py -batch_size 32
```
# What this does:
- Loads and preprocesses the dataset
- Creates vocab.txt inside openwebtext/
- Trains the model for 200 iterations (can be modified)
- Saves the model as model-openwebtext.pt

## 🌐 Step 2: Launch the Web Chatbot
```bash
python app.py
```
Then open http://localhost:5000 in your browser.

#UI Features:
- Type your question and press Send
- Chat history scrolls and shows user/bot messages
- Click Reset to clear the session

## 📦 Requirements

```bash
pip install torch datasets flask
