from flask import Flask, render_template, request, session, redirect, url_for
import torch
import os
from model import GPTLanguageModel

# Hyperparameters (must match training config)
block_size = 128
n_embd = 384
n_head = 1
n_layer = 1
dropout = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load vocab
with open("openwebtext/vocab.txt", "r", encoding="utf-8") as f:
    chars = list(f.read())
string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s if c in string_to_int]
decode = lambda l: ''.join([int_to_string[i] for i in l])

vocab_size = len(chars)

# Initialize model
model = GPTLanguageModel(vocab_size, block_size, n_embd, n_head, n_layer, dropout).to(device)
model.load_state_dict(torch.load('model-openwebtext.pt', map_location=device))
model.eval()

# Flask app
app = Flask(__name__)
app.secret_key = "secret_key_here"  # Replace with a secure key in production

@app.route('/', methods=['GET'])
def index():
    session.setdefault('chat_history', [])
    return render_template('index.html', chat_history=session['chat_history'])

@app.route('/chat', methods=['POST'])
def chat():
    prompt = request.form['prompt']
    input_ids = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    output_ids = model.generate(input_ids, max_new_tokens=100)[0].tolist()
    response = decode(output_ids)[len(prompt):]

    # Store in session
    session['chat_history'].append(('user', prompt))
    session['chat_history'].append(('bot', response))
    session.modified = True

    return redirect(url_for('index'))

@app.route('/reset', methods=['POST'])
def reset_chat():
    session['chat_history'] = []
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
