import os
import datetime
import torch
import argparse
from transformer_decoder import LanguageModel

block_size = 64
batch_size = 64
seq_len = 10
learning_rate = 0.003
epochs = 10000
n_emd_dim = 64
n_layer = 6
n_head = 6
drop_rate = 0.2


def load_input(input_filepath):
    with open(input_filepath, 'r') as f:
        text = f.read()
    return text


def get_input_params(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    encode = lambda x: [char_to_idx[ch] for ch in x]
    decode = lambda x: ''.join([idx_to_char[i] for i in x])

    return vocab_size, char_to_idx, idx_to_char, encode, decode


def get_data(text, encode, pct_train=0.9):
    data = torch.tensor(encode(text))

    start_test = pct_train * len(data)
    train_data = data[:int(start_test)]
    test_data = data[int(start_test):]

    return train_data, test_data


def get_batch(split, train_data, test_data):
    data = train_data if split == 'train' else test_data
    start_indices = torch.randint(0, data.size(0) - block_size, (batch_size,))
    x = torch.stack([data[start_idx:start_idx + block_size] for start_idx in start_indices])
    y = torch.stack([data[start_idx + 1:start_idx + block_size + 1] for start_idx in start_indices])
    return x, y


def train_model(train_data, test_data, vocab_size, n_emd_dim, block_size, n_layer, n_head, drop_rate, epochs):

    model = LanguageModel(vocab_size=vocab_size, n_emd_dim=n_emd_dim, block_size=block_size, n_layer=n_layer,
                          n_head=n_head, drop_rate=drop_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in range(epochs):
        x, y = get_batch('train', train_data, test_data)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print(i, loss)

    now_timetsamp = datetime.datetime.now().timestamp()
    torch.save(model, f'model/trained_model_{now_timetsamp}.pth')
    return model


def generate_new_script(trained_model, decode):
    context = torch.zeros((1, 1), dtype=torch.long)
    return decode(trained_model.generate(context, max_tokens=10000)[0].tolist())

def save_output_script(new_script):
    timestamp = datetime.datetime.now().timestamp()
    with open(f'output/new_script_{timestamp}.txt', 'w') as f:
        f.write(new_script)

def main(input_filepath):
    """
    Load the input script, train the model, generate a new script, and save it to a file.
    Args:
        input_filepath (str): Path to the input script.
    """
    text = load_input(input_filepath)
    vocab_size, char_to_idx, idx_to_char, encode, decode = get_input_params(text)
    train_data, test_data = get_data(text, encode)

    trained_model = train_model(train_data, test_data, vocab_size, n_emd_dim, block_size, n_layer, n_head, drop_rate, epochs)
    new_script = generate_new_script(trained_model, decode)
    save_output_script(new_script)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input_filepath', type=str, default='data/script.txt')
    args = arg_parser.parse_args()
    main(args.input_filepath)
