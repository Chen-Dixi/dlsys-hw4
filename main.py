import sys
sys.path.append('./python')
import itertools
import numpy as np
import torch
import needle as ndl
from needle import backend_ndarray as nd

SEQ_LENGTHS = [1, 13]
NUM_LAYERS = [1, 2]

BATCH_SIZES = [1, 15]
INPUT_SIZES = [1, 11]
HIDDEN_SIZES = [1, 12]
BIAS = [True, False]
INIT_HIDDEN = [True, False]
NONLINEARITIES = ['tanh', 'relu']

np.random.seed(3)

def test_lstm_cell():
    batch_size = BATCH_SIZES[0]
    input_size = INPUT_SIZES[1]
    hidden_size = HIDDEN_SIZES[1]
    bias = True
    init_hidden = False
    x = np.random.randn(batch_size, input_size).astype(np.float32)
    h0 = np.random.randn(batch_size, hidden_size).astype(np.float32)
    c0 = np.random.randn(batch_size, hidden_size).astype(np.float32)

    torch_lstm_cell = torch.nn.LSTMCell(input_size, hidden_size, bias=bias)
    h_, c_ = torch_lstm_cell(torch.tensor(x), None)
    print(h_, c_)
    print("========")
    device = ndl.cpu_numpy()
    model = ndl.nn.LSTMCell(input_size, hidden_size, bias=bias, device=device)
    model.W_ih = ndl.Tensor(torch_lstm_cell.weight_ih.detach().numpy().transpose(), device=device)
    model.W_hh = ndl.Tensor(torch_lstm_cell.weight_hh.detach().numpy().transpose(), device=device)
    if bias:
        model.bias_ih = ndl.Tensor(torch_lstm_cell.bias_ih.detach().numpy(), device=device)
        model.bias_hh = ndl.Tensor(torch_lstm_cell.bias_hh.detach().numpy(), device=device)
    h, c = model(ndl.Tensor(x, device=device), None)
    print(h, c)

def test_lstm():
    seq_length, num_layers = SEQ_LENGTHS[1], NUM_LAYERS[1]
    batch_size = BATCH_SIZES[0]
    input_size = INPUT_SIZES[1]
    hidden_size = HIDDEN_SIZES[0]
    bias = True
    init_hidden = False
    device = ndl.cpu_numpy()

    x = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)
    h0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)
    c0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)

    torch_model = torch.nn.LSTM(input_size, hidden_size, bias=bias, num_layers=num_layers)

    needle_model = ndl.nn.LSTM(input_size, hidden_size, num_layers=num_layers, bias=bias, device=device)
    for k in range(num_layers):
        needle_model.lstm_cells[k].W_ih = ndl.Tensor(getattr(torch_model, f'weight_ih_l{k}').detach().numpy().transpose(), device=device)
        needle_model.lstm_cells[k].W_hh = ndl.Tensor(getattr(torch_model, f'weight_hh_l{k}').detach().numpy().transpose(), device=device)
        if bias:
            needle_model.lstm_cells[k].bias_ih = ndl.Tensor(getattr(torch_model, f'bias_ih_l{k}').detach().numpy(), device=device)
            needle_model.lstm_cells[k].bias_hh = ndl.Tensor(getattr(torch_model, f'bias_hh_l{k}').detach().numpy(), device=device)
    
    # 不传 hidden_state
    # torch 模型输出
    output_, (h_, c_) = torch_model(torch.tensor(x), None)
    output, (h, c) = needle_model(ndl.Tensor(x, device=device), None)
    print(output_)
    print(output)

if __name__ == '__main__':
    # test_tanh_backward()
    # cifar10_dataset()
    # cifar10_dataloader()
    test_lstm()