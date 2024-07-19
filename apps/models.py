import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)
# 抄作业 https://github.com/Mmmofan/dlsyscourse-hw/blob/main/hw4/apps/models.py


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        convBN1 = nn.ConvBN(3, 16, 7, 4, device=device, dtype=dtype)
        convBN2 = nn.ConvBN(16, 32, 3, 2, device=device, dtype=dtype)
        # 中间价格 res 网络
        res1 = nn.Residual(
            nn.Sequential(
                nn.ConvBN(32, 32, 3, 1, device=device, dtype=dtype),
                nn.ConvBN(32, 32, 3, 1, device=device, dtype=dtype)
            )
        )
        convBN3 = nn.ConvBN(32, 64, 3, 2, device=device, dtype=dtype)
        convBN4 = nn.ConvBN(64, 128, 3, 2, device=device, dtype=dtype)
        res2 = nn.Residual(
            nn.Sequential(
                nn.ConvBN(128, 128, 3, 1, device=device, dtype=dtype),
                nn.ConvBN(128, 128, 3, 1, device=device, dtype=dtype)
            )
        )
        # flatten
        flatten = nn.Flatten()
        # 特征提取
        self.feature = nn.Sequential(
            convBN1,
            convBN2,
            res1,
            convBN3,
            convBN4,
            res2,
            flatten
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 128, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(128, 10, device=device, dtype=dtype)
        )
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        x = self.feature(x)
        x = self.fc(x)
        return x
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        seq_model
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        self.seq_model_name = seq_model
        if seq_model == 'rnn':
            self.seq_model = nn.RNN(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        elif seq_model == 'lstm':
            self.seq_model = nn.LSTM(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        else:
            raise ValueError("seq_model must be 'rnn' or 'lstm'")
        # outputs probabilities of the next word
        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
    
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        embedding = self.embedding(x) # (seq_len, bs, embedding_dim)
        out, h_out = self.seq_model(embedding, h)
        out = self.linear(out.reshape((-1, self.hidden_size)))
        return out, h_out
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)