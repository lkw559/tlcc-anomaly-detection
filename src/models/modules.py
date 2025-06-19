import torch
import torch.nn as nn
import pandas as pd
import numpy as np
################# GCN ######################
import torch.nn.functional as F
#from torch_geometric.nn import GCNConv
############################################

class ConvLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation
    """

    def __init__(self, n_features, kernel_size=7):
        super(ConvLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        #print('\n\n\n1. ConvLayer x value\n\t', x.shape)
        #print('2. ConvLayer return value\n\t', x.permute(0, 2, 1).shape)
        #print("x type:", type(x))
        #print('ConvLayer x:\n', x)
        return x.permute(0, 2, 1)  # Permute back
    

######################## GCN #################################
class WeightLayer(torch.nn.Module):
    def __init__(self, n_features, window_size, corr_adj_np, use_bias=True):
        super(WeightLayer, self).__init__()
        self.w = nn.Parameter(torch.empty((n_features, n_features)))
        # 인접행렬을 텐서로 변환하여 저장
        self.corr_adj_tensor = torch.FloatTensor(corr_adj_np)
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(window_size, n_features))
            # Initialize bias parameters to zero for stability
            nn.init.zeros_(self.bias.data)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 인접행렬 적용 (상관관계 정보 반영)
        device = x.device
        corr_adj = self.corr_adj_tensor.to(device)
        x = x.matmul(corr_adj)  # 인접행렬 적용
        x = x.matmul(self.w)    # 학습 가능한 가중치 적용
        if self.use_bias:
            x += self.bias        
        x = self.relu(x)
        return x
    
##############################################################


class FeatureAttentionLayer(nn.Module):
    """Single Graph Feature/Spatial Attention Layer
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer
    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):
        super(FeatureAttentionLayer, self).__init__()
        self.n_features = n_features
        #print('3. FeatureAttentionLayer n_features:\n\t', self.n_features)
        self.window_size = window_size
        #print('4. FeatureAttentionLayer window_size:\n\t', self.window_size)
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else window_size
        #print('5. FeatureAttentionLayer embed_dim:\n\t', self.embed_dim)
        self.use_gatv2 = use_gatv2
        self.num_nodes = n_features
        #print('6. FeatureAttentionLayer num_nodes:\n\t', self.num_nodes)
        self.use_bias = use_bias

        # Because linear transformation is done after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * window_size
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = window_size
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(n_features, n_features))
            # Initialize bias parameters to zero for stability
            nn.init.zeros_(self.bias.data)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For feature attention we represent a node as the values of a particular feature across all timestamps

        x = x.permute(0, 2, 1)
        #print('7. FeatureAttentionLayer x\n\t', x.shape)

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)                 # (b, k, k, 2*window_size)
            a_input = self.leakyrelu(self.lin(a_input))             # (b, k, k, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)            # (b, k, k, 1)
            #print('8. FeatureAttentionLayer e value \n\t', e.shape)

        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (b, k, k, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, k, k, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, k, k, 1)
            #print('9. FeatureAttentionLayer e value \n\t', e.shape)

        if self.use_bias:
            e += self.bias
            #print('10. FeatureAttentionLayer e value \n\t', e.shape)

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)
        #print('11. FeatureAttentionLayer attention weights\n\t', attention.shape)

        # Computing new node features using the attention
        h = self.sigmoid(torch.matmul(attention, x))
        #print('12. FeatureAttentionLaye h value\n\t', h.shape)
        #print('13. FeatureAttentionLaye return value\n\t', h.permute(0, 2, 1).shape)

        return h.permute(0, 2, 1)

    def _make_attention_input(self, v):
        """Preparing the feature attention mechanism.
        Creating matrix with all possible combinations of concatenations of node.
        Each node consists of all values of that node within the window
            v1 || v1,
            ...
            v1 || vK,
            v2 || v1,
            ...
            v2 || vK,
            ...
            ...
            vK || v1,
            ...
            vK || vK,
        """

        K = self.num_nodes
        #print('14. _make_attention_input K value\n\t', K)
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)  # (b, K*K, 2*window_size)
        #print('15. _make_attention_input combined value\n\t', combined.shape)

        if self.use_gatv2:
            #print('16. _make_attention_input return value\n\t', combined.view(v.size(0), K, K, 2 * self.window_size).shape)
            return combined.view(v.size(0), K, K, 2 * self.window_size)
        else:
            #print('17. _make_attention_input return value\n\t', combined.view(v.size(0), K, K, 2 * self.embed_dim).shape)
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class TemporalAttentionLayer(nn.Module):
    """Single Graph Temporal Attention Layer
    :param n_features: number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer

    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):
        super(TemporalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        self.embed_dim = embed_dim if embed_dim is not None else n_features
        self.num_nodes = window_size
        self.use_bias = use_bias

        # Because linear transformation is performed after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * n_features
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = n_features
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(window_size, window_size))
            # Initialize bias parameters to zero for stability
            nn.init.zeros_(self.bias.data)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For temporal attention a node is represented as all feature values at a specific timestamp

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        #print('18. TemporalAttentionLayer x value\n\t', x.shape)
        if self.use_gatv2:
            a_input = self._make_attention_input(x)              # (b, n, n, 2*n_features)
            a_input = self.leakyrelu(self.lin(a_input))          # (b, n, n, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)         # (b, n, n, 1)
            #print('19. TemporalAttentionLayer e value\n\t', e.shape)

        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (b, n, n, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, n, n, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, n, n, 1)
            #print('20. TemporalAttentionLayer e value\n\t', e.shape)

        if self.use_bias:
            e += self.bias  # (b, n, n, 1)
            #print('21. TemporalAttentionLayer e value\n\t', e.shape)

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)
        #print('22. TemporalAttentionLayer attention weight\n\t', attention.shape)

        h = self.sigmoid(torch.matmul(attention, x))    # (b, n, k)
        #print('23. TemporalAttentionLayer h value\n\t', h.shape)

        return h

    def _make_attention_input(self, v):
        """Preparing the temporal attention mechanism.
        Creating matrix with all possible combinations of concatenations of node values:
            (v1, v2..)_t1 || (v1, v2..)_t1
            (v1, v2..)_t1 || (v1, v2..)_t2

            ...
            ...

            (v1, v2..)_tn || (v1, v2..)_t1
            (v1, v2..)_tn || (v1, v2..)_t2

        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)
        #print('24. _make_attention_input combined value\n\t', combined.shape)

        if self.use_gatv2:
            #print('25. _make_attention_input combined value\n\t', combined.view(v.size(0), K, K, 2 * self.n_features).shape)
            return combined.view(v.size(0), K, K, 2 * self.n_features)
        else:
            #print('26. _make_attention_input combined value\n\t', combined.view(v.size(0), K, K, 2 * self.embed_dim).shape)
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class GRULayer(nn.Module):
    """Gated Recurrent Unit (GRU) Layer
    :param in_dim: number of input features
    :param hid_dim: hidden size of the GRU
    :param n_layers: number of layers in GRU
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(GRULayer, self).__init__()
        self.hid_dim = hid_dim
        #print('27. GRULayer in_dim:\n\t', in_dim)
        #print('28. GRULayer hid_dim:\n\t', hid_dim)
        self.n_layers = n_layers
        #print('29. GRULayer n_layers:\n\t', n_layers)
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        #print('30. GRU x value\n\t', x.shape)
        out, h = self.gru(x)
        #print('31. GRU h value\n\t', h.shape)
        out, h = out[-1, :, :], h[-1, :, :]  # Extracting from last layer
        #print('32. GRU h value\n\t', h.shape)
        return out, h


class RNNDecoder(nn.Module):
    """GRU-based Decoder network that converts latent vector into output
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        #print('33. RNNDecoder x value\n\t', x.shape)
        decoder_out, _ = self.rnn(x)
        #print('34. RNNDecoder decoder out value\n\t', decoder_out.shape)
        return decoder_out


class ReconstructionModel(nn.Module):
    """Reconstruction Model
    :param window_size: length of the input sequence
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param in_dim: number of output features
    :param dropout: dropout rate
    """

    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconstructionModel, self).__init__()
        self.window_size = window_size
        self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        # x will be last hidden state of the GRU layer
        h_end = x
        #print('35. ReconstructionModel h_end value\n\t', h_end.shape)
        h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)
        #print('36. ReconstructionModel h_end_rep\n\t', h_end_rep.shape)

        decoder_out = self.decoder(h_end_rep)
        #print('37. ReconstructionModel decoder_out\n\t', decoder_out.shape)
        out = self.fc(decoder_out)
        #print('38. ReconstructionModel out\n\t', out.shape)
        return out


class Forecasting_Model(nn.Module):
    """Forecasting model (fully-connected network)
    :param in_dim: number of input features
    :param hid_dim: hidden size of the FC network
    :param out_dim: number of output features
    :param n_layers: number of FC layers
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        #print('Forecasting_Model hid_dim\n\t', hid_dim)
        super(Forecasting_Model, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        #print('39. Forecasting_Model x value\n\t', x.shape)
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)
        #print('40. Forecasting_Model x\n\t', self.layers[-1](x).shape)
        return self.layers[-1](x)
