import torch
from torch import nn


class AttentionWithContext(nn.Module):
    """
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    """

    def __init__(self, input_shape, return_coefficients=False, bias=True):
        super(AttentionWithContext, self).__init__()
        self.return_coefficients = return_coefficients

        # input shape: (samples, steps, features)
        self.W = nn.Linear(input_shape, input_shape, bias=bias) # output: (samples, steps, features)
        self.tanh = nn.Tanh()
        self.u = nn.Linear(input_shape, 1, bias=False) # output: (samples, steps, 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.W.weight.data.uniform_(-initrange, initrange)
        self.W.bias.data.uniform_(-initrange, initrange)
        self.u.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz):
        # do not pass the mask to the next layers
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, x, mask=None):
        ########## fill
        uit = self.W(x)  # fill the gap # compute uit = W . x  where x represents ht
        ########## end fill
        uit = self.tanh(uit)
        ait = self.u(uit) # output size: (batch_size, steps, 1)
        a = torch.exp(ait) # output size: (batch_size, steps, 1)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            a = a * mask.double()

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        eps = 1e-9
        a = a / (torch.sum(a, axis=1, keepdim=True) + eps)
        weighted_input = torch.sum(a * x, axis=1)  ### fill the gap ### # compute the attentional vector
        if self.return_coefficients:
            return weighted_input, a  ### [attentional vector, coefficients] ### use torch.sum to compute s
        else:
            return weighted_input  ### attentional vector only ###
        # return a tensor of size (batch_size, features)


class AttentionBiGRU(nn.Module):
    def __init__(self, input_shape, n_units, index_to_word, dropout=0, d=30, drop_rate=0.5):
        super(AttentionBiGRU, self).__init__()
        self.embedding = nn.Embedding(input_shape, # size of the vocab
                                      d,
                                      padding_idx=0)  # fill the gap # vocab size # dimensionality of embedding space
        self.dropout = nn.Dropout(drop_rate)
        self.gru = nn.GRU(input_size=d, # size of the embedding
                          hidden_size=n_units, # number of h_t vectors
                          num_layers=1,
                          bias=True,
                          batch_first=True,
                          bidirectional=True)
        # output_size of the GRU layer: (input_shape) == (batch_size, steps=n_units, features=d)
        self.attention = AttentionWithContext(d,  # fill the gap # the input shape for the attention layer '(samples, steps, features)'
                                              return_coefficients=True)

    def forward(self, sent_ints):
        sent_wv = self.embedding(sent_ints) # (batch_size, steps, d)
        sent_wv_dr = self.dropout(sent_wv) # same

        sent_wa, _ = self.gru(sent_wv_dr) # fill the gap # RNN layer
        sent_att_vec, word_att_coeffs =  self.attention(sent_wa) # fill the gap # attentional vector for the sent
        sent_att_vec_dr = self.dropout(sent_att_vec)
        return sent_att_vec_dr, word_att_coeffs


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size) (448, 30)
        sent_att_vec_dr, word_att_coeffs = self.module(x_reshape)
        # We have to reshape the output
        if self.batch_first:
            sent_att_vec_dr = sent_att_vec_dr.contiguous().view(x.size(0), -1, sent_att_vec_dr.size(-1))  # (samples, timesteps, output_size)
            word_att_coeffs = word_att_coeffs.contiguous().view(x.size(0), -1, word_att_coeffs.size(-1))  # (samples, timesteps, output_size)
        else:
            sent_att_vec_dr = sent_att_vec_dr.view(-1, x.size(1), sent_att_vec_dr.size(-1))  # (timesteps, samples, output_size)
            word_att_coeffs = word_att_coeffs.view(-1, x.size(1), word_att_coeffs.size(-1))  # (timesteps, samples, output_size)
        return sent_att_vec_dr, word_att_coeffs


class HAN(nn.Module):
    def __init__(self, input_shape, n_units, index_to_word, dropout=0, d=30, drop_rate=0.5):
        super(HAN, self).__init__()
        self.encoder = AttentionBiGRU(input_shape, n_units, index_to_word, dropout)
        self.timeDistributed = TimeDistributed(self.encoder, True)
        self.dropout = nn.Dropout(drop_rate)
        self.gru = nn.GRU(input_size=d, # fill the gap # the input shape of GRU layer
                          hidden_size=n_units,
                          num_layers=1,
                          bias=True,
                          batch_first=True,
                          bidirectional=True)
        self.attention = AttentionWithContext(2 * n_units, # fill the gap # the input shape of between-sentence attention layer
                                              return_coefficients=True)
        self.lin_out = nn.Linear(n_units * 2,   # fill the gap # the input size of the last linear layer
                                 1)
        self.preds = nn.Sigmoid()

    def forward(self, doc_ints):
        print(f"Input doc_ints shape: {doc_ints.shape}")
        sent_att_vecs_dr, word_att_coeffs = self.timeDistributed(doc_ints) # fill the gap # get sentence representation
        print(f"sent_att_vecs_dr shape: {sent_att_vecs_dr.shape}")  # Check the output shape
        doc_sa, _ = self.gru(sent_att_vecs_dr)
        print(f"doc_sa shape after GRU: {doc_sa.shape}")  # Should be (batch_size, 7, 2*n_units)
        doc_att_vec, sent_att_coeffs = self.attention(doc_sa)
        print(f"doc_att_vec shape: {doc_att_vec.shape}")  # Check final attention output shape

        doc_att_vec_dr = self.dropout(doc_att_vec)
        doc_att_vec_dr = self.lin_out(doc_att_vec_dr)
        return self.preds(doc_att_vec_dr), word_att_coeffs, sent_att_coeffs
