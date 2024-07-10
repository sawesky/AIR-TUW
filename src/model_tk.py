from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable

import math

from allennlp.modules.text_field_embedders import TextFieldEmbedder

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TK(nn.Module):
    '''
    Paper: S. Hofstätter, M. Zlabinger, and A. Hanbury 2020. Interpretable & Time-Budget-Constrained Contextualization for Re-Ranking. In Proc. of ECAI 
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 n_kernels: int,
                 n_layers: int,
                 n_tf_dim: int,
                 n_tf_heads: int):

        super(TK, self).__init__()

        self.word_embeddings = word_embeddings
        
        # positional encoding
        self.positional_encoding = PositionalEncoding(d_model=n_tf_dim)

        # transformer layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_tf_dim, nhead=n_tf_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # static - kernel size & magnitude variables
        mu = torch.FloatTensor(self.kernel_mus(n_kernels)).view(1, 1, 1, n_kernels)
        sigma = torch.FloatTensor(self.kernel_sigmas(n_kernels)).view(1, 1, 1, n_kernels)

        self.register_buffer('mu', mu)
        self.register_buffer('sigma', sigma)
        
        # one branch for log normalization, one for length normalization
        self.log_norm_layer = nn.Linear(n_kernels, 1, bias=False)
        self.length_norm_layer = nn.Linear(n_kernels, 1, bias=False)

        # output layer
        self.out = nn.Linear(2, 1, bias=False)


    def forward(self, query: torch.Tensor, document: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        # shape: (batch, query_max)
        query_pad_oov_mask = (query > 0).float() # > 1 to also mask oov terms
        # shape: (batch, document_max)
        document_pad_oov_mask = (document > 0).float()

        # shape: (batch, query_max, emb_dim)
        query_embeddings = self.word_embeddings({'tokens': {'tokens': query}})
        # shape: (batch, document_max, emb_dim)
        document_embeddings = self.word_embeddings({'tokens': {'tokens': document}})
        
        # shape: (batch, query_max, emb_dim)
        query_embeddings = self.positional_encoding(query_embeddings)
        # shape: (batch, document_max, emb_dim)
        document_embeddings = self.positional_encoding(document_embeddings)

        # apply transformer layers
        query_embeddings = self.transformer(query_embeddings)
        document_embeddings = self.transformer(document_embeddings)

        # normalization
        query_normd = query_embeddings / (query_embeddings.norm(p=2, dim=-1, keepdim=True) + 1e-13)
        document_normd = document_embeddings / (document_embeddings.norm(p=2, dim=-1, keepdim=True) + 1e-13)

        # shape: (batch, query_max, document_max)
        similarity_matrix = torch.bmm(query_normd, document_normd.transpose(-1, -2))

        # shape: (batch, query_max, doc_max, n_kernels)
        kernel_pooling = torch.exp(-0.5 * ((similarity_matrix.unsqueeze(3) - self.mu) ** 2) / (self.sigma ** 2))
        
        # shape: (batch, query_max, 1)
        query_mask = query_pad_oov_mask.unsqueeze(2)
        # shape: (batch, 1, document_max)
        document_mask = document_pad_oov_mask.unsqueeze(1)
        
        # shape: (batch, query_max, doc_max, n_kernels)
        kernel_pooling = kernel_pooling * query_mask.unsqueeze(3) * document_mask.unsqueeze(3)
        # shape: (batch, query_max, n_kernels)
        kernel_sum_by_doc = torch.sum(kernel_pooling, dim=2)

        # shape: (batch, query_max, n_kernels)
        kernel_log = torch.log(kernel_sum_by_doc + 1e-12)

        # Calculate document lengths for length normalization
        document_lengths = document_mask.sum(dim=2, keepdim=True)
        # shape: (batch, query_max, n_kernels)
        kernel_length = kernel_sum_by_doc / (document_lengths + 1e-12)
        # shape: (batch, n_kernels)
        kernel_length_sum = torch.sum(kernel_length, dim=1)

        # apply dense layers for both log and length normalized outputs
        # shape: (batch, 1)
        log_norm_score = self.log_norm_layer(kernel_log.sum(dim=1))  # summing across query terms
        # shape: (batch, 1)
        length_norm_score = self.length_norm_layer(kernel_length_sum)

        # shape: (batch, 2)  
        combined_features = torch.cat([log_norm_score, length_norm_score], dim=1)
        
        # finally, we get to the output layer
        output = self.out(combined_features)
        return output


    def kernel_mus(self, n_kernels: int):
        """
        get the mu for each guassian kernel. Mu is the middle of each bin
        :param n_kernels: number of kernels (including exact match). first one is exact match
        :return: l_mu, a list of mu.
        """
        l_mu = [1.0]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu



    def kernel_sigmas(self, n_kernels: int):
        """
        get sigmas for each guassian kernel.
        :param n_kernels: number of kernels (including exactmath.)
        :param lamb:
        :param use_exact:
        :return: l_sigma, a list of simga
        """
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.0001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return l_sigma
