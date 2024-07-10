from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable

from allennlp.modules.text_field_embedders import TextFieldEmbedder


class KNRM(nn.Module):
    '''
    Paper: End-to-End Neural Ad-hoc Ranking with Kernel Pooling, Xiong et al., SIGIR'17
    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 n_kernels: int):

        super(KNRM, self).__init__()

        self.word_embeddings = word_embeddings

        # static - kernel size & magnitude variables
        mu = torch.FloatTensor(self.kernel_mus(n_kernels)).view(1, 1, 1, n_kernels)
        sigma = torch.FloatTensor(self.kernel_sigmas(n_kernels)).view(1, 1, 1, n_kernels)

        self.register_buffer('mu', mu)
        self.register_buffer('sigma', sigma)

        # assumed there is no bias, since it is gray colored in presentation 
        self.out = nn.Linear(n_kernels, 1, bias=False)

    def forward(self, query: torch.Tensor, document: torch.Tensor) -> torch.Tensor:
        #pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------
        # shape: (batch, query_max)
        query_pad_oov_mask = (query > 0).float() # > 1 to also mask oov terms
        # shape: (batch, document_max)
        document_pad_oov_mask = (document > 0).float()

        # shape: (batch, query_max, emb_dim)
        query_embeddings = self.word_embeddings({'tokens': {'tokens': query}})
        # shape: (batch, document_max, emb_dim)
        document_embeddings = self.word_embeddings({'tokens': {'tokens': document}})

        #todo

        # calculating cosine similarity
        # firstly we norm our query and document embeddings
        query_normd = query_embeddings/(query_embeddings.norm(p = 2, dim = -1, keepdim = True) + 1e-13)
        document_normd= document_embeddings/(document_embeddings.norm(p = 2, dim = -1, keepdim = True) + 1e-13)

        # shape: (batch, query_max, document_max)
        similarity_matrix = torch.bmm(query_normd, document_normd.transpose(-1, -2))
        
        # kernel pooling tensor -> unsqueeze similarity matrix to be (batch, query_max, document_max, 1)
        # then with broadcasting, we are matching dimensions of self.mu and self.sigma:
        # (1, 1, 1, n_kernels) -> (batch, query_max, document_max, n_kernels)
        # shape: (batch, query_max, doc_max, n_kernels)
        kernel_pooling = torch.exp(-0.5 * ((similarity_matrix.unsqueeze(3) - self.mu) ** 2) / (self.sigma ** 2))

        # we get some values in kernel pooling in padding that are != 0
        # so we need to use the masks to replace these values with 0 (as it was noted in hints and tricks in assignment2.md)
        # shape: (batch, query_max, 1)
        query_mask = query_pad_oov_mask.unsqueeze(2)
        # shape: (batch, 1, document_max)
        document_mask = document_pad_oov_mask.unsqueeze(1)

        # here we use also unsqueeze(3) to match query_mask and document_mask size with kernel pooling:
        # shape: (batch, query_max, doc_max, n_kernels)
        kernel_pooling = kernel_pooling * query_mask.unsqueeze(3) * document_mask.unsqueeze(3)

        # now we need to sum by document dimension to prepare for logarithm
        # shape: (batch, query_max, n_kernels)
        kernel_sum_by_doc = torch.sum(kernel_pooling, dim = 2)

        # logarithm calculation, we recall from slides that it's better with log(soft_tf), but we need to add really small value
        # shape: (batch, query_max, n_kernels)
        kernel_log = torch.log(kernel_sum_by_doc + 1e-12)

        # then, we sum the values by query dimension from kernel_log tensor
        # shape: (batch, n_kernels)
        kernel_log_sum = torch.sum(kernel_log, dim = 1)

        # finally, we get to the output layer
        output = self.out(kernel_log_sum)

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

