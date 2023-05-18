import torch as T
import math 
import numpy as np

from T.nn import TransformerEncoder, TransformerEncoderLayer

#--------------- Define Positional Encoder Class  -------------------

class PositionalEncoding(T.nn.Module):

    '''
    implement sin cosine positional encodings as described by:

        P(k, 2i) = sin(k / n^(2i/d))
        P(k, 2i+1) = cos(k / n^(2i/d))

    '''

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):

        super().__init__()
        self.dropout = T.nn.Dropout(p = dropout)
          
        position = T.arange(max_len).unsqueeze(1)  # get position tensor in form [ [1], [2], ..., [max_len] ]
        
        '''
        denominator for positional encodings given by: n^(2i/d)
        
        given the exponent rule:

            a^-b = 1/a^b

        => e^(-2iln(10000)/d_model) = 1/10000^-2i/d_model = 10000^(-2i/d_model)
        '''
        
        denominator = T.exp(T.arange(0,d_model,2)*(-math.log(10000.0)/d_model))
        
        pe = T.zeros(max_len, 1, d_model)  # initalise positional encoding matrix 
        pe[:,0,0::2] = T.sin(position * denominator) # fill in sin positional embeddings in even columns
        pe[:,0,1::2] = T.cos(position * denominator) # fill in cosine positional embeddings in odd columns
        
        # store encodings in buffer (gradient free)
        self.register_buffer('pe', pe)


    def forward(self, x: Tensor) -> Tensor:
        '''
        Arguments:
              x: Tensor, shape [seq_len, batch_size, embedding_dim]
        ''

        x = x + self.pe[:x.size(0)]  # use additive positional embeddings, using stored values to save calculation time
        return self.dropout(x)

#--------------- Define Multihead Attention ---------------

class MultiHeadSelfAttention(T.nn.Module):

    def __init__(self, d_model, num_heads):
        
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0  # enure that embedding dimension can be equally divided among attention heads
        
        # store parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // d_heads  # calculate dimension of input to attention heads

        self.W_Q = T.nn.Linear(d_model, d_model)  # initalise query weight matrix
        self.W_K = T.nn.Linear(d_model, d_model)  # initalise key weight matrix
        self.W_V = T.nn.Linear(d_model, d_model)  # initalise value weight matrix
        
        self.linear_out = nn.Linear(d_model, d_model )
    

    # split data across attention heads
    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1,2) # split dimensions across attention heads

    
    # calculate scaled dot product attention
    def attention(self, Q, K, V, mask = None):
        
        if mask is not None:
            scores = scores.
        
        attention_weights = T.softmax(scores, dim=-1)

        return T.matmul()

    def forward(self, input_embeddings, mask = None):


#--------------- Define Transformer Class -------------------

# Transformer class adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html 
class Transformer(T.nn.Module):

    def __init__(
        self,
        num_tokens, # vocabulary size
        d_model, # number of embedding dimensions
        num_heads, # number of heads in multihead attention
        num_encoder_layers,
        num_decoder_layers,
        dropout):

        super().__init__()  # init nn.module
        
        self.model_type = 'Transformer'
        
        # define positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # define transformer encoder
        encoder_layers = TransformerEncoderLayers 
        self.transformer_encoder = TransformerEncoderLayer(d_model, num_heads)
        self.encoder = T.nn.Emebdding(num_tokens, d_model)  # look up table to store word embeddings
    
        # define transformer decoder
        

        # initalise weights
        self.init_weights()

        """
        # Code for T.nn.Transformer Implementation

        # define transformer layers
        self.transformer = T.nn.Transformer(
            d_model = d_model,
            nhead = num_heads,
            num_encoder_layers = num_encoder_layers,
            num_decoder_lauers = num_decoder_layers,
            dropout = dropout)
        """

    def forward(self):
        pass

    def init_weights(self):
        pass
