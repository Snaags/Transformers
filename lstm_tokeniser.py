"""

Code to tokenise binary data using an lstm autoencoder

Created on: 08/05/23

"""

import torch as T
import numpy as np

class ByteLevelAutoencoder(T.nn.Module):
    def __init__(self, encoder_layers, decoder_layers, input_dim = 256, hidden_dim = 512, enocder_bidirectional = True, decoder_bidirectional = True, architecture_type = 'LSTM'):

        super(BiLSTMAutoencoder, self).__init__() #initalise torch nn modlue
        
        #define recurrent block architecture for encoding/decoding
        if architecture_type.lower() = 'lstm':
            self.reccurent_block = T.nn.LSTM 
        elif architecture_type.lower() == 'gru':
            self.recurrent_block = T.nn.GRU

        else:
            raise ValueError('ERROR::: Invalid Architecture Type')
        

        #define encoder and decoder architectures:
        self.encoder = recurrent_block(input_dim, hidden_dim, num_layers = encoder_layers, batch_first  = True, bidirectional = encoder_bidirectional)
        self.decoder = recurrent_block(2 * hidden_dim, input_dim, num_layers = decoder_layers, batch_first = True, bidirectional = decoder_bidirectional)


    #function to encode data
    def encode(self, x):
        pass
    

    #function to decode encoded data 
    def decode(self, x):
        pass


    #implement autoencode forward pass
    def forward(self, x):
        
        #pass data throu
        _, (hidden, _) = self.encode(x)

        hidden = hidden.view(1, x.size(0), -1)

        output, _ = self.decoder(hidden.repeat(x.size(1), 1, 1)).transpose(0,1)
        return output

def lstm 

def Custom_GRU(T.nn.Module);

    def __init__


