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


#Encoder using bidirectional GRU Architecture
class BiGRUEncoder(T.nn.Module):
    
    def __init__(self, input_dim = 256, hidden_dim = 512, num_layers = 1, dropout = 0.0, bidirectional = True, device = 'cuda'):
        
        super(BiGRUEncoder, self).__init__()  #initalise T.nn.Module
        
        #store model configuration
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.device = device

        #initalise gru
        self.gru = T.nn.GRU(input_dim, hidden_dim, num_layers, batch_first = True, bidirectional = bidirectional, dropout = dropout, device = device)

    def forward(self, x):
         
        # get final time step hidden state from last layer gru, discard intermediate oututs
        _, hidden = self.gru(x)        
        
        # concantenate hidden states from each direction
        hidden = T.cat((hidden[-2]. hidden[-1]), dim = -1)

        return hidden  # return dim is B, D * hidden_dim, where D = 2 if bidirectional, otherwise 1 


#Decoder using birdirectional GRU Architecture
class BiGRUDecoder(T.nn.Module):

    def __init__(self, output_dim: int, input_dim = 512, hidden_dim = 512, num_layers = 1, dropout = 0.0, bidirectional = False, device = 'cuda'):
        
        super(BiGRUDecoder, self).__init__() #initalise T.nn.module
        
        #store model configuration
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional 
        self.device = device

        #initalise gru
        self.gru = T.nn.GRU(input_dim, hidden_dim, num_layers, batch_first = True, bidirectional = bidirectional, dropout)


        #initalise fully connected layer for final classifcation
        if bidirectional:
            fc_size = hidden_dim *2
        else:
            fc_size = hidden_dim

        self.fc = T.nn.Linear(fc_size, output_dim)


    def forward(self, h, seq_len = 512):
        '''
        hidden is initial hidden state from encoder (encoded token) 
        
        '''
        
        # inital state
        batch_size = h0.size(0) #batch is first dim since batch_first = True
        outputs = []

        # generate inital input to gru (zero matrix) of dims (B, seq_len, Hin) Note sequence length is one as single token used as input
        x = T.zeros(batch_size, 1, self.fc.out_features, device = self.device)
        
        for _ in range(seq_len):
            
            x, hidden = self.gru(x, hidden) #get output and hidden state
            
            x = self.fc(x)
            outputs.append(x)
            




#define autoencoder using budirectional GRUs
def BiGruAutoencoder(T.nn.Module):

    def __init__(self):
        pass

    def encode(self):
        pass

    def decode(self):
        pass

    def forward(self):
        pass


#### *** USE GRADIENT CLIPPING WIHLE TRAINING ***

