"""

Code to tokenise binary data using an lstm autoencoder

Created on: 08/05/23

"""

import torch as T
import torch.nn.functional as F
import numpy as np


#Encoder using bidirectional GRU Architecture                                                                                                                                            
class BiGRUEncoder(T.nn.Module):                                                                                                                                                           
    
    def __init__(self, input_dim = embedding_dim, hidden_dim = 512, num_layers = 1, dropout = 0.0, bidirectional = True, device = 'cuda'):                                                         
                                                                                                                                                                                          
        super(BiGRUEncoder, self).__init__() #initalise T.nn.Module                                                                                                                                 

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
        hidden = T.cat((hidden[-2], hidden[-1]), dim = -1)                                                                                                                                              

        return hidden  # return dim is B, D * hidden_dim, where D = 2 if bidirectional, otherwise 1                                                                                           


# Decoder using birdirectional GRU Architecture                                                                                                                                           
class BiGRUDecoder(T.nn.Module):                                                                                                                                                         

    def __init__(self, input_dim = 512, hidden_dim = 512, vocab_size = 258, embedding_layer = None, num_layers = 1, dropout = 0.0, bidirectional = False, init_scheme = 'zeros', device = 'cuda'):                                       

        super(BiGRUDecoder, self).__init__() #initalise T.nn.module                                                                                                                      

        #store model configuration                                                                                                                                                       
        self.input_dim = input_dim                                                                                                                                                       
        self.hidden_dim = hidden_dim                                                                                                                                                     
        self.num_layers = num_layers                                                                                                                                                     
        self.dropout = dropout                                                                                                                                                           
        self.bidirectional = bidirectional                                                                                                                                               
        self.device = device                                                                                                                                                             
        self.vocab_size = vocab_size
        
        if self.bidirectional: 
            self.D = 2
        else:
            self.D = 1
        
        #initalise gru                                                                                                                                                                   
        self.gru = T.nn.GRU(input_dim, hidden_dim, num_layers,batch_first = True, bidirectional = bidirectional, dropout = dropout)                                                               
        self.embedding_layer = embedding_layer

        #initalise fully connected layer for final classifcation                                                                                                                         
        if bidirectional:                                                                                                                                                                
            fc_size = hidden_dim *2                                                                                                                                                      
        else:                                                                                                                                                                            
            fc_size = hidden_dim                                                                                                                                                         

        self.fc = T.nn.Linear(fc_size, self.vocab_size)                                                                                                                                       
        
        self.embedding_layer  = embedding_layer
        
        # if init scheme is zeros, first gru layer is initalied with encoder output as hidden state whilst rest are initalised with zero
        # if init scheme is encoder, all gru layers are initilased with encoder output as hidden state
        
        self.init_scheme = init_scheme.lower()
        
        if self.init_scheme != 'zeros' and self.init_scheme != 'encoder': 
            raise ValueError('BiGRUDecoder Invalid init_scheme selected, valid options are: "zeros", "encoder"')
            
    
    def forward(self, token_in, hidden, argmax = False):                                                                                                                                                 

        '''                                                                                                                                                                              
        hidden is initial hidden state from encoder (encoded token)                                                                                                                      

        '''                                                                                                                                                                              
        
        _, hidden = self.gru(token_in, hidden)
            
        if bidirectional:
            out = T.cat((hidden[-2], hidden[-1]), dim = -1)  
                
        else:
            out = hidden[-1]
            
            
        #get predicted tokens
        pred_token = F.softmax(self.fc(out))
        
        if argmax: _, pred_token = T.max(pred_token, dim =-1) 
        
        return hidden, pred_tokem 
      
    
    def decode_seq(self, h0 = None, seq_len = 512):
        
        outputs = []
        
        #get inital hidden state
        hidden = self.encoder_token_to_hiddne_state(h0)
        
        # get inital input as start token (256)
        start_token = self.tokenise_int(T.tensor(256, device = self.device, dtype = T.int64))
        next_input = self.tokenise_int(start_token)
        
        #convert to (N, h) dim 
        next_input = next_input.view((1,-1))
        
        # iterate over sequence
        for i in range(seq_len):
            
            hidden, pred_token = self(next_token, hidden, argmax = True)
            
            output.append(pred_token)
            
            next_input = self.tokenise_int(pred_token)
            
        return outputs
        
        
    def tokenise_int(self, x: int):
        
        if self.embedding_layer is not None: 
            return self.embedding_layer(pred_token)
        
        else:
            return F.one_hot(pred_token, self.vocab_size)
            
    def encoder_token_to_hidden_state(self, h):
        
        if self.init_scheme == 'zeros':
            h0 = T.zeros((self.D * self.num_layers, h.size(0), self.hidden_dim),dtype = T.float32, device = device)
            h0[0] = h
            
        elif self.init_scheme == 'encoder':
            h0 = h.unsqueeze(0).repeat(self.D*self.num_layers, 1, 1)
        
        return h0
        
        
        


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

