#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Encoding models to compress byte sequences into 1d embeddings

Created on Mon May 15 12:24:42 2023

@author: jack
"""

# import required modules and libraries
import torch as T
import torch.nn.functional as F
import time
import random

# --------------------------- Sequence Encoding using GRUs --------------------

#Encoder using bidirectional GRU Architecture                                                                                                                                            
class BiGRUEncoder(T.nn.Module):                                                                                                                                                               
    def __init__(self, input_dim: int, hidden_dim: int, num_layers = 1, dropout = 0.0, bidirectional = True, device = 'cuda'):                                                         
        '''
        Encoder using GRU architecture

        Parameters
        ----------
        input_dim : Int
            Input dimensions to GRU.
        hidden_dim : Int
            Hidden dimensions of GRU encoder.
        num_layers : Int, optional
            Number of stacked GRU layers. The default is 1.
        dropout : Float, optional
            Dropout ratio (ratio of connections zeroed). The default is 0.0.
        bidirectional : Bool, optional
            Use birdirectional GRU if true, otherwise use single GRU encoder. The default is True.
        device : Str, optional
            Name of device to store model on. The default is 'cuda'.

        Returns
        -------
        None.

        '''
        
        super(BiGRUEncoder, self).__init__() #initalise T.nn.Module                                                                                                                                 

        #store model configuration                                                                                                                                                       
        self.input_dim = input_dim                                                                                                                                                       
        self.hidden_dim = hidden_dim                                                                                                                                                     
        self.num_layers = num_layers                                                                                                                                                     
        self.dropout = dropout                                                                                                                                                           
        self.bidirectional = bidirectional                                                                                                                                               
        self.device = device                                                                                                                                                                      
        
        # dimension scalar if bidirections (output size is doubled due to concat operation)
        if self.bidirectional: 
            self.D = 2
        else:
            self.D = 1
        
        
        #initalise gru                                                                                                                                                                   
        self.gru = T.nn.GRU(input_dim, hidden_dim, num_layers, batch_first = True, bidirectional = bidirectional, dropout = dropout, device = device)                                    

    def forward(self, x):                                                                                                                                                                   

        # get final time step hidden state from last layer gru, discard intermediate oututs                                                                                              
        _, hidden = self.gru(x)                                                                                                                                                          

        # concantenate hidden states from each direction        
        if self.bidirectional:                                                                                                                         
            hidden = T.cat((hidden[-2], hidden[-1]), dim = -1)                                                                                                                                              
        else:  #otherwise use final state output
            hidden = hidden[-1]
            
        return hidden  # return dim is B, D * hidden_dim, where D = 2 if bidirectional, otherwise 1                                                                                           



#---------------- Decoder using birdirectional GRU Architecture ---------------
                                                                                                                                     
class BiGRUDecoder(T.nn.Module):                                                                                                                                                         

    def __init__(self, input_dim: int, hidden_dim: int, vocab_size = 258, embedding_layer = None, num_layers = 1, dropout = 0.0, bidirectional = False, init_scheme = 'zeros', device = 'cuda'):                                       
        '''
        Defines GRU based decoder for decoding encoder hidden state into byte sequence

        Parameters
        ----------
        input_dim : Int, optional
            Numbber of dimensions of input tokens.
        hidden_dim : Int, optional
            Number of dimensions in GRU hidden state. 
        vocab_size : Int, optional
            Size of vocabulary used during encoding (bytes plus speicla tokens). The default is 258.
        embedding_layer : T.nn.module, optional
            Embedding layer to convert one hot predictions into net input token, should be same as that used to train encoder, use one hot encoding if None. The default is None.
        num_layers : Int, optional
            Number of stacked GRU layers for decoder. The default is 1.
        dropout : Float, optional
            Dropout ratio (ratio of connections zeroed). The default is 0.0.
        bidirectional : Bool, optional
            Use birdirectional GRU if true, otherwise use single GRU encoder. The default is True.
        init_scheme : Str, optional
            If 'zeros' then first gru hidden state is initalised with encoder output, if 'encoder', then all stacked grus are initalised with encoder output. The default is 'zeros'.
        device : Str, optional
            Name of device to store model on. The default is 'cuda'.   

        Raises
        ------
        ValueError
            Raised if invalid hidden state innit scheme is specified ias input argument.

        Returns
        -------
        None.

        '''
        
        
        super(BiGRUDecoder, self).__init__() #initalise T.nn.module                                                                                                                      

        #store model configuration                                                                                                                                                       
        self.input_dim = input_dim                                                                                                                                                       
        self.hidden_dim = hidden_dim                                                                                                                                                     
        self.num_layers = num_layers                                                                                                                                                     
        self.dropout = dropout                                                                                                                                                           
        self.bidirectional = bidirectional                                                                                                                                               
        self.device = device                                                                                                                                                             
        self.vocab_size = vocab_size
        
        # dimension scalar if bidirections (output size is doubled due to concat operation)
        if self.bidirectional: 
            self.D = 2
        else:
            self.D = 1
        
        #initalise gru                                                                                                                                                                   
        self.gru = T.nn.GRU(input_dim, hidden_dim, num_layers,batch_first = True, bidirectional = bidirectional, dropout = dropout, device = self.device)                                                               
        self.embedding_layer = embedding_layer  #store model embedding layer
        
        # fc input is gru output, double hidden dim if bidirectional                                                                                                                                         
        # fc output is a token in vocabulary   
        self.fc = T.nn.Linear(hidden_dim * self.D , self.vocab_size, device = self.device)                                                                                                                                  
        
        # if init scheme is zeros, first gru layer is initalied with encoder output as hidden state whilst rest are initalised with zero
        # if init scheme is encoder, all gru layers are initilased with encoder output as hidden state
        self.init_scheme = init_scheme.lower()
        
        if self.init_scheme != 'zeros' and self.init_scheme != 'encoder':  # validate innit scheme
            raise ValueError('BiGRUDecoder Invalid init_scheme selected, valid options are: "zeros", "encoder"')
            
    
    def forward(self, token_in, hidden, argmax = False):                                                                                                                                                 

        '''                                                                                       
        forward pass of decoder
                                                                                       
        hidden is initial hidden state from encoder (encoded token)                                                                                                                      
        token in is start token/ last predicted/ teacher forced token
        argmax returns one hot prediction if true, otherwise probability distribution for training
        
        returns decoder hidden state, token predictions
        '''                                                                                                                                                                              
        
        _, hidden = self.gru(token_in, hidden)  # pass data through gru
            
        # concat outputs of both decoders if birdirectional gru is used
        if self.bidirectional:
            out = T.cat((hidden[-2], hidden[-1]), dim = -1)  
                
        else:  # otherwise us last hidden state
            out = hidden[-1]
            
        
        #convert to integer token prediction if argmax is true,
        if argmax: 
            
            #get predicted tokens using fc layers
            out = F.softmax(self.fc(out))
            
            _, out = T.max(out, dim =-1) 
        
        return hidden, out 
      
    
    def decode_seq(self, h0 = None, seq_len = 512):
        
        outputs = []
        
        #get inital hidden state
        hidden = self.encoder_token_to_hidden_state(h0)
        
        # get inital input as start token (256)
        start_token = T.tensor(256, device = self.device, dtype = T.int64)
        next_input = self.tokenise_int(start_token)
        
        #convert to (N, h) dim 
        next_input = next_input.view((1,-1))
        
        # iterate over sequence
        for i in range(seq_len):
            
            hidden, pred_token = self(next_input, hidden, argmax = True)  #get next predicted token and decoder hidden state
            outputs.append(pred_token) # append predicted token to output sequence
            
            # tokenise predicted vocab item using embedding layer for next input
            next_input = self.tokenise_int(pred_token)
            
        return outputs #return output sequences
        
    # function to convert integer sequences into token representation
    def tokenise_int(self, x: int):
        
        # use embedding layer to tokenise int is provided during init
        if self.embedding_layer is not None:
            return self.embedding_layer(x)
        
        # otherwise use one hot encoding
        else:
            return F.one_hot(x, self.vocab_size)
        
        
    # covert encoder token to decoder hidden state
    def encoder_token_to_hidden_state(self, h):
        # h is encoder hidden state
        
        # if zeros, init first layer gru with encoder output and rest with zeros matrices
        if self.init_scheme == 'zeros':
            h0 = T.zeros((self.D * self.num_layers, h.size(0), self.hidden_dim), dtype = T.float32, device = self.device)  #init zero matrix
            h0[0] = h  #replace first layer with encoder hidden stat
        
        # if encoder initalise all decoderes stacked gru layers with encoder hidden state 
        elif self.init_scheme == 'encoder':
            h0 = h.unsqueeze(0).repeat(self.D*self.num_layers, 1, 1)
        
        return h0
        
      
class GRUAutoencoder(T.nn.Module):
    
    def __init__(self, 
                 embedding_dim: int, 
                 hidden_dim: int,
                 encoder_layers: int,
                 decoder_layers: int,
                 vocab_size = 258,
                 byte_seq_len = 512,
                 encoder_bidirectional = True,
                 decoder_bidirectional = False,
                 decoder_init_scheme = 'zeros',
                 embedding_layer = True,
                 dropout = 0.0,
                 rng_seed = time.time(),
                 device = 'cuda'):
        
        
        '''
        Defines GRU based decoder for decoding encoder hidden state into byte sequence

        Parameters
        ----------
        embedding_dim : Int
            Number of dimensions of input tokens to model. 
        hidden_dim : Int
            Number of dimensions in encoder and decoder hidden state ( must be equal to d_model in downstream transformer). 
        encoder_layers : Int
            Number of stacked GRU cells in model encoder (should be higher than decoder to ensure embedding has most information)
        decoder_layers : Int
            Number of stacked GRU cells in model decoder.
        vocab_size : Int, optional
            Size of vocabulary used during encoding and  (bytes plus speicla tokens). The default is 258.
        byte_seq_len : Int, optional
            Number of bytes in a sequence to encode into a single vector. The default is 512.
        encoder_bidirectional : Bool, optional
            Use bidirectional encoder if True, otherwise use single GRU encoder. The default is True
        decoder_bidirectional : Bool, optional
            Use bidirectional decoder if True, otherwise use single GRU encoder. The default is False
        decoder_init_scheme : Str, optional
            If 'zeros' then first gru hidden state in decoder is initalised with encoder output with the rest being initalised with zeros, if 'encoder', then all stacked grus are initalised with encoder output. The default is 'zeros'.
        embedding_layer : Bool, optional
            Use embedding layer to tokenise vocabulary if True, otherwise use one hot encoding. The default is True
        dropout : Float, optional
            Dropout ratio for model. The default is 0.0
        rng_seed : int, optional
            Seed for random number generation. The default is current time.
        device : Str, optional
            Name of device to store model on. The default is 'cuda'. 

        Returns
        -------
        None.

        '''
        
        super(GRUAutoencoder, self).__init__() # initalise t.nn.module
        
        # initalise autoencoder configuration
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.byte_seq_len = byte_seq_len
        self.dropout = dropout
        self.device = device
        self.rng = random.Random(rng_seed)
        
        # initalise module layers
        if embedding_layer:  # use embedding layer if specified
            self.embedding_layer = T.nn.Embedding(self.vocab_size, self.embedding_dim, device = self.device, dtype= T.float32)
        
        # otherwise use one hot encoding
        else:  
            self.embedding_layer = self.one_hot_encode
            self.embedding_dim = self.vocab_size
        
        # initalise gru encoder
        self.encoder = BiGRUEncoder(input_dim = self.embedding_dim, # input is embedding dim, as hidden states are tokenised before being used as input 
                                    hidden_dim = self.hidden_dim,  # hyperparameter used to specify number of dimensiuons in hidden state ( will be equal to d_model in downstream transformer)
                                    num_layers = encoder_layers,  # number of stacker gru layers for encoder 
                                    dropout = self.dropout,  # dropout ratio
                                    bidirectional = encoder_bidirectional,  # use bidirectional encoder if True
                                    device = self.device)  # device to run model on
        
        # initalise gru decoder
        self.decoder = BiGRUDecoder(input_dim = self.embedding_dim,  # input is embedding dim, as hidden states are tokenised before being used as input
                                    hidden_dim = self.hidden_dim * self.encoder.D,  # FIXweifnseofn # must be equal to encoder hidden dim as state is transferred (unless projection is implemented)
                                    num_layers = decoder_layers, # number of stacked grus in decoder
                                    vocab_size = self.vocab_size,  # model vocabulary size
                                    embedding_layer = self.embedding_layer,  # layer used to convert tokens to embeddings
                                    bidirectional = decoder_bidirectional,  # bidirectional decoder used if True
                                    init_scheme = decoder_init_scheme,  # initalise stack
                                    dropout = self.dropout,  # dropout ratio 
                                    device = self.device)  # device to run model on
        
        
        
    # autoencoder forward pass, takes sequence input and converts to decoded sequence 
    def forward(self, x):
        
        x = self.embedding_layer(x)
        
        encoded_seq = self.encoder(x)
        
        decoded_seq = self.decoder.decode_seq(encoded_seq, self.byte_seq_len)
        
        return decoded_seq
    
        
        
    def train(self, train_dl, val_dl = None, teacher_forcing_ratio = 1.):
        
        hyperparams = { 'learning_rate' : 0.001,
                       'weight_decay' : 0.0001,
                       'batch_size' : 32,
                       'dropout' : 0.1,
                       'epochs' : 100,
                       'teacher_forcing_ratio' : 0.9}
        
        
        loss_fn = T.nn.CrossEntropyLoss()
        optimiser = T.optim.AdamW(self.parameters(), lr=hyperparams['learning_rate'] , weight_decay=hyperparams['weight_decay'])  #define optimiser
        scheduler = T.optim.lr_scheduler.CosineAnnealingLR(optimiser, hyperparams['epochs'], verbose=False)
        
        
        # start recording metrics
        history = {}
        history['train_loss'] = []
        history['val_loss'] = []
        history['train_correct'] = []
        history['val_correct'] = []
        
        start_time = time.time()
        
        for epoch in range(hyperparams['epochs']):

            train_loss = 0
            train_bytes_correct = 0
            
            val_loss = 0
            val_bytes_correct = 0
     
            for batch_num, batch in enumerate(train_dl):
                
                # get batch of sequences
                x,_ = batch
                
                #convert x to tokens using embedding layer
                x_tokens = self.embedding_layer(x)
                
                #encode sequences using encoder to get (B x hidden_dim) vector representation
                x_encoded = self.encoder(x_tokens)
                
                # get inital input as start token (256)
                start_token = T.tensor(256, device = self.device, dtype = T.int64)
                next_input = self.embedding_layer(start_token)
                next_input = next_input.unsqueeze(0).unsqueeze(0).expand(x_encoded.size(0),-1,-1)
                
                 # resize to correct dimensions
                # get inital decoder hidden state
                hidden = self.decoder.encoder_token_to_hidden_state(x_encoded)
                
                #initalise loss
                batch_loss = 0
                batch_bytes_correct = 0
                loss = None
                
                #decide whether to apply teach forcing
                teacher_forcing = True if self.rng.random() < teacher_forcing_ratio else False
                
                #decode tokens
                for i in range(self.byte_seq_len + 1):
                    
                    #print(f'input shape: {next_input.shape}   | hidden shape: {hidden.shape}')
                    
                    hidden, pred_state = self.decoder(next_input, hidden)
                    
                    unnormalised_prediction = self.decoder.fc(pred_state)
                    
                    if loss is None:
                        loss = loss_fn(unnormalised_prediction, x[:,i + 1])
                    else:
                        loss += loss_fn(unnormalised_prediction, x[:,i + 1])
                    
                    batch_loss += loss.item()
     
                    pred_token = T.argmax(unnormalised_prediction, dim = -1)  #convert decoder hidden state to 
                    
                    batch_bytes_correct += T.sum(T.eq(pred_token, x[:,i + 1])).item()
                    
                    if teacher_forcing: pred_token = x[:,i]
                    
                    next_input = self.embedding_layer(pred_token).unsqueeze(1)
                    
                   
                #calculate loss for batch as mean loss over sequence
                loss = loss/(self.byte_seq_len+1)  #mean loss over sequence to make invariant to length
                batch_loss = batch_loss/(self.byte_seq_len+1) 
                batch_bytes_correct = batch_bytes_correct/ (self.byte_seq_len * hyperparams['batch_size'])
                
                print(f'Train Loss: {batch_loss}   |   Train Correct: {batch_bytes_correct}', end = '\r')
                
                train_loss += batch_loss
                train_bytes_correct += batch_bytes_correct
                
                optimiser.zero_grad()
                loss.backward()
                T.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  #apply gradient clipping
                optimiser.step()
                

            if val_dl is not None:
                
                for batch_num, batch in val_dl:
                    
                    # get batch of sequences
                    x,_ = batch

                    #convert x to tokens using embedding layer
                    x_tokens = self.emebdding_layer(x)

                    #encode sequences using encoder to get (B x hidden_dim) vector representation
                    x_encoded = self.encoder(x_tokens)

                    # get inital input as start token (256)
                    start_token = T.tensor(256, device = self.device, dtype = T.int64)
                    next_input = self.embedding_layer(start_token)
                    hidden = self.decoder.encoder_token_to_hidden_state(x_encoded)
                    next_input = next_input.unsqueeze(0).unsqueeze(0).expand(x_encoded.size(0),-1,-1)
                    
                    batch_loss = 0
                    batch_bytes_correct = 0
                    
                    #decode tokens
                    for i in range(self.byte_seq_len + 1):

                        hidden, pred_state = self.decoder(next_input, hidden)  # pass token through decoder

                        unnormalised_prediction = self.decoder.fc(pred_state) #predict probabilities for next token
                        
                        batch_loss += loss_fn(unnormalised_prediction, x[:,i + 1]).item() # calculate validation loss
                        
                        pred_token = T.argmax(unnormalised_prediction, dim = -1)  #convert decoder hidden state to 
                        
                        batch_bytes_correct += T.sum(T.eq(pred_token, x[:,i + 1])).item()
                        
                        next_input = self.embedding_layer(pred_token).unsqueeze(1)
                        
                    batch_loss = batch_loss / self.byte_seq_len
                    batch_bytes_correct = batch_bytes_correct/ (self.byte_seq_len * hyperparams['batch_size'])  #normalise bytes correct
                    
                    print(f'Val Loss: {batch_loss}   |   Val Correct: {batch_bytes_correct}', end = '\r')
                    
                    val_loss += batch_loss
                    val_bytes_correct += batch_bytes_correct
            
                    
                history['val_loss'].append(val_loss/len(val_dl.dataset))
                history['val_correct'].append(val_bytes_correct/len(val_dl.dataset))
            
            history['train_loss'].append(train_loss/len(train_dl.dataset))
            history['train_correct'].append(train_bytes_correct/len(train_dl.dataset))
            
            
            
            if scheduler is not None:  # step learning rate if schedular provided
                scheduler.step()
             
                
        return history
                
    def one_hot_encode(self,x):
        return F.one_hot(x, self.vocab_size)