#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Datasets and loaders for importing androzoo dataset in pytorch

Created on Mon May 15 11:32:21 2023

@author: jack
"""

#CHAGE GET INTEM RETURN Y FAMILY

# import required modules and libraries
import torch as T
import os 
import numpy as np
import random
import time
import binascii
import torch.multiprocessing as mp

# ------------------------ Dataset Class for Androzoo -------------------------


class HexDumpDataset(T.utils.data.Dataset):
    def __init__(self, path, device = 'cuda', load_len = 0, one_hot = False, rng_seed = time.time()):
        '''
        Pytorch dataset for downloaded androzoo binary data. Assumes split is
        contained in its own folder with its sub directories structured as:
        
            /malware_/malware_type/malware_hash.bin

        Parameters
        ----------
        path : Str
            File path to directory containing dataset.
        device : Str, optional
            Device to instatiate tensors on to. The default is 'cuda'.
        load_len : Int, optional
            Number of bytes to return as a sapmle sequence, loads whole file if 0. The default is 0.
        one_hot : Bool, optional
            Returns data in one hot encdoded format if True, otherwise returns bytes as ints. The default is False.
        rng_seed : Int, optional
            Seed for random number generation, set for reproducible results, otherwise use current time. The default is time.time().

        Returns
        -------
        None.

        '''
        
        #store dataset configuration
        self.path = path
        self.device = device
        self.load_len = load_len    
        self.one_hot = one_hot
        self.rng = random.Random(rng_seed)
        
        # store path to samples and there length
        self.sample_lens = []
        self.x_paths = []
        
        # store famliy labels
        self.family_label_to_int = {'benign' : 0}
        self.int_to_family_label = ['benign']  # list of labels (index corresponds to integer label)
        self.y_family = []
        
        # store type labels
        self.type_label_to_int = {'benign' : 0}
        self.int_to_type_label = ['benign']  # list of labels (index corresponds to integer label)
        self.y_type = []
        
        # store special token -> int mapping
        self.special_token_to_int = { '<START>' : 256,
                                '<STOP>' : 257,
                                '<PAD>' : 258}
        
    
        #iterate over all subdirectories
        for root, dirs, files in os.walk(path):                                                                                                                                                                                      
            for file in files:  
                
                # find malware family and type by the subdirs the sample is located in
                file_path = os.path.join(root, file)
                split_path = root.split('/')
                malware_family = split_path[-2]
                malware_type = split_path[-1]
                
                # add family int mapping if family is previously unseen 
                if malware_family not in self.family_label_to_int:
                    self.family_label_to_int[malware_family] = len(self.int_to_family_label)
                    self.int_to_family_label.append(malware_family)
                
                # add type to int mapping if type is previously unseen 
                if malware_type not in self.type_label_to_int:
                    self.type_label_to_int[malware_type] = len(self.int_to_type_label)
                    self.int_to_type_label.append(malware_type)
                    
                # store malware path and label details
                self.y_family.append(self.family_label_to_int[malware_family])
                self.y_type.append(self.type_label_to_int[malware_type])
                self.x_paths.append(file_path)
                self.sample_lens.append(self.count_chars(file_path))
                
                
        #find length of dataset
        self.n = len(self.y_family)
        
    
    # return dataset length (in samples)
    def __len__(self):
        return self.n
    
    
    # get dataset sample given index
    def __getitem__(self, i):
        
        path = self.x_paths[i]  # get path to samples bin file
        
        # read binary file
        with open(path, 'rb') as file:                                                                                                                                                                       
            raw_bytes = file.read()                                                                                                                                                                                                    
        
        
        # randomly sample sequence if load length specifeied, otherwise use whole file
        if self.load_len > 0 : 
            seq_len = self.sample_lens[i]
            
            #sample sequences with uniform distribution (sliding window)
            if seq_len > self.load_len:
                
                start_pos = self.rng.randint(0, seq_len - self.load_len)
                raw_bytes = raw_bytes[start_pos : start_pos + self.load_len] 
                
            
        # convert bytes into int list
        hex_string = binascii.hexlify(raw_bytes).decode('utf-8')  # convert raw bytes into hex string e.g. 'FF0A23'
        split_string = [hex_string[i:i+2] for i in range(0, len(hex_string), 2)]  # split into list of bytes e.e. ['FF', '0A', '23']
        
        int_list = [self.special_token_to_int['<START>']]  # add start token to beggining of sequence 
        int_list.extend([int(val, 16) for val in split_string])  # convert byte list into int lists e.g. [255, 10, 23]
        
        # adding padding to sequence in the event that the file size is smaller than the load length
        if len(int_list) < self.load_len + 1: 
            int_list.extend([self.special_token_to_int['<PAD>'] for x in range((self.load_len + 1)- len(int_list))])  # pad until sequence is reuqired size 
        
        
        int_list.append(self.special_token_to_int['<STOP>'])  # add stop token to sequence
        
        int_tensor = T.tensor(int_list, dtype = T.int64, device = self.device)  # convert to tensor
        
        # use one hot encoding if specified, otherwise return int list
        if self.one_hot:
            return T.nn.functional.one_hot(int_tensor, num_classes = 258), self.y_family[i]
        else:
            return int_tensor, self.y_family[i]
       
    
    # count bytes in a file given file path, loads file in chunk_size number of bytes, counts entire fie at once if chunk_size = 0
    def count_chars(self, file_path, chunk_size = 0):
        
        with open(file_path, 'rb') as file:  # open file
            
            # load entire file if chunk size is 0
            if chunk_size == 0:  
                count = len(file.read())
            
            # otherwise load file in chunks
            else:
                
                chunk = file.read(chunk_size)
                count = len(chunk)  # initalise byte counter
                
                while chunk:  # count bytre for chunk in all chunks
                    chunk = file.read(chunk_size)  #load next chunk
                    count += len(chunk)  # count current chunk
        
        return count  #return byte count
        
    
    # get number of bytes in a sample from index, return list of lengths if i = -1
    def get_sample_lens(self, i = -1):
        if i == -1:
            return self.sample_lens
        
        else:
            return self.sample_lens[i]
    
    
    # getter function to return dataset labels
    def get_labels(self, level = 'family', as_tensor = True):
        
        # get family level labels
        if level == 'family':
            labels =  self.y_family
        
        # get type level labels
        if level == 'type':
            labels = self.y_type
        
        # invalid label option selected
        else:
            raise ValueError('HexDumpDataset get_labels Invalid level specified, valid options are: "family", "level"')
        
        # convert labels to torch tensor if required
        if as_tensor: labels = T.tensor(labels, dtype = T.int64, device = self.device)
        
        return labels
    
# --------------------- Function to Construct Dataloader ----------------------

def get_hex_dataloader(path, batch_size, byte_seq_len = 512, num_workers = 1, seq_level = True, supervised = False, device = 'cuda'):
    '''
    Function to Construct Dataloader for binary dataset, assumes split is
    contained in its own folder with its sub directories structured as:
    
        /malware_/malware_type/malware_hash.bin
    
    Parameters
    ----------
    byte_seq_len : Int, optional
        Sequence length to sample from binary files. The default is 512.
    seq_level : Bool, optional
        Scales file sample probability by file length if True for uniform sequence sampling across all files. The default is True.
    supervised : Bool, optional
        Imports class labels and samples classes uniformly if True. The default is False.

    Returns
    -------
    dataloader : torch.utils.data.Dataloader
        Dataloader for androzoo dataset.

    '''
    
    dataset = HexDumpDataset(path, load_len = byte_seq_len, device = device) #instantiate androzoo dataset class

    sample_weights = np.ones(dataset.__len__())  # begin with uniform sampling

    # scale sample proability by length if uniform sequence sampling required
    # (as opposed to uniform file sampling which will oversample sequences from smaller files)
    if seq_level: sample_weights  *= dataset.get_sample_lens()

    # scale probabilities by inverse of class size if labels used during training (uniform chance of smapling each class)
    if supervised:
    
        y = dataset.get_labels()  # get datalabels

        # calculate probability of sampling each sample depending on class label
        class_sample_count = np.array([T.sum(y == t).item() for t in range(T.max(y).item() + 1)])
        class_weights = 1. / class_sample_count
        
        sample_class_weights = np.array([class_weights[t] for t in y]) # assing class weights to each sample
        sample_weights *= sample_class_weights  # scale sampling probabilities by inverse of class size


    # normalise probabilities and instantiate sampler
    sample_weights *= 1. / np.sum(sample_weights)  # divide by partition function to normalise sampling probabilities
    sample_weights = T.from_numpy(sample_weights)  # convert to tensor

    sampler = T.utils.data.WeightedRandomSampler(sample_weights.type('torch.DoubleTensor'), len(sample_weights))  #instantiate sampler
    
    mp.set_start_method('spawn', force = True)
    
    # instantiate and return dataloader
    dataloader = T.utils.data.DataLoader(dataset, batch_size = batch_size,sampler = sampler, num_workers = num_workers)
    return dataloader