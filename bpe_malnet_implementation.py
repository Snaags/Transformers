"""

Tokenising malnet images

created: 28/04/2023

"""


import numpy as np
from PIL import Image
import os
import random 
import time 
import binascii

#preprocessing malnet
class malnet_dataloader():

    def __init__(self, 
                 dataset_path = '/media/jack/hd/malnet_raw_bins/family/0.01/train',
                seed = time.time()):
        
        #store dataset path
        self.dataset_path = os.path.expanduser(dataset_path)
        
        self.rng = random.Random(seed)

    #function to convert malware image to string of bytes
    def malnet_image_to_bytes(self, file_path):
        
        #open image using 
        img = Image.open(file_path)

        #convert to greyscale
        img_array = img.convert('L')

        #convert to numpy array and flatter
        return np.array(img_array).flatten()
    
    def load_byte_file(self, file_path):
        with open(file_path, 'rb') as file:
            byte_string = file.read() 
        
        return byte_string
    
    def random_sample(self, seed, dir_size = 0):
        
        dir_path = self.dataset_path

        #initalise uniform sampling params (if dir size is known)
        sampling_probability = 0
        if dir_size > 0: sampling_probability = 1/dir_size

        #initalise reservoir sampling params
        num_samples = 0
        selected_sample = None

        #iterate over directory
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                
                # use uniform sampling if size known (rng.random() > 0)
                if self.rng.random() < sampling_probability: return os.path.join(root,file)  # return sample with uniform probability

                #sample file with proability 1/n
                if num_samples == 0: selected_sample = os.path.join(root,file)
                else:
                    if self.rng.randrange(0, num_samples) == 0: 
                        selected_sample = os.path.join(root, file)
                
                num_samples += 1 #increment sample counter 
        
        return selected_sample 

    #iterate over entire dataest one file at a time
    def iterate_dataset(self):
        for i in range(100): 
            yield binascii.hexlify(self.load_byte_file(self.random_sample(self.dataset_path))).decode('utf-8')

from tokenizers import Tokenizer, pre_tokenizers, decoders, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

#initalise dataloader
loader = malnet_dataloader()

#initalise BPE tokenizer
malnet_tokeniser = Tokenizer(BPE())

malnet_tokeniser.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False, use_regex = False)
malnet_tokeniser.decoder = decoders.ByteLevel()
malnet_tokeniser.post_processor = processors.ByteLevel(trim_offsets = True)

trainer = BpeTrainer(
    vocab_size = 512,  # Set the desired vocabulary size
    min_frequency = 2,
    show_progress= True,
    initial_alphabet = pre_tokenizers.ByteLevel.alphabet(),
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
)

malnet_tokeniser.train_from_iterator(loader.iterate_dataset(), trainer)

malnet_tokeniser.save(os.path.expanduser('~/malnet_tokeniser.json'))




