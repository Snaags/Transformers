"""

Tokenising malnet images

created: 28/04/2023

"""


import numpy as np
from PIL import Image
import os
import random 

#preprocessing malnet
class malnet_dataloader():

    def __init__(self, dataset_path =
                 '~/data/malnet.cc.gatech.edu/image-data/full-data-as-6GB/malnet-images/',
                seed = time.time()):
        
        #store dataset path
        self.dataset_path = os.path.expanduser(path)
        
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
        


        return byte_string
    
    def random_sample(self, seed, dir_path: str, dir_size = 0):
        
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
                if rng.random() < sampling_probability: return os.path.join(root,file)  # return sample with uniform probability

                #sample file with proability 1/n
                if rng.randrange(num_files = 0): 
                    selected_sample = os.path.join(root, file)
                
                num_samples += 1 #increment sample counter 
        
       return selected_sample 

    #iterate over entire dataest one file at a time
    def iterate_datset(self):
        # iterate over dataset directory and sub directories
        for root, _, files in os.walk(self.path)
            for file in file:
                yield self.malnet_image_to_bytes(file)


from tokenizers import Tokenizer, pre_tokenizers, decoders, processors
from tokenizers.models import BPE
from tokenizers.trainer import BpeTrainer

#initalise dataloader
loader = malnet_dataloader()

#initalise BPE tokenizer
malnet_tokeniser = Tokenizer(BPE())

malnet_tokeniser.pre_tokenizers.ByteLevel(add_prefix_space = False, use_regex = False)
malnet_tokeniser.decoder = decoders.ByteLevel()
malnet_tokeniser.post_processor = processors.ByteLevel(trim_offsets = True)

trainer = BpeTrainer(
    vocab_size = 30000,  # Set the desired vocabulary size
    min_frequency = 2,
    show_progress= True,
    initial_alphabet = pre_tokenizers.ByteLevel.alphabet(),
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
)

malner_tokeniser.train_from_iterator(loader.iterate_datset(), trainer)





