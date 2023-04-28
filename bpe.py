"""

Tokenisers for byte data 

Created: 11/04/2032

"""

import numpy as np
from PIL import Image
import os

#preprocessing malnet
class malnet_dataloader():

    def __init__(self, dataset_path = '~/data/malnet.cc.gatech.edu/image-data/full-data-as-6GB/malnet-images/'):
        
        #store dataset path
        self.dataset_path = os.path.expanduser(path)


    #function to convert malware image to string of bytes
    def malnet_image_to_bytes(self, file_path):
        
        #open image using 
        img = Image.open(file_path)

        #convert to greyscale
        img_array = img.convert('L')

        #convert to numpy array and flatter
        return np.array(img_array).flatten()
    

    #iterate over entire dataest one file at a time
    def iterate_datset(self):
        # iterate over dataset directory and sub directories
        for root, _, files in os.walk(self.path)
            for file in file:
                yield self.malnet_image_to_bytes(file)



# Generic Tokeniser Class
class Tokeniser():

    def __init__(self):



# Tokeniser using Byter Pair Encodings
class BPE_Tokeniser(Tokeniser):

    def __init__(self):
        
        super(BPE_Tokeniser, self).__init__()



# Tokeniser using WordPiece
class WordPiece_Tokeniser(Tokeniser):
    
    def __init__(self):

        super(WordPiece_Tokeniser, self).__init__()



