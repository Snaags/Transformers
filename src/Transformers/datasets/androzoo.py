#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:25:21 2023

@author: jack
"""


import requests
import os
import gzip
import binascii
from androguard.core.bytecodes import apk


#function to downlaod apk from androzoo using sha256 has and api key
def download_apk(apk_hash: str, download_path: str, api_key: str):
    
    # define apk download url and get
    url = f" https://androzoo.uni.lu/api/download?apikey={api_key}&sha256={apk_hash}"
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        #if file found download to specified path
        with open(download_path, 'wb') as apk_file: 
            for chunk in response.iter_content(chunk_size=8192):
                apk_file.write(chunk)
        

    # if file not been downloaded correctly
    else:
        print(f"Error downloading APK. Status code: {response.status_code}")
                     

#function to extract raw bytecode from apk file
def extract_bytecode(apk_path: str):

    # Load the APK file
    a = apk.APK(apk_path)

    # Get the DEX files
    dex_files = a.get_all_dex()

    # Convert the generator to a list and return the raw bytecode
    return list(dex_files)
                     

#function to convert binary formatted data to string of hexadecimal representation
def binary_to_hex(binary_data, use_format = 'byte_string'):
        #hex_bytes = binascii.hexlify(binary_data)
        hex_bytes = binary_data

        if use_format == 'text': 
            return binascii.hexlify(hex_bytes).decode('utf-8')
        elif use_format == 'byte_string':
            return hex_bytes
        else:
            raise ValueError('ERROR::: Invalid Format For Parsing Binary')

def write_large_string_to_file(file_path, large_string, chunk_size=1024 * 1024):
    with open(file_path, 'w', encoding='utf-8') as file:
        start_index = 0
        end_index = chunk_size
        
        while start_index < len(large_string):
            file.write(large_string[start_index:end_index])
            start_index += chunk_size
            end_index += chunk_size


#function to save hex string as .txt file 
def save_hex_strings_to_file(hex_string, file_path, save_format = 'byte_string'):
     
    #save as byte string
    if save_format == 'byte_string':
        with open(f'{file_path}.bin', 'wb') as file:
            file.write(hex_string)

    #compress using gzip
    elif save_format == 'compressed':
        with gzip.open(f'{file_path}.txt', "wt") as file:
            file.write(hex_string)
    
    #save as raw txt file
    elif save_format == 'text':
        write_large_string_to_file(f'{file_path}.txt', hex_string)
        
        '''
        with open(file_path, "w") as file:
            file.write(hex_string)
        '''
    else:
        raise ValueError('ERROR::: Invalid hex save format specified')

#function to downlaod and return hex string from sha256 hash                
def download_hex(apk_hash: str, api_key: str, use_format = 'byte_string', temp_path = os.path.expanduser('~/temp/temp_apk.apk')):
    
    #download android apk from androzoo using previously defined function
    download_apk(apk_hash, temp_path, api_key)
    dex_files = extract_bytecode(temp_path)  #find dex files in apk

    #convert dex files to hex dumps
    hex_files = [binary_to_hex(x, use_format = use_format) for x in dex_files]
    
    #combine in case of multiple dex files
    if use_format == 'byte_string':
        hex_string = b''.join(hex_files)
    else:
        hex_string = ''.join(hex_files)
    os.remove(temp_path)
                                                                
    return hex_string
                                                                    

#generator function for downloading hex files
def generate_hex_files(apk_hashes, api_key, temp_path = os.path.expanduser('~/temp_apks/temp_apk')):
    for apk_hash in apk_hashes:        
        yield download_hex(apk_hash,api_key, temp_path)

