'''

Script to download malnet samples from androozoo using sha256 hash, parse to
text file, and save

created on 06/05/23

'''

import os
import shutil
import gzip
import binascii
from androguard.core.bytecodes import apk
import requests
import multiprocessing
from tqdm import tqdm
import sys
import argparse
import json
import traceback

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
def binary_to_hex_string(binary_data):
        return binascii.hexlify(binary_data).decode('utf-8')


def write_large_string_to_file(file_path, large_string, chunk_size=1024 * 1024):
    with open(file_path, 'w', encoding='utf-8') as file:
        start_index = 0
        end_index = chunk_size
        
        while start_index < len(large_string):
            file.write(large_string[start_index:end_index])
            start_index += chunk_size
            end_index += chunk_size


#function to save hex string as .txt file 
def save_hex_strings_to_file(hex_string, file_path, compressed = True):
    
    #compress using gzip
    if compressed:
        with gzip.open(file_path, "wt") as file:
            file.write(hex_string)
    
    #save as raw txt file
    else:
        write_large_string_to_file(file_path, hex_string)
        
        '''
        with open(file_path, "w") as file:
            file.write(hex_string)
        '''

#function to downlaod and return hex string from sha256 hash                
def download_hex(apk_hash: str, api_key: str, temp_path = os.path.expanduser('~/temp/temp_apk.apk')):
    
    #download android apk from androzoo using previously defined function
    download_apk(apk_hash, temp_path, api_key)
    dex_files = extract_bytecode(temp_path)  #find dex files in apk
    
    #convert dex files to hex dumps
    hex_files = [binary_to_hex_string(x) for x in dex_files]
    
    #combine in case of multiple dex files
    hex_string = ''.join(hex_files)
    os.remove(temp_path)
                                                                
    return hex_string
                                                                    

#generator function for downloading hex files
def generate_hex_files(apk_hashes, api_key, temp_path = os.path.expanduser('~/temp_apks/temp_apk')):
    for apk_hash in apk_hashes:        
        yield download_hex(apk_hash,api_key, temp_path)


#worker for downloading dataset
def download_worker(args):
    try:
        #get details on sample for worker download
        family = args['family']
        malware_type = args['malware_type']
        apk_hash = args['apk_hash']
        ids = args['ids']
        save_dir = args['save_dir']
        api_key = args['api_key'] 


        #save hex string as txt file
        file_save_dir = f'{save_dir}/{family}/{malware_type}'

        if not os.path.exists(file_save_dir): os.makedirs(file_save_dir)
        if os.path.exists(f'{file_save_dir}/{apk_hash}.txt'): 
            if not args['overwrite']: 
                return 
            else:
                os.remove(f'{file_save_dir}/{apk_hash}.txt')
        
        #download apk and extract hex string
        hex_string = download_hex(apk_hash, api_key, temp_path =
                                  os.path.expanduser(f'~/temp/temp_{ids}.apk'))
                                                        
        print(len(hex_string))
        save_hex_strings_to_file(hex_string, f'{file_save_dir}/{apk_hash}.txt', compressed = args['compressed'])
        
        del hex_string

    except Exception as e:
        print( "exception occurred:")
        print(type(e))
        print(e.args)
        print(e)
                                
        exc_type, exc_value, exc_traceback = sys.exc_info()
        filename, line_number, function_name, _ = traceback.extract_tb(exc_traceback)[-1]      
        
        print("An exception occurred:")
        print(f"File: {filename}")
        print(f"Line: {line_number}")
        print(f"Type: {type(e)}")
        print(f"Message:{e}")
        sys.stdout.flush()
        return


#functino to read malnet splits and create corresponding folders of text files
def create_data_folder(save_dir, hashes = None, load_path = None, overwrite = False, api_key = None, compressed = False, processes = 10):
                                      
    if hashes is None and load_path is None: 
        raise ValueError('No data provided')
    
    elif load_path is not None:

        #iterate over load path and parse spk details
        with open(load_path, 'r') as f:
                                 
            samples = []                                              
            
            for i,line in enumerate(f):
                
                args = {}

                fields = line.strip().split('/')
                
                args['family'] = fields[0]
                args['malware_type'] = fields[1]
                args['apk_hash'] = fields[2]
                args['ids'] = i
                args['save_dir'] = save_dir
                args['compressed'] = compressed 
                args['api_key'] = api_key
                args['overwrite'] = overwrite

                samples.append(args)
                                                  
    #print(f'family: {fields[0]} |   type: {fields[1]}   |   id: {fields[2]}')

    #begin downloading dataset using multiprocessing
    with multiprocessing.Pool(processes) as p:
        with tqdm(total=len(samples)) as pbar:
            for _ in p.imap_unordered(download_worker, samples):
                pbar.update()
                                         

#main function
def main():
    
    # get configuration file from command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_config', type=str)
    parser.add_argument('-s', '--script_config', type=str)
    args = parser.parse_args()
    
    # Convert the JSON string back to a dictionary
    script_config = json.loads(args.script_config)
    num_processes = script_config['cpu_processes']
        
    if num_processes == -1: num_processes = 20  #androozoo permits up to 20 concurrent downloads
   
    #get config info
    split_path = script_config['split_path']
    download_path = script_config['download_path']
    api_key = script_config['api_key']
    overwrite = script_config['overwrite']

    #download train data
    if script_config['train']['download']:
        create_data_folder(os.path.expanduser(f'{download_path}/train'),
                           load_path = os.path.expanduser(f'{split_path}/train.txt'),
                           processes = num_processes, 
                           compressed = script_config['train']['compress'],
                           api_key = api_key,
                           overwrite = overwrite)
    
        print('--- Finished Train Set ---')
    
    #download val data
    if script_config['val']['download']:
        create_data_folder(os.path.expanduser(f'{download_path}/val'),
                           load_path = os.path.expanduser(f'{split_path}/val.txt'),
                           processes = num_processes, 
                           compressed = script_config['val']['compress'],
                           api_key = api_key,
                           overwrite = overwrite)

        print('--- Finished Val Set ---')
   
    #download test data
    if script_config['test']['download']:
        create_data_folder(os.path.expanduser(f'{download_path}/test'),
                           load_path = os.path.expanduser(f'{split_path}/test.txt'),
                           processes = num_processes, 
                           compressed = script_config['test']['compress'],
                           api_key = api_key,
                           overwrite = overwrite)
    
        print('--- Finished Test Set ---')

if __name__ == '__main__':
    main()
