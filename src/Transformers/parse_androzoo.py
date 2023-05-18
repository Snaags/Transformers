'''

Script to download malnet samples from androozoo using sha256 hash, parse to
text file, and save

created on 06/05/23

'''

import os
import multiprocessing
from tqdm import tqdm
import sys
import argparse
import json
import traceback
from datasets.androzoo import download_hex, save_hex_strings_to_file

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
        
        file_save_path = f'{file_save_dir}/{apk_hash}'
        
        if os.path.exists(f'{file_save_dir}/{apk_hash}.txt') and args['overwrite']:
            os.remove(f'{file_save_dir}.txt')
        
        elif os.path.exists(f'{file_save_path}.bin') and args['overwrite']: 
            os.remove(f'{file_save_path}.bin')
        
        elif (os.path.exists(f'{file_save_path}.bin') or os.path.exists(f'{file_save_path}.txt')) and (not args['overwrite']):
            return
        

        #download apk and extract hex string
        hex_string = download_hex(apk_hash, api_key, 
                                  use_format = args['save_format'], 
                                  temp_path = os.path.expanduser(f'~/temp/temp_{ids}.apk'))

        save_hex_strings_to_file(hex_string, f'{file_save_dir}/{apk_hash}',
                                 save_format = args['save_format'])
        
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



#functino to read malnet splits and create corresponding folders of text files
def create_data_folder(save_dir, hashes = None, load_path = None, overwrite =
                       False, api_key = None, save_format = 'save_format', processes = 10):
                                      
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
                args['compressed'] = save_format
                args['api_key'] = api_key
                args['overwrite'] = overwrite
                args['save_format'] = save_format
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
                           save_format = script_config['train']['save_format'],
                           api_key = api_key,
                           overwrite = overwrite)
    
        print('--- Finished Train Set ---')
    
    #download val data
    if script_config['val']['download']:
        create_data_folder(os.path.expanduser(f'{download_path}/val'),
                           load_path = os.path.expanduser(f'{split_path}/val.txt'),
                           processes = num_processes, 
                           save_format = script_config['val']['save_format'],
                           api_key = api_key,
                           overwrite = overwrite)

        print('--- Finished Val Set ---')
   
    #download test data
    if script_config['test']['download']:
        create_data_folder(os.path.expanduser(f'{download_path}/test'),
                           load_path = os.path.expanduser(f'{split_path}/test.txt'),
                           processes = num_processes, 
                           save_format = script_config['test']['save_format'],
                           api_key = api_key,
                           overwrite = overwrite)
    
        print('--- Finished Test Set ---')

if __name__ == '__main__':
    main()
