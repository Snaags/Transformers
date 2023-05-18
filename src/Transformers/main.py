#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main scripts for running scripts specified in yaml config file

Created on Wed Apr  5 14:43:23 2023

@author: jack
"""

import yaml
import json
import subprocess
import os

def main():
    
    #read yaml config file
    test_path = os.path.expanduser('~/transformer_config.yaml')
    if os.path.isfile(test_path): #check for local config
        path = test_path
    else:   #otherwise use default
        path = 'config.yaml'
        
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    
    current_config = config['current_script']
    dataset_config = current_config['dataset']
    
    #run scripts
    for script_config in current_config['run_scripts']:
        #get script parameters
        script_name = script_config['name']
        script_params = script_config['parameters']
        script_params['gpus'] = current_config['gpus']
        script_params['processes_per_gpu'] = current_config['processes_per_gpu']
        script_params['cpu_processes'] = current_config['cpu_processes']
        script_params['device'] = current_config['device']
        
        #run script
        subprocess.run(['python3', script_name, '--dataset_config', json.dumps(dataset_config), '--script_config', json.dumps(script_params)])
        


if __name__ == '__main__':
    main()
    
