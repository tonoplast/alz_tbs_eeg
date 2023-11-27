# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 14:26:41 2023

@author: sungw
"""

import yaml
import json

def read_yaml_file(yaml_file_path):
    '''
    turn yaml into simple dictionary (inner dictionary)
    '''
    # Load YAML configuration file
    with open(yaml_file_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Create variables from inner dictionary keys and values, excluding _comment
    if isinstance(config, dict):
        # if config is a dictionary
        cfg = {k: v for k, v in config.items() if k != '_comment'}
    else:
        # if config is a list
        cfg = {k: v for d in config[0].values() for k, v in d.items() 
               if d.get('_comment') is None or '_comment' not in k}


    
    return cfg


def read_json_cfg(file_path):
    with open(file_path, 'r') as f:
      return json.load(f)


def overwrite_json_cfg(my_dict, file_path):
    with open(file_path, 'w') as f:
        json.dump(my_dict, f)

