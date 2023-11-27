# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 08:43:41 2023

@author: sungw
"""

from pathlib import Path
from src.utils import run_matlab_script
from src.utils import config_creator, config_reader
import time

## overwrite these configs (so that I can run multiple matlab)
reverse_order = False
lo_bandwidth = 2
eye_status_eo = ['eyes_open']
eye_status_ec = ['eyes_closed']
how_many_permutations = 5


overwrite_dict_eo = {
                    'reverse_order': reverse_order, 
                    'eye_status': eye_status_eo, 
                    'lo_bandwidth': lo_bandwidth,
                    'how_many_permutations': how_many_permutations,
                    'output_filename_tag': '',
                     }

overwrite_dict_ec = {
                    'reverse_order': reverse_order, 
                    'eye_status': eye_status_ec, 
                    'lo_bandwidth': lo_bandwidth,
                    'how_many_permutations': how_many_permutations,
                    'output_filename_tag': ''
                     }


def overwrite_cfg(matlab_config, **kwargs):
    for k, v in kwargs.items():
        if isinstance(v, dict):
            if k in matlab_config:
                overwrite_cfg(matlab_config[k], **v)
        else:
            if k in matlab_config:                
                matlab_config[k] = v

    config_reader.overwrite_json_cfg(my_dict = matlab_config, file_path = matlab_config_file)
    
    pac_matlab = p.joinpath('src', 'analysis', 'pac', 'pac_glmf_matlab.m')
    
    return pac_matlab
       


p = Path('.').resolve()
config_folder = p.joinpath('src', 'config')
pac_fixed_folder = fr"{p.joinpath('src', 'utils', 'pac_src_fixed')}"

## create config, load config, and overwrite... Strange I know.
config_creator.create_matlab_config(config_folder)

matlab_config_file = config_folder.joinpath('config_matlab.json')
matlab_config = config_reader.read_json_cfg(file_path = matlab_config_file)



def run():

    # # =============================================================================
    # # ## running eyes open
    # # =============================================================================
    
    pac_matlab_eo = overwrite_cfg(matlab_config = matlab_config, pac = overwrite_dict_eo)
    run_matlab_script.run_matlab_file(pac_matlab_eo, keep_window_open=True, run_concurrently=True)

    
    time.sleep(60) # this is to give a bit of space so that it doesn't use the same config for both eo and ec
    
    # # =============================================================================
    # # ## running eyes closed
    # # =============================================================================
    
    
    pac_matlab_ec = overwrite_cfg(matlab_config = matlab_config, pac = overwrite_dict_ec)
    run_matlab_script.run_matlab_file(pac_matlab_ec, keep_window_open=True, run_concurrently=True)



if __name__ == "__main__":
    run()

