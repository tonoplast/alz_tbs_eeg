# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 21:06:47 2023

@author: sungw
"""
from pathlib import Path
from src.analysis.pac.run_pac import run_pac
from src.utils import run_matlab_script
import warnings
from src.utils import config_creator, config_reader, move_files


def main():
    
    p = Path('.').resolve()
    config_folder = p.joinpath('src', 'config')
    pac_fixed_folder = fr"{p.joinpath('src', 'utils', 'pac_src_fixed')}"
    
    try:
        pipeline_step_cfg = config_reader.read_yaml_file(config_folder.joinpath('config_pipeline_step.yaml'))
    except:
        print('"config_pipeline_step.yaml" may not exist in the config folder' )

    if pipeline_step_cfg.get('overwrite_config'):
        config_creator.create_pipeline_step_config(config_folder)
        config_creator.create_move_copy_file_config(config_folder, pac_fixed_folder)
        config_creator.create_matlab_config(config_folder)
        config_creator.create_pac_config(config_folder)
        # if overwrite, then it needs to apply as well
        pipeline_step_cfg = config_reader.read_yaml_file(config_folder.joinpath('config_pipeline_step.yaml'))

    keep_windows_open = pipeline_step_cfg.get('keep_window_open')    
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
                        
        # =============================================================================
        #         # Copying fixed PAC source code into the package code
        # =============================================================================
        
        if pipeline_step_cfg.get('run_copy_fixed_pac_file'):
            move_files.run_copy_files()
            print('Finished copying fixed tensorpac source code.')
            
        
        # =============================================================================
        #         # running preprocessing (MATLAB)
        # =============================================================================
        
        # .cnt -> .set, select channel, downsample if needed
        if pipeline_step_cfg.get('run_set_conversion'):
            path_to_relax_preparator = p.joinpath('src', 'preprocessing', 'RELAX_preparator.m')
            run_matlab_script.run_matlab_file(path_to_relax_preparator, keep_window_open=keep_windows_open, run_concurrently=False)
            print('Converting native EEG file to EEGLAB format finished.')
        
            
        # preprocessing using RELAX
        if pipeline_step_cfg.get('run_relax_preprocessing'):
            path_to_relax_preprocessing = p.joinpath('src', 'preprocessing', 'preprocessing_RELAX.m')
            run_matlab_script.run_matlab_file(path_to_relax_preprocessing, keep_window_open=keep_windows_open, run_concurrently=False)
            print('RELAX preprocessing finished.')
            

        # epoching using RELAX
        if pipeline_step_cfg.get('run_relax_epoching'):
            path_to_relax_epoching = p.joinpath('src', 'processing', 'processing_RELAX.m')
            run_matlab_script.run_matlab_file(path_to_relax_epoching, keep_window_open=keep_windows_open, run_concurrently=False)
            print('RELAX epoching finished.')
        
        
        # split files into eo/ec
        if pipeline_step_cfg.get('run_file_splitting'):
            path_to_split_eoec = p.joinpath('src', 'processing', 'processing_split_eoec.m')
            run_matlab_script.run_matlab_file(path_to_split_eoec, keep_window_open=keep_windows_open, run_concurrently=False)
            print('eo ec split finished (if any).')
        
            print('MATLAB processes finished')
        
        # =============================================================================
        #         # running pac (Python)
        # =============================================================================
        if pipeline_step_cfg.get('run_pac_analysis'):
            run_pac(output_filename_tag='')
            print('PAC analysis finished.')
        
        # =============================================================================
        #         # Move all data into today date time folder
        # =============================================================================
        if pipeline_step_cfg.get('run_moving_data'):
            move_files.run_move_files(use_default = pipeline_step_cfg.get('use_default_output_folder'))
            print('File moving/copying finished.')
                    


if __name__ == "__main__":
    main()