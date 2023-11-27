# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 08:15:47 2023

@author: sungw
"""

from pathlib import Path
import shutil
from src.utils import config_reader
from tqdm import tqdm

def move_files(source_dir, dest_dir):
    '''
    recursively moving data into designated folder
    '''

    src_dir = Path(source_dir)
    dst_dir = Path(dest_dir)

    # Create the destination directory if it doesn't exist
    if not dst_dir.exists():
        dst_dir.mkdir(parents=True)

    # Iterate over each item in the source directory
    for item in tqdm(src_dir.iterdir()):
        # Move the item to the destination directory
        item.rename(dst_dir / item.name)
    
    print(f'Moved files to {dst_dir}')
        
        
def copy_files(source_file, dest_file):
    '''
    copies files
    '''
    assert source_file.is_file(), f'{source_file} is not a file!'
    src_file = Path(source_file)
    dst_file = Path(dest_file)
    
    shutil.copy(src_file, dst_file)
    

def copy_files_and_folders(source_dir, dest_dir):
    '''
    Moving may not work if the folder is open in windows, so copying it
    '''
    # Set the source and destination folder paths
    src_folder = Path(source_dir)
    dst_folder = Path(dest_dir)
    
    # Remove the destination folder if it already exists
    if dst_folder.exists():
        shutil.rmtree(dst_folder)
    
    # Use shutil to copy the entire folder and its subfolders and files to the destination folder
    shutil.copytree(src_folder, dst_folder)
    


def run_move_files(use_default: bool = True):
    
    assert isinstance(use_default, bool), 'use_default must be either True or False'
    
    p = Path('.').resolve()
    config_folder = p.joinpath('src', 'config')

    pipeline_step_cfg = config_reader.read_yaml_file(config_folder.joinpath('config_move_copy_files.yaml'))
    
    if use_default:
        move_from_folder = f"{pipeline_step_cfg.get('which_drive_from')}/{pipeline_step_cfg.get('move_from_folder_default')}/"
        move_to_folder = f"{pipeline_step_cfg.get('which_drive_to')}/{pipeline_step_cfg.get('move_to_folder_default')}/"
    else:
        move_from_folder = f"{pipeline_step_cfg.get('which_drive_from')}/{pipeline_step_cfg.get('move_from_folder_user_defined')}/"
        move_to_folder = f"{pipeline_step_cfg.get('which_drive_to')}/{pipeline_step_cfg.get('move_to_folder_user_defined')}/"
    
    try:
        move_files(move_from_folder, move_to_folder)
    except:
        copy_files_and_folders(move_from_folder, move_to_folder)
        print("Tried moving the files but something went wrong. The folder might be in use (open). Copying instead.")
        

def run_copy_files():
    '''
    list of files to be copied across will be copied
    '''
    
    p = Path('.').resolve()
    config_folder = p.joinpath('src', 'config')

    copy_config = config_reader.read_yaml_file(config_folder.joinpath('config_move_copy_files.yaml'))
    list_of_files = copy_config.get('files_to_copy')
    
    for file in list_of_files:
        folder_path = copy_config.get('copy_from_folder')
        file_path = Path(folder_path).joinpath(file)
        
        copy_files(file_path, copy_config.get('copy_to_folder'))
        print(f'{file} copied across.')
        
    