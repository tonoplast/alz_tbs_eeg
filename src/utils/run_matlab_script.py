# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 08:15:47 2023

@author: sungw
"""

import subprocess
from pathlib import Path


def run_matlab_file(path_to_matlab_file: Path, 
                    keep_window_open: bool = False, 
                    run_concurrently: bool = False) -> None:
    '''
    runs matlab file using matlab commands (requires matlab license).
    
    Example:

    path_to_matlab_file_quoted should look like this: (NOTE: single quotes inside double quotes "'directory'")
    path_to_matlab_file_quoted = "'H:/GitHub/tram_protocol_eeg/src/preprocessing/preprocessing_RELAX.m'"
        
    '''
    
    if not isinstance(keep_window_open, bool):
        raise TypeError("Only booleans (True/False) are allowed")
    
    if not isinstance(run_concurrently, bool):
        raise TypeError("Only booleans (True/False) are allowed")
        
    path_to_matlab_file_quoted = f"'{path_to_matlab_file}'"
    
    if run_concurrently:
        wait = ''
    else:
        wait = '-wait'    
    
    # '-wait' allows it to run one after another
    if keep_window_open:
        run_matlab = f'matlab -nojvm -nosplash -nodesktop {wait} -r "try, run({path_to_matlab_file_quoted}), catch e, disp(getReport(e)), end;"'
    elif not keep_window_open:
        run_matlab = f'matlab -nojvm -nosplash -nodesktop {wait} -r "try, run({path_to_matlab_file_quoted}), catch e, disp(getReport(e)), exit(1), end, exit(0);"'

    subprocess.call(run_matlab, shell=0)
