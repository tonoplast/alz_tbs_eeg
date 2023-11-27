# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 08:00:04 2023

@author: sungw
"""

from pathlib import Path
import json
from tqdm import tqdm
from src.utils import config_creator
from src.analysis.pac.run_pac import get_config_and_files_to_process, process_epoched_data
from src.analysis.pac.verification.get_epoch_numbers_and_freq_power import get_epoch_no_and_power

# =============================================================================
# ### VERY IMPORTANT TO CHANGE config_creator.py so that you get the right data extracted!
# =============================================================================

p = Path('.').resolve()
config_folder = p.joinpath('src', 'config')
pac_fixed_folder = fr"{p.joinpath('src', 'utils', 'pac_src_fixed')}"

config_creator.create_pipeline_step_config(config_folder)
config_creator.create_pac_config(config_folder)


# FREQ_RANGES = [[4, 8], [8, 13], [13, 30], [30, 39], [40, 46], [47, 53], [54, 60]]


pac_cfg, save_output, output_folder, data_folder, data_files, data_type = get_config_and_files_to_process()


df_list = []
for number_of_files_ran, data_file in enumerate(tqdm(data_files, desc="Getting info from Re-referenced EEG data", colour='green'), 1):
    filename, resting_epochs_reref = process_epoched_data(data_file, data_folder, data_type, pac_cfg)
    
    
    # epoch number and freq power for record
    epoch_no_and_freq_power = get_epoch_no_and_power(filename, 
                                                     resting_epochs_reref, 
                                                     freq_ranges = pac_cfg.get('freq_ranges_to_check_power'))

    
    df_list.append(epoch_no_and_freq_power)


with open(data_folder.joinpath("epoch_number.json"), "w") as f:
    json.dump(df_list, f)
    
    
    