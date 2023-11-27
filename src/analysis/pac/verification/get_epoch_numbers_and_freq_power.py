# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 08:44:46 2023

@author: sungw
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 12:05:08 2023

This is to extract epoch numbers and the power at different frequencies given a folder where data exists.
This may provide some insight into statistically differences between cleaning/pac methods

@author: sungw
"""

import asyncio
import json
import mne
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
from src.utils import config_reader

import nest_asyncio
nest_asyncio.apply()

# MAIN_FOLDER = r'D:\EEG_data_collected_today\processed_EEG'

MAIN_FOLDER = r'O:\\Cognitive Disorders\\Data Transfer\\ALZ TBS\\e. EEG data\\TGC\\processed_EEG\\'


p = Path('.').resolve()
config_folder = p.joinpath('src', 'config')
pac_cfg = config_reader.read_yaml_file(config_folder.joinpath('config_pac.yaml'))
FREQ_RANGES = pac_cfg.get('freq_ranges_to_check_power')


# def get_avg_psd_at_given_range(eeg_epochs : object, fmin: int, fmax: int, method: str = 'multitaper') -> float:
#     '''
#     calculating psd at the frequency of interest
#     This may be useful for looking at 50 Hz noise comparison when using zapline in the eeg preprocessing
#     '''
#     psd_of_interest = eeg_epochs.compute_psd(fmin=fmin, fmax=fmax, method='multitaper')
#     return np.mean(psd_of_interest, axis=0).mean()



def get_freq_range_str(freq_ranges: list[list[int,int]]) -> list[str]:
    pow_str_1 = [f'pow_{str(i[0]).zfill(2)}_' for i in (j for j in freq_ranges)]
    pow_str_2 = [f'{str(i[1]).zfill(2)}Hz' for i in (j for j in freq_ranges)]
    return [i + j for i, j in zip(pow_str_1, pow_str_2)]


def get_avg_psd_at_given_range(eeg_epochs : object, freq_ranges: list[list[int,int]],
                               fmin: int = 2, fmax: int = 80, 
                               method: str = 'multitaper') -> float:
    '''
    calculating psd at the frequency of interest
    This may be useful for looking at 50 Hz noise comparison when using zapline in the eeg preprocessing
    '''
    psd_of_interest = eeg_epochs.compute_psd(fmin=fmin, fmax=fmax, method='multitaper')
    
    power_by_freq_range = []

    for freq_range in freq_ranges:
        freq_mask = np.logical_and(psd_of_interest.freqs >= freq_range[0], psd_of_interest.freqs <= freq_range[1])
        freq_power = np.mean(psd_of_interest._data[:, :, freq_mask], axis=2)
        mean_freq_power_across_channels_and_epoch = np.mean(freq_power)
        power_by_freq_range.append(mean_freq_power_across_channels_and_epoch)
    
    return power_by_freq_range



def get_epoch_no_and_power(filename, resting_epochs, freq_ranges):
    epoch_number = resting_epochs._data.shape[0]
    
    # diciontary for filename and epoch number to combine later
    fn_epoch_no_dict =  {'filename': filename, 'epoch_number': epoch_number}

    
    # compute psd in the range
    # This may give us indication of how well zapline performed over 50Hz noise reduction
    power_by_freq_range = get_avg_psd_at_given_range(resting_epochs, freq_ranges = freq_ranges)
    
    # dictionary for power
    output_str = get_freq_range_str(freq_ranges)
    power_dict = {k:v for k, v in zip(output_str, power_by_freq_range)}
    return {**fn_epoch_no_dict, **power_dict}


    
async def read_data_file(data_file: Path, data_folder: str, freq_ranges: list[list[int,int]]):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # get the file name (stem without extension)
        filename = data_file.relative_to(data_folder).stem
    
        resting_epochs = await asyncio.to_thread(mne.io.read_epochs_eeglab, data_file, verbose=False)
        # epoch_number = resting_epochs._data.shape[0]
        
        # # diciontary for filename and epoch number to combine later
        # fn_epoch_no_dict =  {'filename': filename, 'epoch_number': epoch_number}

        
        # # compute psd in the range
        # # This may give us indication of how well zapline performed over 50Hz noise reduction
        # power_by_freq_range = get_avg_psd_at_given_range(resting_epochs, freq_ranges = freq_ranges)
        
        # # dictionary for power
        # output_str = get_freq_range_str(freq_ranges)
        # power_dict = {k:v for k, v in zip(output_str, power_by_freq_range)}
        
        return get_epoch_no_and_power(filename, resting_epochs, freq_ranges)

    
       
            
async def process_files(sorted_files: Path, data_folder: str):
    if sorted_files:
        tasks = []
        for number_of_files_ran, data_file in enumerate(sorted_files, 1):
            tasks.append(asyncio.create_task(read_data_file(data_file, data_folder, FREQ_RANGES)))
    
        df_list = []
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Loading mne epoch file", colour='green', position=0, leave=True):
            epoch_number_dict = await task
            df_list.append(epoch_number_dict)
    
        with open(data_folder.joinpath("epoch_number.json"), "w") as f:
            json.dump(df_list, f)
    
    else:
        df_list = []

    return df_list


async def main():
    folder = MAIN_FOLDER
    # highpass_subfolders = [subfolder for subfolder in Path(folder).iterdir() if subfolder.is_dir() and 'HighPass' in subfolder.name]
    # highpass_subfolders = [subfolder for subfolder in Path(folder).iterdir() if subfolder.is_dir() and '025Hz_HighPass_MWF_01lostep_2bw' in subfolder.name]
    
    # exact_match = ['1Hz_HighPass_MWF', '025Hz_HighPass_MWF']
    exact_match = ['1Hz_MWF', '1Hz_NoMWF', '025Hz_MWF', '025Hz_NoMWF']
    # import pdb; pdb.set_trace()
    partial_match = ['18s', 'Zapline', 'NoMWF'] # replace with your desired list of strings
    subfolders = [subfolder for subfolder in Path(folder).iterdir() if subfolder.is_dir() \
                  and ((any(match_str in subfolder.name for match_str in partial_match)) \
                   or (subfolder.name in exact_match))]

    for subfolder in tqdm(subfolders, desc="Doing each folder", colour='magenta', position=0, leave=True):
        # data_folder = subfolder.joinpath(r'RELAXProcessed\Cleaned_Data\Epoched\eoec')
        data_folder = subfolder.joinpath(r'RELAXProcessed\Cleaned_Data\Epoched')
        
        # get data in the folder and sort by the latest date. Only looks at .set files
        input_files = [f for f in data_folder.glob('*.set') if f.is_file()]
        if input_files:
            sorted_files = sorted(input_files, key=lambda x: x.stat().st_mtime)
        else:
            sorted_files = []
            print('no files available')

        await process_files(sorted_files, data_folder)


if __name__ == '__main__':
    asyncio.run(main())