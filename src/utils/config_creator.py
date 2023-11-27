# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 14:09:27 2023

@author: WookS
"""

import json
import yaml
from pathlib import Path
from datetime import datetime

WHICH_DRIVE_FROM = 'O:' # 'O:' 'D:'
WHICH_DRIVE_TO = 'O:' # 'O:' 'D:'
STARTING_FOLDER = r'CogTx\\DataTransfer\\ALZ TBS\\e. EEG data\\TGC' ## TBS ALZ theta-gamma coupling
# STARTING_FOLDER = r'CogTx\\DataTransfer\\tACS ALZ\\d. Data\\1. EEG data\\resting' ## tacs ALZ resting
ANALYSIS_FOLDER_RESTING_EEG = r'CogTx\\DataTransfer\\ALZ TBS\\e. EEG data\\Analysis2021\\RELAX_Epoched_ALL'
OVERWRITE_CONFIG = True
KEEP_WINDOW_OPEN = True
SHOW_EEG_TRACE = False
OPEN_OUTPUT_FOLDER = False
NO_OF_PERMUTATIONS = 2500 # 2500
PROJ_EEGSYSTEM = "ALZ_NEUROSCAN" # ALZ_BIOSEMI / ALZ_NEUROSCAN / OTHER_NEUROSCAN / ALZ_BRAINVISION

def create_matlab_config(save_path: Path) -> json:
    '''
    Modify this prior to running so that it will create appropriate
    configuration file for eeg data preprocessing (MATLAB)
    '''
    # =============================================================================
    # ### Preparation ###
    # =============================================================================
    eeglab_version = 'eeglab2022.1'
    
    data_drive = WHICH_DRIVE_FROM # your external hard drive
    starting_folder = STARTING_FOLDER
    analysis_folder_resting_eeg = ANALYSIS_FOLDER_RESTING_EEG
    set_srate_preprocessing = 256  # We'll stick with 256 so that we don't have to add more steps inbetween. 256 is also good for zapline.
    proj_eegsystem = PROJ_EEGSYSTEM
    
    ## Different system settings/montage
    if proj_eegsystem == 'OTHER_NEUROSCAN':
        channels_to_keep_preprocessing =  ['AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8', 
                                            'FC5','FC3','FC1','FCZ','FC2','FC4','FC6','T7',
                                            'C5','C3','C1','CZ','C2','C4','C6', 'T8','CP3','CP1','CP2','CP4',
                                            'P7','P5','P3','P1','PZ','P2','P4','P6','P8', 'PO3','POZ','PO4','O1','OZ','O2']
        file_extension_preprocessing = '.cnt'

    
    elif proj_eegsystem == 'ALZ_NEUROSCAN':
        # Monash Alz
        channels_to_keep_preprocessing =  ['AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8', 
                                            'FC5','FC3','FC1','FCZ','FC2','FC4','FC6',
                                            'C5','C3','C1','CZ','C2','C4','C6',
                                            'P7','P5','P3','P1','PZ','P2','P4','P6','P8', 'PO3','POZ','PO4','O1','OZ','O2']   
        file_extension_preprocessing = '.cnt'

    
    elif proj_eegsystem == 'ALZ_BIOSEMI':
        # biosemi montage
        channels_to_keep_preprocessing = ["Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1", "C1", "C3",
                                          "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7", "P9", "PO7", "PO3", "O1",
                                          "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2", "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6",
                                          "F8", "FT8", "FC6", "FC4", "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", 
                                          "CP2", "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2", ]
        file_extension_preprocessing = '.bdf'
    
    elif proj_eegsystem == 'ALZ_BRAINVISION':
        # brain vision (?) montage
        channels_to_keep_preprocessing =  [
                                            'Fp1', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8',
                                            'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7',
                                            'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6',
                                            'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2', 'FCz', 'Fz',
                                        ]
        
        file_extension_preprocessing = '.vhdr'

    
    caploc_filename = 'standard-10-5-cap385.elp'
    

    # =============================================================================
    # ### RELAX ###
    # =============================================================================
    # RELAX preprocessing
    high_pass_filter: float = 0.25 # .25 / 1
    low_pass_filter: int = 80
    line_noise_frequency: int = 50
    use_mwf_to_clean_muscle: bool = False # False # True
    use_mwf_to_clean_blinks: bool = False # False # True
    use_mwf_to_clean_heog: bool = False # False # True
    notch_filter_type: str = 'ZaplinePlus' # Butterworth / ZaplinePlus
    ica_method: str = 'fastica_symm' # fastica_symm / extended_informax_ICA / amica / cudaica
    perform_wica_on_iclabel: int = 1 # 1:yes, 2:no
    perform_ica_subtract: bool = False
    
    # RELAX processing / epoching
    data_type: str = 'Resting'
    baseline_preiod: list[int, int] = [-6000, 6000] 
    period_to_epoch: list[int, int] = [-6, 6]  #
    resting_data_trigger_interval: int = 3 #marking at X s interval (every 3 sec, marks, which means 75% overlap)
    baseline_correction_method: int = 'none' # we do baseline correction later before PAC
    
    # Epoch rejection settings (these seem to work okay)
    single_channel_improbable_data_threshold: int  = 5 # 6 / default == 5
    all_channel_improbable_data_threshold: int = 3 # 4 / default == 3
    single_channel_kurtosis_threshold: int = 5 # 6 / default == 5
    all_channel_kurtosis_threshold: int = 3 # 4 / default == 3
    reject_amp: int = 100 # default == 60 this can remove all epochs, leaving nothing #100
    muscle_slope_threshold: float = -0.31 # default = -0.31
    max_proportion_of_muscle_epochs_to_clean: float = 0.5 # default 0.5
    
    # =============================================================================
    # ### Post-processing ###
    # =============================================================================
    # Split eo ec
    # set_srate_processing = 256 # downsampling (only if raw data)
    file_extension_processing: str = '.set' # file extention to run
    channels_to_remove_processing: list[str] = [''] # make sure it is either empty list or list
    rereference_without_these_electrodes_processing: list[str] = [''] # make sure it is either empty list or list
    split_eo_ec: bool = True # if the data already contains eo and ec together


    # =============================================================================
    # ### PAC (glmf / MATLAB) ### NOT USED FOR TRIAL
    # =============================================================================
    lo_bounds: list[int, int] = [3, 9]
    lo_step: float = 0.1
    lo_bandwidth: float = 2
    hi_bounds: list[int, int] = [20, 70]
    hi_step: int = 1
    hi_bandwidth: str = 'adaptive'
    frontchan: list['str'] = 'FZ' 
    backchan: list['str'] = 'PZ'
    reverse_order: bool = False
    eye_status: list[str, str] = ['eyes_open','eyes_closed']
    how_many_permutations: int = 5
    output_filename_tag: str = ''

    
    # =============================================================================
    # ### config dict ###
    # =============================================================================
    # Create a dictionary to store the configuration
    # (DO NOT CHANGE THIS)
    config = {
            'preparation': {
                'eeglab_version': eeglab_version,
                'data_drive': data_drive,
                'set_srate': set_srate_preprocessing,
                'proj_eegsystem': PROJ_EEGSYSTEM,
                'file_extension': file_extension_preprocessing,
                'channels_to_keep': channels_to_keep_preprocessing,
                'caploc': caploc_filename,
                'starting_folder': starting_folder
                },
            

            'relax_preprocessing': {
                'eeglab_version': eeglab_version,
                'data_drive': data_drive,
                'caploc': caploc_filename,
                'HighPassFilter': high_pass_filter,
                'LowPassFilter': low_pass_filter ,
                'LineNoiseFrequency': line_noise_frequency,
                'LineNoiseFilterType': notch_filter_type,
                'DoMWFOnce': use_mwf_to_clean_muscle,
                'DoMWFTwice': use_mwf_to_clean_blinks,
                'DoMWFThrice': use_mwf_to_clean_heog,
                'ica_method': ica_method,
                'starting_folder': starting_folder,
                'perform_wica_on_iclabel': perform_wica_on_iclabel,
                'perform_ica_subtract': perform_ica_subtract,
                },
                        
            
            'relax_processing': {
                'eeglab_version': eeglab_version,
                'data_drive': data_drive,
                'data_type': data_type,
                'baseline_preiod': baseline_preiod,
                'period_to_epoch': period_to_epoch,
                'resting_data_trigger_interval': resting_data_trigger_interval,
                'baseline_correction_method': baseline_correction_method,
                'single_channel_improbable_data_threshold': single_channel_improbable_data_threshold,
                'all_channel_improbable_data_threshold': all_channel_improbable_data_threshold,
                'single_channel_kurtosis_threshold': single_channel_kurtosis_threshold,
                'all_channel_kurtosis_threshold': all_channel_kurtosis_threshold,
                'reject_amp': reject_amp,
                'muscle_slope_threshold': muscle_slope_threshold,
                'max_proportion_of_muscle_epochs_to_clean': max_proportion_of_muscle_epochs_to_clean,
                'starting_folder': starting_folder
                },           
            
            
            'postprocessing': {
                'eeglab_version': eeglab_version,
                'data_drive': data_drive,
                # 'set_srate': set_srate_processing ,
                'file_extension': file_extension_processing,
                'channels_to_remove': channels_to_remove_processing,
                'rereference_without_these_electrodes': rereference_without_these_electrodes_processing,
                'split_eo_ec': split_eo_ec,
                'starting_folder': starting_folder
                 },
            
            'pac': {
                    'lo_bounds': lo_bounds,
                    'lo_step': lo_step,
                    'lo_bandwidth': lo_bandwidth,
                    'hi_bounds': hi_bounds,
                    'hi_step': hi_step,
                    'hi_bandwidth': hi_bandwidth,
                    'frontchan': frontchan,
                    'backchan': backchan,
                    'reverse_order': reverse_order,
                    'eye_status': eye_status,
                    'how_many_permutations': how_many_permutations,
                    'output_filename_tag': output_filename_tag,
                 },
            
            'analysis_resting_eeg' :{
                'eeglab_version': eeglab_version,
                'data_drive': data_drive,
                'caploc': caploc_filename,
                'analysis_folder': analysis_folder_resting_eeg,
                
                }
            
            }

    # Save the configuration as a json file
    with open(save_path.joinpath('config_matlab.json'), 'w') as f:
        json.dump(config, f)



def create_pac_config(save_path: Path) -> yaml:
    '''
    Modify this prior to running so that it will create appropriate
    configuration file for PAC analysis
    '''
    
    # =============================================================================
    #     # data settings
    # =============================================================================
    which_drive_analysis: str = WHICH_DRIVE_FROM
    starting_folder: str = STARTING_FOLDER
    input_folder: str = rf'/{starting_folder}/processed_EEG/RELAXProcessed/Cleaned_Data/Epoched/'
    output_folder: str = rf'/{starting_folder}/processed_EEG/RELAXProcessed/Cleaned_Data/Epoched/results_pac'      
    data_type: str = 'epoched' # epoched / raw
    # set_srate_pac = 256 ## only applied when continuous data // decided to remove since we'll just feed in downsampled data from the beginning
    save_output: bool = True
    number_of_files_to_process: int = '' ## default should be '' : empty string
    reverse_order: bool = False # order of data to be processed
    open_output_folder: bool = OPEN_OUTPUT_FOLDER # open the folder so that user can see the output

    # =============================================================================
    #     # pre-PAC settings (on EEG data)
    # ============================================================================
    average_reref = True
    laplacian_transform = True
    baseline_correction = True
    

    # =============================================================================
    #     # epoch variables (if not already epoched before PAC)
    # =============================================================================
    tmin: int = -6
    tmax: int = 6
    overlap: int = 9 # 75% overlap

    # =============================================================================
    #     # frequency of interest
    # =============================================================================
    freqs_of_interest: list[str,str] = ['theta', 'gamma']
    modulating_freq_range: list[int, int] = [4, 8]
    modulated_freq_range: list[int, int] = [30, 60]
    
    if 'neuroscan' in PROJ_EEGSYSTEM.lower():
        # better to do single electrodes than the average of them
        modulating_channels: list[str] = ['FZ'] #['FZ'] #['F1','FZ','F2']
        modulated_channels: list[str] = ['PZ']  #['PZ'] #['P1','PZ','P2']
    elif 'biosemi' in PROJ_EEGSYSTEM.lower():
        # better to do single electrodes than the average of them
        modulating_channels: list[str] = ['Fz'] #['FZ'] #['F1','FZ','F2']
        modulated_channels: list[str] = ['Pz']  #['PZ'] #['P1','PZ','P2']
        
    elif 'brainvision' in PROJ_EEGSYSTEM.lower():
        # better to do single electrodes than the average of them
        modulating_channels: list[str] = ['Fz'] #['FZ'] #['F1','FZ','F2']
        modulated_channels: list[str] = ['Pz']  #['PZ'] #['P1','PZ','P2']
    # =============================================================================
    #     # filter width/step
    # =============================================================================
    lo_bandwidth: float = 1 # 1 / 2
    lo_step: float = 1 # 1 / .1
    hi_bandwidth: float = 1
    hi_step: float = 1

    # =============================================================================
    #     # peak detection
    # =============================================================================
    # minimum distance required for peak detection
    # This shoud change based on the filter step (3 was good for lo_step was .1)
    # Might need to go to 1 or 2 depending (when lo_step is 1)
    minimum_distance: int = 1
    central_tendency: list[str] = ['mean', 'median']
    
    # =============================================================================
    #     # Other filter settings
    # =============================================================================
    # remove edge effect (500 ms)
    edge_length: float = .5 
    # power spectrum range for visualisation
    f_min_max: list[int, int] = [2, 80]


    # =============================================================================
    # # stats
    # =============================================================================
    # pvalue threshold for PAC
    pvalue_threshold: float = .05
    # multiple comparison procedures
    mcp: str = 'maxstat' ## 'maxstat' or maybe try 'fdr'? ('bonferroni' will be too harsh)
    # number of permutation for surrogate
    no_of_permutations: int = NO_OF_PERMUTATIONS

    # =============================================================================
    #     # PAC settings
    # =============================================================================
    # filter method    
    dcomplex: str = 'hilbert' # hilbert or wavelet
    # cycle for hilbert
    hilbert_cycle: tuple(int, int) = (3, 6)
    # width for wavelet
    wavelet_width: int = 7
    # which PAC to use, surrogate and normalisation method
    idpac: tuple(int, int, int) = (6, 1, 4) # MI == (2, 1, 4) / gcPAC == (6, 1, 4)
    # seed used for pac
    random_state: int = 0
    
    # =============================================================================
    #     # plot settings
    # =============================================================================
    # this is used to skip plotting eeg trace / power spectral density / topoplot
    plot_eeg_trace: bool = SHOW_EEG_TRACE
    freq_ranges_to_check_power: list[list[int, int]] = [[4, 8], [8, 13], [13, 30], [30, 39], [40, 46], [47, 53], [54, 60]]


    # yaml format (DO NOT CHANGE THIS)
    config = [
                {
                'data_settings':
                    {'which_drive': which_drive_analysis,
                     'input_folder': input_folder,
                     'output_folder': output_folder,
                     'data_type' : data_type,
                     # 'set_srate': set_srate_pac,
                     'save_output': save_output,
                     'number_of_files_to_process': number_of_files_to_process,
                     'reverse_order': reverse_order,
                     'open_output_folder': open_output_folder,
                     '_comment': 'The drive of the data to be analysed, path and whether save output or not, and the output folder',
                     },
                    
                'pre_pac_settings':
                    {'average_reref': average_reref,
                     'laplacian_transform': laplacian_transform,
                     'baseline_correction': baseline_correction,
                     '_comment': 'Settings before PAC - rereferencing',
                    },  
                    
                'epoch_variable':
                    {'tmin': tmin,
                    'tmax': tmax,
                    'overlap': overlap,
                    '_comment': 'Epoch parameters',
                    },
                
                'frequency_of_interest':
                    
                    {'freqs_of_interest': freqs_of_interest,
                    'modulating_freq_range': modulating_freq_range,
                    'modulated_freq_range': modulated_freq_range,
                    'modulating_channels': modulating_channels,
                    'modulated_channels': modulated_channels,
                    '_comment': 'Frequency parameters',
                    },
                
                'filter_width_step':
                    
                    {'lo_bandwidth': lo_bandwidth,
                    'lo_step': lo_step,
                    'hi_bandwidth': hi_bandwidth,
                    'hi_step': hi_step,
                    '_comment': 'Filter parameters',
                    },
                        
                     
                'other_filter_settings':
                    
                    {'edge_length': edge_length,
                    'f_min_max': f_min_max,
                    '_comment': 'Other filter parameters',
                    },
                    
                
                'pac_settings':
                    
                    {'dcomplex': dcomplex,
                    'idpac': idpac,
                    'random_state': random_state,
                    'hilbert_cycle': hilbert_cycle,
                    'wavelet_width': wavelet_width,
                    '_comment': 'PAC parameters',
                    },
                    
                'stats':
                    
                    {'pvalue_threshold': pvalue_threshold,
                    'mcp': mcp,
                    'no_of_permutations': no_of_permutations,
                    '_comment': 'Statistical parameters',
                    },      
                      
                'peak_detection':
                    
                    {'minimum_distance': minimum_distance,
                     'central_tendency': central_tendency,
                     '_comment': 'Peak detection parameters, including if using mean or median for PAC plot',
                     },     
                        
                'plot_settings':
                    
                    {'plot_eeg_trace': plot_eeg_trace,
                     'freq_ranges_to_check_power': freq_ranges_to_check_power,
                     '_comment': 'EEG trace plot will pop and and will not continue without intervening, so adding option to turn it off for multiple data run',
                     },    
                    
                    }
                ]   
        
    with open(save_path.joinpath('config_pac.yaml'), 'w') as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)



def create_move_copy_file_config(save_path: Path, pac_fixed_path: Path) -> yaml:
    
    date_time = datetime.now().strftime(format="%Y%m%d_%H%M%S")
    starting_folder = STARTING_FOLDER
    
    # this is to move files after the whole process is complete
    move_from_folder_user_defined = f'/{starting_folder}/processed_EEG/RELAXProcessed'
    # move_from_folder_user_defined = f'/{starting_folder}/processed_EEG/025Hz_HighPass_MWF_01lostep_2bw/RELAXProcessed'
    move_to_folder_user_defined = f'/{starting_folder}/processed_EEG/RELAXProcessed'

    move_from_folder_default = f'/{starting_folder}'
    move_to_folder_default = f'/pac_output_{date_time}'
    
    # this is for copying fixed pac files to the source code package
    home_dir = Path.home()
    tensorpac_dir = r'Anaconda3/envs/tram_protocol_eeg/Lib/site-packages/tensorpac'
    tensorpac_fulldir = fr'{home_dir.joinpath(tensorpac_dir)}'
    
    
    # Create a dictionary to store the configuration
    # (DO NOT CHANGE THIS)
    config = [{
            
            'moving_data': {
                'which_drive_from': WHICH_DRIVE_FROM,
                'which_drive_to': WHICH_DRIVE_TO,
                'move_from_folder_user_defined': move_from_folder_user_defined,
                'move_to_folder_user_defined': move_to_folder_user_defined,
                'move_from_folder_default': move_from_folder_default,
                'move_to_folder_default': move_to_folder_default,
                '_comment': 'Data moving from and to',
                },
            
            'replace_pac_src': {
                'copy_from_folder': pac_fixed_path,
                'copy_to_folder': tensorpac_fulldir,
                'files_to_copy': ['pac.py', 'spectral.py', 'utils.py'],
                '_comment': 'This is to replace pac source code file that throw erros.'
                
                }
            }]
            
    with open(save_path.joinpath('config_move_copy_files.yaml'), 'w') as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)



def create_pipeline_step_config(save_path: Path) -> yaml:
    
    # overwrite all config files
    overwrite_config = OVERWRITE_CONFIG
    
    # for each step to run or not run
    run_copy_fixed_pac_file = True
    run_set_conversion = True
    run_relax_preprocessing = True
    run_relax_epoching = True
    run_file_splitting = False ## best to turn it off
    run_pac_analysis = True
    run_moving_data = False  
   
    # keep windows open until closing for MATLAB subprocesses
    keep_window_open = KEEP_WINDOW_OPEN
    
    # use default (timestamp) when moving files
    use_default_output_folder = True
    
    # Create a dictionary to store the configuration
    # (DO NOT CHANGE THIS)
    config = [{
            'running_steps': {
                'overwrite_config': overwrite_config,
                'run_copy_fixed_pac_file': run_copy_fixed_pac_file,
                'run_set_conversion': run_set_conversion,
                'run_relax_preprocessing': run_relax_preprocessing,
                'run_relax_epoching': run_relax_epoching,
                'run_file_splitting': run_file_splitting,
                'run_pac_analysis': run_pac_analysis,
                'run_moving_data': run_moving_data,
                'keep_window_open': keep_window_open,
                'use_default_output_folder': use_default_output_folder,
                '_comment': 'Running step sequence',
                },

            }]
            
    with open(save_path.joinpath('config_pipeline_step.yaml'), 'w') as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)


    

