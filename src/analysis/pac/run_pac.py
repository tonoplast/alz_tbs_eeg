# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 16:03:56 2023

@author: sungw
"""

from pathlib import Path
import matplotlib.pyplot as plt
import mne
import json
from tqdm import tqdm
import pandas as pd
from src.analysis.pac.resting_eeg_preparator import EpochsRestingEeg, ChannelSelector
from src.analysis.pac.pac_calculator import PacCalculator
from src.visualization.pac_plotter import PowerSpectrumDensity, PacPlots, PhasePlots
from src.visualization.eeg_plotter import plot_epochs
from src.utils import config_reader
from src.analysis.pac.verification.get_epoch_numbers_and_freq_power import get_epoch_no_and_power
import math

def get_config_and_files_to_process():
    '''
    loading config files and get the list of data to be processed
    '''
    # =============================================================================
    # Create configuration 
    # =============================================================================
    p = Path('.').resolve()
    config_folder = p.joinpath('src', 'config')
    pac_cfg = config_reader.read_yaml_file(config_folder.joinpath('config_pac.yaml'))
    
    
    # =============================================================================
    # configuration for results output  
    # =============================================================================
    save_output = pac_cfg.get('save_output')
    output_folder = Path(f"{pac_cfg.get('which_drive')}/{pac_cfg.get('output_folder')}/")
    output_folder.mkdir(parents=True, exist_ok=True)
    
    
    # =============================================================================
    # configuration for data input
    # =============================================================================
    # get data in the folder and sort by the latest date. Only looks at .set files
    data_folder = Path(f"{pac_cfg.get('which_drive')}/{pac_cfg.get('input_folder')}/")
    input_files = [f for f in data_folder.glob('*.set') if f.is_file()]
    sorted_files = sorted(input_files, key=lambda x: x.stat().st_mtime, reverse=pac_cfg.get('reverse_order'))
    
    # if process evertyhing, then we process all, if not, only take the lastest file
    data_files = sorted_files if pac_cfg.get('number_of_files_to_process') == '' \
        else sorted_files[0:int(pac_cfg.get('number_of_files_to_process'))]
    
    if not data_files:
        raise TypeError ('There is no data in the designated folder!')
    
    # =============================================================================
    # Loading data
    # =============================================================================
    data_type = pac_cfg.get('data_type')
    assert data_type in ['raw','epoched'], "'data_type' should be either 'raw' or 'epoched'"
    
    return pac_cfg, save_output, output_folder, data_folder, data_files, data_type


def process_epoched_data(data_file, data_folder, data_type, pac_cfg):
    '''
    processing data - baseline correction / re-referencing / surface laplacian transform
    '''
    # get the file name (stem without extension)
    filename = data_file.relative_to(data_folder).stem

    if data_type == 'raw':
        ## if raw, epoch continous data
        raw = mne.io.read_raw_eeglab(data_file, preload=True)
        # raw_downsampled = raw.resample(sfreq=pac_cfg.get('set_srate'))
        resting_epochs = EpochsRestingEeg(mne_raw_eeg=raw.copy(), 
                                          tmin = pac_cfg.get('tmin'), 
                                          tmax = pac_cfg.get('tmax'), 
                                          overlap = pac_cfg.get('overlap'))()
                                          
    elif data_type == 'epoched':
        resting_epochs = mne.io.read_epochs_eeglab(data_file)
        if (pac_cfg.get('baseline_correction')):
            # baseline correctiong entire epoch
            resting_epochs = resting_epochs.apply_baseline(baseline = (None,None))
    
    # average re-referencing
    if (pac_cfg.get('average_reref')) & (not pac_cfg.get('laplacian_transform')):    
        resting_epochs_reref = resting_epochs.copy().set_eeg_reference(ref_channels='average', projection=True).apply_proj()
    
    # laplacian transform
    elif (not pac_cfg.get('laplacian_transform')) & (pac_cfg.get('average_reref')):
        resting_epochs_reref = mne.preprocessing.compute_current_source_density(resting_epochs)
        
    # average then laplacian transform
    elif (pac_cfg.get('average_reref')) & (pac_cfg.get('laplacian_transform')):
        resting_epochs_reref = resting_epochs.copy().set_eeg_reference(ref_channels='average', projection=True).apply_proj()
        resting_epochs_reref = mne.preprocessing.compute_current_source_density(resting_epochs_reref)
    else:
        resting_epochs_reref = resting_epochs.copy()

    ## down-sampling removed because it is not good to do it on epoched data
    # # down-sampling
    # resting_epochs_downsampled = resting_epochs_avg_reref.resample(sfreq=pac_cfg.get('set_srate'))
    
    # plot eeg trace, epoch psd and topoplot (this option is for when running multiple and just want to continue)
    if pac_cfg.get('plot_eeg_trace'):
        plot_epochs(resting_epochs_reref)
        
    return filename, resting_epochs_reref
    

def run_pac(output_filename_tag: str = '_hilbert'):
    
    '''
    This is a 'running' function for pac. We load up the config file so that pac will run based on the configs   
    
    'output_filename_tag' is for any additional string that can be attached to the name of the output.
    It would be good to start with underscore, 
    e.g) '_hilbert' so that the output might look like the below.
    e.g) if filename is 'H100_resting_EC'
    output name -> 'H100_resting_EC_hilbert_xxx.png'
    '''
    
    def my_round(i):
      f = math.floor(i)
      return f if i - f < 0.5 else f+1
        
    pac_cfg, save_output, output_folder, data_folder, data_files, data_type = get_config_and_files_to_process()
    
    epoch_no_and_freq_power_list = []
    for number_of_files_ran, data_file in enumerate(tqdm(data_files, desc="Running PAC", colour='green'), 1):
                
        filename, resting_epochs_reref = process_epoched_data(data_file, data_folder, data_type, pac_cfg)
        
        # Sampling frequency
        sampling_frequency = resting_epochs_reref.info.get('sfreq')
        
        # =============================================================================
        # Select channels to couple
        # =============================================================================    
        modulating_data, modulated_data = ChannelSelector(
            resting_epochs_reref.copy(), 
            [pac_cfg.get('modulating_channels'), pac_cfg.get('modulated_channels')])()
    
        
        # =============================================================================
        # Calculate PAC
        # =============================================================================
        pac_object, \
        pac_calculation, \
        modulating_phases, \
        modulated_amplitudes = PacCalculator(
                                    modulating_data = modulating_data,
                                    modulated_data = modulated_data,
                                    sampling_frequency = sampling_frequency,
                                    edge_length = pac_cfg.get('edge_length'),
                                    modulating_freq_range = pac_cfg.get('modulating_freq_range'),
                                    modulated_freq_range = pac_cfg.get('modulated_freq_range'),
                                    lo_bandwidth = pac_cfg.get('lo_bandwidth'),
                                    lo_step = pac_cfg.get('lo_step'),
                                    hi_bandwidth = pac_cfg.get('hi_bandwidth'),
                                    hi_step = pac_cfg.get('hi_step'),
                                    minimum_distance = pac_cfg.get('minimum_distance'),
                                    idpac = pac_cfg.get('idpac'),
                                    no_of_permutations = pac_cfg.get('no_of_permutations'),
                                    pval = pac_cfg.get('pvalue_threshold'),
                                    mcp = pac_cfg.get('mcp'),
                                    random_state = pac_cfg.get('random_state'),
                                    dcomplex = pac_cfg.get('dcomplex'),
                                    cycle = pac_cfg.get('hilbert_cycle'),
                                    width = pac_cfg.get('wavelet_width')
                                    )()    
        
        # =============================================================================
        # Plotting
        # =============================================================================
        ## PSD
        plotPSD = PowerSpectrumDensity(
                                        eeg_data = modulating_data,
                                        sampling_frequency = sampling_frequency,
                                        f_min = pac_cfg.get('f_min_max')[0],
                                        f_max = pac_cfg.get('f_min_max')[1],
                                        modulating_freq_range = pac_cfg.get('modulating_freq_range')
                                        )
        plotPSD.plot_psd()
        
        if save_output:
            plt.savefig(output_folder.joinpath(f'{filename}{output_filename_tag}_psd.png'))
        
            
        if isinstance(pac_cfg.get('central_tendency'), str):
            central_tendency = [pac_cfg.get('central_tendency')]
        else:
            central_tendency = pac_cfg.get('central_tendency')

        # We'll look at both median and mean, but perhaps select median as priority
        for ct in central_tendency:
            ## PAC
            plotPAC = PacPlots(
                               pac_object = pac_object,
                               pac_calculation = pac_calculation,
                               freqs_of_interest = pac_cfg.get('freqs_of_interest'),
                               minimum_distance = pac_cfg.get('minimum_distance'),
                               mcp = pac_cfg.get('mcp'),
                               pvalue_threshold = pac_cfg.get('pvalue_threshold'),
                               modulating_freq_range = pac_cfg.get('modulating_freq_range'),
                               modulated_freq_range = pac_cfg.get('modulated_freq_range'),
                               central_tendency = ct,
                               )
               
            plotPAC.plot_pac()
            
            ## Individualised peak & value
            individualised_peaks = plotPAC.individualised_peaks
            individualised_peaks_sig = plotPAC.individualised_peaks_sig_subset
            
            individualised_peaks_values = plotPAC.individualised_peaks_values
            individualised_peaks_sig_values = plotPAC.individualised_peaks_sig_values_subset
            
            
            if save_output:
                output_folder_ct = output_folder.joinpath(f'{ct}')
                output_folder_ct.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_folder_ct.joinpath(f'{filename}{output_filename_tag}_{ct}_pac.png'))
            
            ## Preferred Phase
            if individualised_peaks:
                plotPP = PhasePlots(
                                  individualised_amplitude_peak_freq=individualised_peaks[0],
                                  modulating_data = modulating_data,
                                  modulated_data = modulated_data,
                                  modulating_freq_range = pac_cfg.get('modulating_freq_range'), 
                                  modulated_freq_range = pac_cfg.get('modulated_freq_range'),
                                  modulating_freq_bandwidth = pac_cfg.get('lo_bandwidth'),
                                  modulated_freq_bandwidth = pac_cfg.get('hi_bandwidth'),
                                  modulating_freq_step = pac_cfg.get('lo_step'),
                                  modulated_freq_step = pac_cfg.get('hi_step'),
                                  sampling_frequency = sampling_frequency, 
                                  dcomplex = pac_cfg.get('dcomplex'),
                                  )
                
                plotPP.plot_pp_bin()
                
                if save_output:
                    plt.savefig(output_folder_ct.joinpath(f'{filename}{output_filename_tag}_{ct}_pp.png'))
            else:
                print('No peak to check for preferred phase.')
                
            ## Save individualised freq in json/csv format
            if save_output:
                # save into json format
                save_dict = {'peak': individualised_peaks, 'peak_value': individualised_peaks_values, 'sig_pixel': individualised_peaks_sig, 'sig_pixel_value': individualised_peaks_sig_values}
                with open(output_folder_ct.joinpath(f'{filename}{output_filename_tag}_{ct}_ind_peaks.json'), "w") as f:
                    json.dump(save_dict, f)     
                
                # save into csv
                # import pdb; pdb.set_trace()
                hz_to_sec = lambda x: 0 if x == 0 else round((1/x * 1000) * 2) / 2
                save_csv = pd.DataFrame({key: pd.Series(value) for key, value in save_dict.items()})
                
                # Check if DataFrame is empty
                if not save_csv.empty:
                    fill_cols = ['peak', 'sig_pixel']
                    save_csv[fill_cols] = save_csv[fill_cols].fillna('(0, 0)')
                    
                    save_csv['peak'] = save_csv['peak'].apply(lambda x: (0, 0) if not isinstance(x, tuple) else x)
                    save_csv['sig_pixel'] = save_csv['sig_pixel'].apply(lambda x: (0, 0) if not isinstance(x, tuple) else x)
                    
                    # Splitting tuple into separate columns
                    save_csv[['peak_theta', 'peak_gamma']] = save_csv['peak'].apply(pd.Series)
                    save_csv[['sig_pixel_theta', 'sig_pixel_gamma']] = save_csv['sig_pixel'].apply(pd.Series)
                    
                    save_csv['peak_theta_ms'] = save_csv['peak_theta'].apply(hz_to_sec)
                    save_csv['peak_gamma_ms'] = save_csv['peak_gamma'].apply(hz_to_sec)
                    save_csv['sig_pixel_theta_ms'] = save_csv['sig_pixel_theta'].apply(hz_to_sec)
                    save_csv['sig_pixel_gamma_ms'] = save_csv['sig_pixel_gamma'].apply(hz_to_sec)
                
                save_csv.to_csv(output_folder_ct.joinpath(f'{filename}{output_filename_tag}_{ct}_ind_peaks.csv'), index=False)

                
                # simple save
                save_csv_simple = save_csv.drop(columns=['peak','peak_value']+[i for i in save_csv.columns if 'sig' in i])
                save_csv_simple.columns = [i.split('peak_')[1] for i in save_csv_simple.columns]
                try:
                    save_csv_simple = save_csv_simple.iloc[0]
                    
                    ### This is added because the machine can't do .5 above 20 ms.. How strange (and rip-off!!).
                    save_csv_simple['gamma_ms'] = my_round(save_csv_simple['gamma_ms']) if (save_csv_simple['gamma_ms'] > 20) else save_csv_simple['gamma_ms']
                except:
                    save_csv_simple = pd.DataFrame()
                    save_csv_simple.loc[0, 0] = 'No data available'       
                    print('No data available!')
                
                # import pdb; pdb.set_trace()
                id_prefix = f"{filename.split('_')[0]}_{filename.split('_')[1]}"
                eo_ec = filename.split('_')[-4]
                save_csv_simple.to_csv(output_folder.joinpath(f'{id_prefix}_{eo_ec}_top_peaks_{ct}.csv'), header=False)
                                  
            # printing all peaks just in case
            print(f'\nAll peaks found for {ct}: {individualised_peaks}')
            print(f'\nAll significant pixels found for {ct}: {individualised_peaks_sig}')
             
        
        # close plots if more than 2 files processed
        if number_of_files_ran > 2:
            plt.close('all')
        
        # epoch number and freq power for record
        epoch_no_and_freq_power = get_epoch_no_and_power(filename, 
                                                         resting_epochs_reref, 
                                                         freq_ranges = pac_cfg.get('freq_ranges_to_check_power'))
        
        # collecting all the epcoh number and freq power
        epoch_no_and_freq_power_list.append(epoch_no_and_freq_power)
    
    with open(data_folder.joinpath("epoch_number_and_freq_power.json"), "w") as f:
        json.dump(epoch_no_and_freq_power_list, f)
        
    # open folder if settings say so
    if pac_cfg.get('open_output_folder'):
        import subprocess
        subprocess.Popen(f'explorer "{output_folder}"')    