# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 08:33:27 2023

@author: sungw
"""
from tensorpac import Pac
import numpy as np
import mne
from dataclasses import dataclass, field


@dataclass
class PacCalculator:
    modulating_data: np.ndarray                     # Modulating data array
    modulated_data: np.ndarray                      # Modulated data array
    sampling_frequency: float = 256                 # Sampling frequency of the data
    edge_length: float = .5                         # Length of the edge to be removed in seconds
    modulating_freq_range: list[float] = field(default_factory = [4, 8])     # Frequency range for modulating signal
    modulated_freq_range: list[float] = field(default_factory = [30, 60])    # Frequency range for modulated signal
    lo_bandwidth: float = 2                         # Bandwidth for low-frequency signal (modulating signal)
    lo_step: float = .1                             # Step size for low-frequency signal (modulating signal)
    hi_bandwidth: float = 1                         # Bandwidth for high-frequency signal (modulated signal)
    hi_step: float = 1                              # Step size for high-frequency signal (modulated signal)
    minimum_distance: int = 3                       # Minimum distance between frequency steps for peak detection
    idpac: tuple[int, int, int] = (6, 1, 4)         # Tuple for Pac calculation (see below)
    no_of_permutations: int = 200                   # Number of permutations for significance estimation
    pval: float = .05                               # pvalue threshold
    mcp: str = 'maxstat'                            # multiple comparison procedures (maxstat, fdr, bonferroni)
    random_state: int = 0                           # random state for reproducibility
    dcomplex: str = 'wavelet'                       # method for complex definition
    cycle: tuple[int] = (3, 6)                      # cycle for hilbert
    width: int = 7                                  # width of the wavelet
    '''
    # =============================================================================
    #     ## PAC info ##
    # =============================================================================
        
    # p.idpac = (pac_method, surrogate, normalisation)
    
    # # pac_method
    # 1: Mean Vector Length (MVL)
    # 2: Modulation Index (MI)
    # 3: Heights ratio (HR)
    # 4: Normalized Direct Pac (ndPac)
    # 5: Phase-Locking Value (PLV)
    # 6: Gaussian Coupula PAC (gcPac)

    # # surrogates
    # 0: No surrogates
    # 1: Swap phase / amplitude across trials (Tort et al., 2010)
    # 2: Swap amplitude time blocks (Bahramisharif et al., 2013)
    # 3: Time lag (Canolty et al., 2006)

    # # normalisation
    # 0: No normalisation
    # 1: Substract mean of surrogates
    # 2: Divide by mean of surrogates
    # 3: Subtract then divide by the mean of surrogates
    # 4: Subtract the mean and divide by the deviation of surrogates (z-score)
    '''
    
    def get_freq_range_for_pac(self, 
                               freq_range: list[float], 
                               freq_bandwidth: float, 
                               freq_step: float, 
                               minimum_distance: int = 3) -> tuple[float]:
        """
        Returns frequency range for PAC calculation
        Note that 'minimum_distance*freq_step and/or freq_step' are subtracted/added to fit into the comodulogram
        """
        f_pha_amp = (freq_range[0] - (freq_bandwidth/2) - minimum_distance*freq_step,
                    freq_range[1] + (freq_bandwidth/2) + minimum_distance*freq_step + freq_step,
                    freq_bandwidth,
                    freq_step)
        return f_pha_amp
    
    
    def get_phase_or_amplitude(self, 
                               pac_object: object, 
                               phase_or_amplitude: str) -> tuple[np.ndarray]:
        phase_or_amplitude = pac_object.filter(self.sampling_frequency, 
                                               self.modulating_data, 
                                               ftype=phase_or_amplitude, 
                                               n_jobs=1, 
                                               edges=self.edge_length*self.sampling_frequency)
        
        return phase_or_amplitude
    
    
    def calculate_pac(self) -> tuple[object, np.ndarray]:
        """Calculates PAC and returns PAC object and calculation"""
        f_pha = self.get_freq_range_for_pac(freq_range=self.modulating_freq_range, 
                                            freq_bandwidth=self.lo_bandwidth, 
                                            freq_step=self.lo_step, 
                                            minimum_distance=self.minimum_distance)
        
        f_amp = self.get_freq_range_for_pac(freq_range=self.modulated_freq_range, 
                                            freq_bandwidth=self.hi_bandwidth, 
                                            freq_step=self.hi_step, 
                                            minimum_distance=self.minimum_distance)
        
        # Initializing the PAC object with frequency ranges and parameters
        pac_object = Pac(f_pha=f_pha, f_amp=f_amp, dcomplex=self.dcomplex, width=self.width, cycle=self.cycle)
        pac_object.idpac = self.idpac
       
        # Filtering the modulating and modulated signals
        modulating_phases = self.get_phase_or_amplitude(pac_object = pac_object,
                                                        phase_or_amplitude = 'phase')
        
        modulated_amplitudes = self.get_phase_or_amplitude(pac_object = pac_object,
                                                        phase_or_amplitude = 'amplitude')
       
        ## TODO: implement edge effect if within data (e.g. event related) here
        ## make a function perhaps for masking (need to use self.sampling_frequency)

        # Calculating PAC
        pac_calculation = pac_object.fit(pha=modulating_phases, 
                                         amp=modulated_amplitudes, 
                                         n_perm=self.no_of_permutations, 
                                         random_state=self.random_state,
                                         p=self.pval,
                                         mcp=self.mcp)

        # Returning PAC object and calculation
        return pac_object, pac_calculation, modulating_phases, modulated_amplitudes


    def __call__(self):
        return self.calculate_pac()