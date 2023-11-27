# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 13:19:35 2023

@author: sungw
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from termcolor import colored
from dataclasses import dataclass, field
from tensorpac.utils import PSD
from tensorpac.pac import _PacObj
from tensorpac import PreferredPhase
from tensorpac.methods.meth_pac import _kl_hr
import logging
from src.analysis.pac.pac_calculator import PacCalculator

logger = logging.getLogger('tensorpac')


@dataclass
class PowerSpectrumDensity:
    eeg_data: object                # input eeg data
    sampling_frequency: float       # sampling frequency of eeg data
    f_min: int                      # minimum frequency for PSD calculation
    f_max: int                      # maximum frequency for PSD calculation
    modulating_freq_range: list = field(default_factory = [4, 8])  # frequency range for modulating signal
    
    def __post_init__(self):
        # calculate PSD for eeg data
        self.psd = PSD(self.eeg_data, self.sampling_frequency)
    
    def plot_psd(self):
        plt.figure(figsize=(12, 6))
        # plot mean PSD over trials
        plt.subplot(1, 2, 1)
        ax = self.psd.plot(confidence=95, f_min=self.f_min, f_max=self.f_max, 
                          log=True, grid=True,
                          fz_labels=10, fz_title=12)
        vmax = (ax.get_ylim()[0] + ax.get_ylim()[1])/2

        # plot modulating signal frequency range as vertical lines
        plt.axvline(self.modulating_freq_range[0], lw=2, color='red')
        plt.axvline(self.modulating_freq_range[1], lw=2, color='red')

        # plot single trial PSD
        plt.subplot(1, 2, 2)
        self.psd.plot_st_psd(cmap='Greys', f_min=self.f_min, f_max=self.f_max, 
                            vmax=vmax, vmin=0, log=True, grid=True,
                            fz_labels=10, fz_title=12)
        # plot modulating signal frequency range as vertical lines
        plt.axvline(self.modulating_freq_range[0], lw=2, color='red')
        plt.axvline(self.modulating_freq_range[1], lw=2, color='red')
        plt.tight_layout()
        plt.show(block=False)


@dataclass
class PacPlots:
    pac_object: object              # an object of the pac class
    pac_calculation: np.ndarray     # numpy array containing the phase amplitude coupling values
    freqs_of_interest: list[str]    # list of frequencies of interest (e.g. ['Theta', 'Gamma'] to be included in title of the plot)
    minimum_distance: int = 3       # minimum distance between peaks for peak detection
    mcp: str = 'maxstat'            # type of multiple comparisons correction
    pvalue_threshold: float = .05   # threshold for p-values
    modulating_freq_range: list[float] = field(default_factory = [4, 8]) 
    modulated_freq_range: list[float] = field(default_factory = [30, 60])
    central_tendency: str = 'mean'  # determine central tendency as mean or median
    
    
    def __post_init__(self):
        # determine central tendency
        if self.central_tendency.strip() == 'mean':
            central_func = np.mean
        elif self.central_tendency.strip() == 'median':
            central_func = np.median
        else:
            raise ValueError("Invalue value for 'central_tendency' argument. It should be either 'mean' or 'median'.")

        # compute the mean over the last dimension of pac calculation array
        self.pac_calculation_mean_or_median = central_func(self.pac_calculation, -1)
        
        # find the coordinates of the local maxima in the mean pac calculation
        self.peak_coordinates = peak_local_max(self.pac_calculation_mean_or_median, min_distance=self.minimum_distance, threshold_abs=0)
        
        # find individualised peaks based on the computed peak coordinates
        self._individualised_peaks, self._individualised_peaks_values = self.get_peak_coordinates(self.peak_coordinates)

        # create a string representation of the recommended peaks
        self.recommended_peaks_str_hz, self.recommended_peaks_str_ms = self.get_recommended_peaks(self._individualised_peaks)
        
        # for significant coupling        
        self.pvalues = self.pac_object.infer_pvalues(p=self.pvalue_threshold, mcp=self.mcp)
        self.pac_prep_ns = self.pac_calculation_mean_or_median.copy()
        self.pac_prep_ns[self.pvalues < self.pvalue_threshold] = np.nan
        
        self.pac_prep_s = self.pac_calculation_mean_or_median.copy()
        self.pac_prep_s[self.pvalues >= self.pvalue_threshold] = np.nan
        
        self.peak_coordinates_sig = peak_local_max(self.pac_prep_s, min_distance=0, threshold_abs=0)
        self.individualised_peaks_sig, self.individualised_peaks_sig_values = self.get_peak_coordinates(self.peak_coordinates_sig) # for significant coupling
        
        # for peak in the significant result (mcp) -- has to be between freq ranges
        self._individualised_peaks_sig_subset = [(x, y) for x, y in self.individualised_peaks_sig \
                                           if self.modulating_freq_range[0] <= x <= self.modulating_freq_range[1] \
                                           and self.modulated_freq_range[0] <= y <= self.modulated_freq_range[1]]
        
        self._individualised_peaks_sig_indices = [i for i, (x, y) in enumerate(self.individualised_peaks_sig)
                                      if self.modulating_freq_range[0] <= x <= self.modulating_freq_range[1]
                                      and self.modulated_freq_range[0] <= y <= self.modulated_freq_range[1]]

        
        self._individualised_peaks_sig_values_subset = [self.individualised_peaks_sig_values[i] for i in self._individualised_peaks_sig_indices]

        # self._individualised_peaks_sig_values_subset = [(x, y) for x, y in self.individualised_peaks_sig_values \
        #                                        if self.modulating_freq_range[0] <= x <= self.modulating_freq_range[1] \
        #                                        and self.modulated_freq_range[0] <= y <= self.modulated_freq_range[1]]
        
        self.recommended_peaks_sig_str_hz, self.recommended_peaks_sig_str_ms = self.get_recommended_peaks(self._individualised_peaks_sig_subset) # for significant coupling

    
    
    def get_peak_coordinates(self, coordinates: list[list[float]]) -> list[float]:
        '''
        This is used to get peak coordinates
        pac object has x (xvec) and y (yvec) labels, and x,y coordinates are used to derive peaks
        The coupled peaks are represented in list of tuples
        '''
        x_coords = coordinates[:, 1]
        y_coords = coordinates[:, 0]
        x_freqs = np.round(self.pac_object.xvec[x_coords],1)
        y_freqs = np.round(self.pac_object.yvec[y_coords])
        peak_coordinates_list = list(zip(x_freqs, y_freqs))
        peak_coordinates_list = [(x, int(y)) for (x, y) in peak_coordinates_list]
        
        peak_values_list = list(np.round(self.pac_calculation_mean_or_median[y_coords, x_coords], decimals=5))
        return peak_coordinates_list, peak_values_list
       
    
    def get_recommended_peaks(self, peaks: list[tuple[float]]) -> None:
        '''
        Prints recommended peak - gets first in the local maximum peak
        '''
        # if no significant values
        if not peaks:
            print_hz = '******     Not found     ******'
            print_ms = '******     Not found     ******'
        else:
            hz_to_sec = lambda x: 0 if x == 0 else round((1/x * 1000) * 2) / 2
            modulating_peak = peaks[0][0]
            modulated_peak = peaks[0][1]
            
            modulating_peak_ms = '{:.1f}'.format(hz_to_sec(modulating_peak))
            modulated_peak_ms = '{:.1f}'.format(hz_to_sec(modulated_peak))
            
            print_hz = f'{self.freqs_of_interest[0]}: {modulating_peak} Hz, {self.freqs_of_interest[1]}: {int(modulated_peak)} Hz      '
            print_ms = f'{self.freqs_of_interest[0]}: {modulating_peak_ms} ms, {self.freqs_of_interest[1]}: {modulated_peak_ms} ms'

        return print_hz, print_ms
    
    
    def plot_pac(self):
        '''
        Using pac_object and pac_calculation, it will find the local max peak using peak_local_max function by skimage.feature.
        It will plot regular comodulogram, as well as stat version of comodulogram, where non-significant pixels are 'greyed' out.
        It will also print recommended individualised peaks for stimulation.
        '''
        # import pdb; pdb.set_trace()
        plt.figure(figsize=(14, 6))
        plt.subplot(121)
        
        title = f'{self.pac_object.method}\n{self.recommended_peaks_str_hz}\n({self.central_tendency})'
        n_contours = self.pac_calculation_mean_or_median.shape[0] + self.pac_calculation_mean_or_median.shape[1]

        if self.peak_coordinates.any():            
            vmax = self.pac_calculation_mean_or_median[self.peak_coordinates[0][0], self.peak_coordinates[0][1]]
        else:
            print(colored('WARNING: No peak was found, perhaps reduce the "minimum_distance" in peak detection and try again.', 'yellow'))
            vmax = self.pac_calculation_mean_or_median.max()

        self.pac_object.comodulogram(self.pac_calculation_mean_or_median, 
                       title=title, 
                       cmap='jet', # 'coolwarm', # jet
                       vmin=0,
                       vmax=vmax,
                       # over='lightgrey',
                       plotas='contour', ncontours=n_contours, #imshow contour pcolor
                       fz_labels=10, fz_title=12, fz_cblabel=10)
        
        # annotates the comodulogram with '*' for the highest PAC calculation and peak. Other peaks will be '+'
        [plt.annotate('+', xy=i, ha='center', va='center', color='black', weight='bold', size=8) if e!=0 else 
         plt.annotate('*', xy=i, ha='center', va='center', color='black', weight='bold', size=14) for e, i in enumerate(self.individualised_peaks)]
            
 
        # =============================================================================
        # # get the p-values
        # =============================================================================
        plt.subplot(122)   
        
        title = (rf"Significant {self.freqs_of_interest[0]}$\Leftrightarrow${self.freqs_of_interest[1]} coupling occurring during "
                 f"the resting phase\n(p<{self.pvalue_threshold}, {self.mcp}-corrected for multiple "
                  f"comparisons)\n{self.recommended_peaks_sig_str_hz}")
        
        # plot the non-significant pac in gray
        self.pac_object.comodulogram(self.pac_prep_ns, cmap='gray', vmin=np.nanmin(self.pac_prep_ns),
                           vmax=np.nanmax(self.pac_prep_ns), colorbar=False,
                           fz_labels=10, fz_title=12, fz_cblabel=10)
        
        # plot the significant pac in color
        self.pac_object.comodulogram(self.pac_prep_s, cmap='Spectral_r', vmin=np.nanmin(self.pac_prep_s),
                           vmax=np.nanmax(self.pac_prep_s), title=title,
                           fz_labels=10, fz_title=12, fz_cblabel=10)
        
        
        [plt.annotate('*', xy=i, ha='center', va='center', color='black', weight='bold', size=20) for e, i in enumerate(self._individualised_peaks_sig_subset) if e==0]

        plt.gca().invert_yaxis()
        
        plt.xlim(self.modulating_freq_range)
        plt.ylim(self.modulated_freq_range)
        
        plt.tight_layout()
        plt.show(block=False)
        print(
              colored(f'\n\n#######################################################################################################\
                          \n#####################################    PAC results ({self.central_tendency.center(6)})   #######################################\
                          \n######                                                                                           ######\
                          \n######     Your individualised frequencies are:         {self.recommended_peaks_str_hz}        ######\
                          \n######                                                  {self.recommended_peaks_str_ms}          ######\
                          \n######                                                                                           ######\
                          \n######     Your sig. individualised frequencies are:    {self.recommended_peaks_sig_str_hz}          ######\
                          \n######                                                  {self.recommended_peaks_sig_str_ms}          ######\
                          \n######                                                                                           ######\
                          \n#######################################################################################################\
                          \n#######################################################################################################\n',
                          'cyan')
              )
            

    @property
    def individualised_peaks(self):
        """Get the individualised_peaks value."""
        return self._individualised_peaks
    
    @property
    def individualised_peaks_sig_subset(self):
        """Get the individualised_peaks_sig_subset value."""
        return self._individualised_peaks_sig_subset

    @property
    def individualised_peaks_values(self):
        """Get the individualised_peaks_values value."""
        return self._individualised_peaks_values
    
    @property
    def individualised_peaks_sig_values_subset(self):
        """Get the individualised_peaks_sig_values_subset value."""
        return self._individualised_peaks_sig_values_subset
    

class BinAmplitude_mod(_PacObj):
    """Bin the amplitude according to the phase.
    Parameters
    ----------
    x : array_like
        Array of data of shape (n_epochs, n_times)
    
    # =============================================================================
    #     x is now x_pha and x_amp
    # =============================================================================
    
    sf : float
        The sampling frequency
    f_pha : tuple, list | [2, 4]
        List of two floats describing the frequency bounds for extracting the
        phase
    f_amp : tuple, list | [60, 80]
        List of two floats describing the frequency bounds for extracting the
        amplitude
    n_bins : int | 18
        Number of bins to use to binarize the phase and the amplitude
    dcomplex : {'wavelet', 'hilbert'}
        Method for the complex definition. Use either 'hilbert' or
        'wavelet'.
    cycle : tuple | (3, 6)
        Control the number of cycles for filtering (only if dcomplex is
        'hilbert'). Should be a tuple of integers where the first one
        refers to the number of cycles for the phase and the second for the
        amplitude :cite:`bahramisharif2013propagating`.
    width : int | 7
        Width of the Morlet's wavelet.
    edges : int | None
        Number of samples to discard to avoid edge effects due to filtering
    """

    def __init__(self, x_pha, x_amp, sf, f_pha=[4, 8], f_amp=[30, 60], n_bins=18,
                 dcomplex='hilbert', cycle=(3, 6), width=7, edges=None,
                 n_jobs=-1):
        """Init."""
        _PacObj.__init__(self, f_pha=f_pha, f_amp=f_amp, dcomplex=dcomplex,
                         cycle=cycle, width=width)
        # check
        x = np.atleast_2d(x_pha)
        assert x.ndim <= 2, ("`x` input should be an array of shape "
                             "(n_epochs, n_times)")
        assert isinstance(sf, (int, float)), ("`sf` input should be a integer "
                                              "or a float")
        assert all([isinstance(k, (int, float)) for k in f_pha]), (
            "`f_pha` input should be a list of two integers / floats")
        assert all([isinstance(k, (int, float)) for k in f_amp]), (
            "`f_amp` input should be a list of two integers / floats")
        assert isinstance(n_bins, int), "`n_bins` should be an integer"
        logger.info(f"Binning {f_amp}Hz amplitude according to {f_pha}Hz "
                    "phase")
        # extract phase and amplitude
        kw = dict(keepfilt=False, edges=edges, n_jobs=n_jobs)
        pha = self.filter(sf, x_pha, 'phase', **kw)
        amp = self.filter(sf, x_amp, 'amplitude', **kw)
        # binarize amplitude according to phase
        self._amplitude = _kl_hr(pha, amp, n_bins, mean_bins=False).squeeze()
        self.n_bins = n_bins

    def plot(self, unit='rad', normalize=False, **kw):
        """Plot the amplitude.
        Parameters
        ----------
        unit : {'rad', 'deg'}
            The unit to use for the phase. Use either 'deg' for degree or 'rad'
            for radians
        normalize : bool | None
            Normalize the histogram by the maximum
        kw : dict | {}
            Additional inputs are passed to the matplotlib.pyplot.bar function
        Returns
        -------
        ax : Matplotlib axis
            The matplotlib axis that contains the figure
        """
        import matplotlib.pyplot as plt
        assert unit in ['rad', 'deg']
        if unit == 'rad':
            self._phase = np.linspace(-np.pi, np.pi, self.n_bins)
            width = 2 * np.pi / self.n_bins
        elif unit == 'deg':
            self._phase = np.linspace(-180, 180, self.n_bins)
            width = 360 / self.n_bins
        amp_mean = self._amplitude.mean(1)
        if normalize:
            amp_mean /= amp_mean.max()
        plt.bar(self._phase, amp_mean, width=width, **kw)
        plt.xlabel(f"Frequency phase ({self.n_bins} bins)", fontsize=10)
        plt.ylabel("Amplitude", fontsize=10)
        plt.title("Binned amplitude", fontsize=12)
        plt.autoscale(enable=True, axis='x', tight=True)

    def show(self):
        """Show the figure."""
        import matplotlib.pyplot as plt
        plt.show()

    @property
    def amplitude(self):
        """Get the amplitude value."""
        return self._amplitude

    @property
    def phase(self):
        """Get the phase value."""
        return self._phase
    
    

@dataclass
class PhasePlots:
    individualised_amplitude_peak_freq: tuple[float]
    modulating_data: np.ndarray
    modulated_data: np.ndarray
    modulating_freq_range: list[float]
    modulated_freq_range: list[float]
    modulating_freq_bandwidth: float
    modulated_freq_bandwidth: float
    modulating_freq_step: float
    modulated_freq_step: float
    n_bins: int = 18
    sampling_frequency: float = 256
    dcomplex: str = 'hilbert'
    central_tendency: str = 'mean'  # determine central tendency as mean or median


    def __post_init__(self):
        '''
        This is used for binned amplitude plot
        We are using the individualised peak frequencies to bin specifically
        '''
        self.f_pha = self.get_individualised_freq_range(pha_or_amp='phase')
        self.f_amp = self.get_individualised_freq_range(pha_or_amp='amplitude')
        # determine central tendency
        if self.central_tendency == 'mean':
            self.central_func = np.mean
        elif self.central_tendency == 'median':
           self.central_func = np.median
        else:
            raise ValueError("Invalue value for 'central_tendency' argument. It should be either 'mean' or 'median'.")
        
        
    def get_individualised_freq_range(self, pha_or_amp: str = 'phase') -> list[float]:
                    
        '''
        This is used for bin amplitude plot.
        We are only interested in the binned amplitude based on the highest coupling
        so we input the peak of individualised freq obtained and +/- freq steps
        '''
        
        assert pha_or_amp in ['phase', 'amplitude'], "pha_or_amp: should be either 'phase' or 'amplitude'."
        freq_step = self.modulating_freq_step if pha_or_amp == 'phase' else self.modulated_freq_step
        pha_amp = self.individualised_amplitude_peak_freq[0] if pha_or_amp == 'phase' else self.individualised_amplitude_peak_freq[1]
        
        # giving range of frequency steps so the mid freq is the individualised freq
        f_pha_amp = [pha_amp - freq_step, pha_amp + freq_step]    

        if pha_or_amp == 'phase':
            f_pha_amp = [round(i, 1) for i in f_pha_amp]
        else:
            f_pha_amp = [int(i) for i in f_pha_amp]
            
        return f_pha_amp
        
    
    def get_preferred_phase(self) -> list[tuple[float]]:
        '''
        Extract phases, amplitudes and compute the preferred phase
        '''
        # self.f_pha_range = PacCalculator.get_freq_range_for_pac(self, freq_range=self.modulating_freq_range,
        #                                           freq_bandwidth=self.modulating_freq_bandwidth,
        #                                           freq_step=self.modulating_freq_step)
        
        self.f_amp_range = PacCalculator.get_freq_range_for_pac(self, freq_range=self.modulated_freq_range,
                                                  freq_bandwidth=self.modulated_freq_bandwidth,
                                                  freq_step=self.modulated_freq_step)     

        # f_amp_range = (self.modulated_freq_range[0], self.modulated_freq_range[1], self.modulated_freq_bandwidth, self.modulated_freq_step)
        preferred_phase = PreferredPhase(f_pha=self.f_pha, ###### f_pha_range
                                         f_amp=self.f_amp_range, 
                                         dcomplex=self.dcomplex)
        
        # Extract the phase and the amplitude
        pha = preferred_phase.filter(self.sampling_frequency, self.modulating_data, ftype='phase', n_jobs=1)
        amp = preferred_phase.filter(self.sampling_frequency, self.modulated_data, ftype='amplitude', n_jobs=1)

        # Compute the preferred phase and reshape for plotting purposes
        ampbin, pp, vecbin = preferred_phase.fit(pha, amp, n_bins=72)
        pp_tpose = np.squeeze(pp).T
        ampbin_mean_or_median_tpose = self.central_func(np.squeeze(ampbin), -1).T
        
        return ampbin_mean_or_median_tpose, pp_tpose, vecbin, preferred_phase



    def get_bin_amplitude(self) -> object:
        b_obj = BinAmplitude_mod(x_pha=self.modulating_data,
                             x_amp=self.modulated_data,
                             sf=self.sampling_frequency, 
                             f_pha=self.f_pha, ## we used individualised peak here
                             f_amp=self.f_amp, ## we used individualised peak here
                             n_bins=self.n_bins,
                             dcomplex=self.dcomplex)
        return b_obj


    def plot_pp_bin(self) -> None:
        '''
        plotting preferred phase
        '''
        # get preferred phase
        ampbin_mean_or_median_tpose, pp_tpose, vecbin, preferred_phase = self.get_preferred_phase()
        
        # =============================================================================
        #         # plotting
        # =============================================================================
        plt.figure(figsize=(10, 8))
        # Plot the prefered phase (single trial)
        plt.subplot(221)
        plt.pcolormesh(preferred_phase.yvec, np.arange(pp_tpose.shape[0]), np.rad2deg(pp_tpose), cmap='RdBu_r')

        cb = plt.colorbar()
        plt.clim(vmin=-180., vmax=180.)
        plt.axis('tight')
        plt.xlabel('Amplitude frequencies (Hz)')
        plt.ylabel('Epochs')
        plt.title(f"Single trial Preferred Phase (PP at {self.individualised_amplitude_peak_freq[0]}Hz)\n according to amplitudes")
        cb.set_label('PP (in degrees)')
        
        # show the histogram corresponding to the individualised frequency amplitude (e.g. gamma)
        idx_ind = np.abs(preferred_phase.yvec - self.individualised_amplitude_peak_freq[1]).argmin()
        plt.subplot(222)
        h = plt.hist(pp_tpose[:, idx_ind], color='#ab4642')
        plt.xlim((-np.pi, np.pi))
        plt.xlabel('PP')
        plt.title(f'PP across trials for the {self.individualised_amplitude_peak_freq[1]}Hz amplitude')
        plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        plt.gca().set_xticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", "$0$",
                                  r"$\frac{\pi}{2}$", r"$\pi$"])
        
        # plot preferred phase (polar plot)
        ax = preferred_phase.polar(ampbin_mean_or_median_tpose, vecbin, preferred_phase.yvec, 
                              cmap='RdBu_r', interp=.1, subplot=223, 
                              cblabel='Amplitude bins')
        ax.set_rlabel_position(-45)
        
        # plot binned amplitude
        plt.subplot(224)
        b_obj = self.get_bin_amplitude()
        ax = b_obj.plot(color='blue', alpha=.5, unit='deg')
        plt.title(f"Binned amplitude (phase={self.f_pha},\n amplitude={self.f_amp})", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        ax = plt.gca()

        plt.show(block=False)
        plt.tight_layout()
        
