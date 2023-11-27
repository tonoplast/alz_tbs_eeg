# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 20:51:19 2023

@author: sungw
"""

from dataclasses import dataclass
from typing import Union
import numpy as np
import mne

@dataclass
class EpochsRestingEeg:
    mne_raw_eeg: object
    tmin: int
    tmax: int
    overlap: int
    
    '''
    Parameters
    ----------
    mne_raw_eeg : object
        cleaned and continuous eeg data (mne) to be epoched.
    tmin : int
        starting time window in seconds.
    tmax : int
        ending time window in seconds.
    overlap : int
        overlapping amount.

    Returns
    -------
    object
        epoched eeg data (mne).

    
    This is mostly used for resting EEG data, epoching with overlaps.
    It first creates events and epochs around these events.
    Baseline correction hard-coded (across all data, not pre-stimulus)
    
    e.g.
    mne_raw_eeg = raw
    tmin = -3
    tmax = 9
    overlap = 9
    
    epochs = EpochsRestingEeg(mne_raw_eeg, tmin, tmax, overlap)()

    '''

    def make_fixed_length_events(self) -> object:
        return mne.make_fixed_length_events(self.mne_raw_eeg,
                                             id=1,
                                             duration=self.tmax - self.tmin,
                                             overlap=self.overlap)

    def epoch_resting(self) -> object:
        events = self.make_fixed_length_events()
        return mne.Epochs(self.mne_raw_eeg, 
                           events, 
                           tmin=self.tmin, 
                           tmax=self.tmax, 
                           baseline=(None,None),
                           preload=True)

    def __call__(self):
        return self.epoch_resting()


@dataclass
class ChannelSelector:
    """
    Selects channels of interest from epoched EEG data.

    Attributes:
        epochs (mne.Epochs): The epoched EEG data.
    """
    epoched_eeg: object
    channels: list[Union[str, list[str]]]

    def __post_init__(self):
        """
        Convert single string input to a list.
        """
        for i, channel in enumerate(self.channels):
            if isinstance(channel, str):
                self.channels[i] = [channel]

        self.channels = [list(set(channel)) for channel in self.channels]

    def select_channels_of_interest(self) -> list[np.ndarray]:
        """
        Selects channels of interest from the epoched EEG data.

        Returns:
            A list of data arrays for the selected channels.
        """
        channel_data = [np.mean(self.epoched_eeg.copy().pick_channels(channel).get_data(), axis=1) 
                        for channel in self.channels]
        
        return channel_data
    
    def __call__(self):
        return self.select_channels_of_interest()