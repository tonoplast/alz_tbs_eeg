close all; clear;

% Path (repo, cap location, RELAX config file)
userprofilePath = getenv('USERPROFILE');
repoPath = [userprofilePath, filesep 'GitHub' filesep 'alz_tbs_eeg'];
configPath = [repoPath, filesep 'src' filesep 'config'];
cfg = jsondecode(fileread([configPath filesep 'config_matlab.json'])).relax_processing;
eeglab_path = [repoPath, filesep 'src' filesep 'toolbox' filesep cfg.eeglab_version];
RELAX_epoching_cfg = [configPath filesep 'config_RELAX_epoching_default.mat'];

% Path (loading / saving)
inPath = [cfg.data_drive, filesep cfg.starting_folder filesep 'processed_EEG' filesep 'RELAXProcessed' filesep 'Cleaned_Data'];
outPath = [inPath, filesep 'Epoched'];

if not(isfolder(outPath))
    mkdir(outPath)
end

% RELAX config
load(RELAX_epoching_cfg);
in_files = dir(fullfile(inPath, '*.set'));
RELAX_epoching_cfg.CleanedPath = inPath;
RELAX_epoching_cfg.FilesToProcess = 1:length(in_files);
RELAX_epoching_cfg.dirList = in_files;
RELAX_epoching_cfg.files = {in_files.name};
RELAX_epoching_cfg.OutputPath = outPath;

% resting EEG specific
RELAX_epoching_cfg.DataType = cfg.data_type;
RELAX_epoching_cfg.BLperiod = cfg.baseline_preiod';
RELAX_epoching_cfg.PeriodToEpoch = cfg.period_to_epoch';
RELAX_epoching_cfg.restingdatatriggerinterval = cfg.resting_data_trigger_interval; % marking, so 9 sec overlap (75 %)
RELAX_epoching_cfg.BL_correction_method = cfg.baseline_correction_method;

% epoch rejection threshold
RELAX_epoching_cfg.SingleChannelImprobableDataThreshold = cfg.single_channel_improbable_data_threshold;
RELAX_epoching_cfg.AllChannelImprobableDataThreshold = cfg.all_channel_improbable_data_threshold';
RELAX_epoching_cfg.SingleChannelKurtosisThreshold = cfg.single_channel_kurtosis_threshold';
RELAX_epoching_cfg.AllChannelKurtosisThreshold = cfg.all_channel_kurtosis_threshold; % marking, so 9 sec overlap (75 %)
RELAX_epoching_cfg.reject_amp = cfg.reject_amp;
RELAX_epoching_cfg.MuscleSlopeThreshold = cfg.muscle_slope_threshold;
RELAX_epoching_cfg.MaxProportionOfMuscleEpochsToClean = cfg.max_proportion_of_muscle_epochs_to_clean;

% initialise eeglab
cd(eeglab_path);
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

cd(inPath)
% run RELAX epoching
RELAX_epoch_the_clean_data_Wrapper(RELAX_epoching_cfg);