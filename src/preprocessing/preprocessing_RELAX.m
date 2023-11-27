close all; clear;

% Path (repo, cap location, RELAX config file)
userprofilePath = getenv('USERPROFILE');
repoPath = [userprofilePath, filesep 'GitHub' filesep 'alz_tbs_eeg'];
configPath = [repoPath, filesep 'src' filesep 'config'];
cfg = jsondecode(fileread([configPath filesep 'config_matlab.json'])).relax_preprocessing;
eeglab_path = [repoPath, filesep 'src' filesep 'toolbox' filesep cfg.eeglab_version];
RELAX_cfgPath = [configPath filesep 'config_RELAX_default.mat'];

% IMPORTANT TO ADD mfw subfolders to path
addpath(genpath([eeglab_path, filesep 'plugins' filesep 'mwf-artifact-removal-master']));

% Path (loading / saving)
inPath = [cfg.data_drive, filesep cfg.starting_folder filesep 'processed_EEG'];
outPath = [inPath, filesep 'RELAXProcessed'];

if not(isfolder(outPath))
    mkdir(outPath)
end

% RELAX config
load(RELAX_cfgPath);
in_files = dir(fullfile(inPath, '*.set'));
RELAX_cfg.caploc = [configPath filesep cfg.caploc];
RELAX_cfg.myPath = inPath;
RELAX_cfg.FilesToProcess = 1:length(in_files);
RELAX_cfg.dirList = in_files;
RELAX_cfg.files = {in_files.name};
RELAX_cfg.OutputPath = outPath;
RELAX_cfg.HighPassFilter = cfg.HighPassFilter;
RELAX_cfg.LowPassFilter = cfg.LowPassFilter;
RELAX_cfg.LineNoiseFrequency = cfg.LineNoiseFrequency;
RELAX_cfg.NotchFilterType = cfg.LineNoiseFilterType;
RELAX_cfg.Do_MWF_Once = cfg.DoMWFOnce;
RELAX_cfg.Do_MWF_Twice = cfg.DoMWFTwice;
RELAX_cfg.Do_MWF_Thrice = cfg.DoMWFThrice;
RELAX_cfg.Perform_wICA_on_ICLabel = cfg.perform_wica_on_iclabel;
RELAX_cfg.Perform_ICA_subtract = cfg.perform_ica_subtract;


% checking to see if it's just one file
if RELAX_cfg.FilesToProcess == 1
    RELAX_cfg.SingleFile = 1;
end

% initialise eeglab
cd(eeglab_path);
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

% run RELAX
RELAX_Wrapper_beta(RELAX_cfg);