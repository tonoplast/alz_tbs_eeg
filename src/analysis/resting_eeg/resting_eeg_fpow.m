close all; clear;

% Path (repo, cap location, RELAX config file)
userprofilePath = getenv('USERPROFILE');
repoPath = [userprofilePath, filesep 'GitHub' filesep 'alz_tbs_eeg'];
configPath = [repoPath, filesep 'src' filesep 'config'];
utilPath = [repoPath, filesep 'src' filesep 'utils'];
cfg_analysis = jsondecode(fileread([configPath filesep 'config_matlab.json'])).analysis_resting_eeg;
eeglab_path = [repoPath, filesep 'src' filesep 'toolbox' filesep cfg_analysis.eeglab_version];


% Path (loading / saving)
inPath = [cfg_analysis.data_drive, filesep cfg_analysis.analysis_folder filesep];
outPath = [inPath, filesep 'Power'];

if not(isfolder(outPath))
    mkdir(outPath)
end

% initialise eeglab
cd(eeglab_path);
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

data_groups = {'eyesclosed_BL', 'eyesclosed_END', 'eyesopen_BL', 'eyesopen_END'};

for i = 1:numel(data_groups)
    current_group = data_groups{i};
    
    file_extension = '.set';
    file_type = ['*', current_group, '*'];
    files = dir(fullfile(inPath, [file_type, file_extension]));
    
    % Initialize an empty cell array to store the file names
    setFiles = cell(1, numel(files));
    
    all_freq = {};
    % Loop through the files and store the names of .set files
    for j = 1:numel(files)
        setFiles{j} = files(j).name;
        EEG = pop_loadset('filename', setFiles{j}, 'filepath', inPath);
        
        %convert to fieldtrip
        ftData = eeglab2fieldtrip(EEG, 'preprocessing');
    
    
        cfg = [];
        cfg.method = 'mtmfft'; % Multi-taper method (you can choose other methods if needed)
        cfg.output = 'pow';
        cfg.foi = 1:1:45; % Frequency of interest (in Hz)
        cfg.taper = 'hanning';
        cfg.tapsmofrq = 2; % Smoothing applied to the spectrum
        cfg.keeptrials = 'no'; % Average across trials
        freq = ft_freqanalysis(cfg, ftData); % your_data should be your segmented EEG data
        freq.info = files(j).name;
        % Store the freq structure in the all_freq cell array
        all_freq{j} = freq;

        disp(['Done: ', current_group, ' >> ' setFiles{j}])


    end

    disp(['Done: ', current_group])
    
    save([outPath filesep 'pow', '_', current_group,'.mat'], 'all_freq');
end

