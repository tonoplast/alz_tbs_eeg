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
inPath_neil = [inPath, filesep 'Transforms'];
outPath = [inPath, filesep 'Power'];

if not(isfolder(outPath))
    mkdir(outPath)
end

% initialise eeglab
cd(eeglab_path);
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;


data_groups = {'power_eyesclosed_BL', 'power_eyesclosed_END', 'power_eyesopen_BL', 'power_eyesopen_END'};


for i = 1:numel(data_groups)
    current_group = data_groups{i};
    
    file_extension = '.mat';
    file_type = ['*', current_group, '*'];

    fileList = dir(fullfile(inPath_neil, [file_type, file_extension]));

    % Initialize an empty cell array to store matching file names
    matchingFiles = {};
    all_freq = {};

    % Loop through the files and check if the name contains 'power_eyesclosed_BL'
    for j = 1:length(fileList)
        fileName = fileList(j).name;
        matchingFiles{j} = fileName;
        load([inPath_neil, filesep, matchingFiles{j}], 'powerfile')
        
        % Turning NaN to 0 for simplicity (this probably happened due to
        % edge effect?)
        powerfile.fourierspctrm(isnan(powerfile.fourierspctrm)) = 0;

        cfg = [];
        cfg.avgoverrpt = 'yes';  % average over trials (repetitions)
        freq = ft_freqdescriptives(cfg, powerfile);
        cfg = [];
        cfg.avgoverfreq = 'no';
        cfg.avgovertime = 'yes';  % average over time
        freq = ft_selectdata(cfg, freq);
        freq.info = matchingFiles{j};
        all_freq{j} = freq;       
        
        disp(['Done: ', current_group, ' >> ' matchingFiles{j}])
    end

    disp(['Done: ', current_group])
     
    save([outPath filesep current_group,'_neil.mat'], 'all_freq');
end


