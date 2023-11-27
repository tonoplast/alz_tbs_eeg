close all; clear;

% config
% Path (repo and cap location)
userprofilePath = getenv('USERPROFILE');
repoPath = [userprofilePath, filesep 'GitHub' filesep 'tram_protocol_eeg'];
configPath = [repoPath, filesep 'src' filesep 'config'];

% config (processing)
cfg = jsondecode(fileread([configPath filesep 'config_matlab.json'])).postprocessing;

eeglab_path = [repoPath, filesep 'src' filesep 'toolbox' filesep cfg.eeglab_version];

cd(eeglab_path);
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

% Path (loading / saving)
inPath = [cfg.data_drive, filesep cfg.starting_folder filesep 'processed_EEG' filesep 'RELAXProcessed' filesep 'Cleaned_Data' filesep 'Epoched'];
outPath_eoec = [inPath, filesep, 'eoec'];

if not(isfolder(inPath))
    mkdir(inPath)
end

% if splitting into eo ec, then make the folder
if cfg.split_eo_ec  
    if not(isfolder(outPath_eoec))
        mkdir(outPath_eoec)
    end
end


% Get a list of all the files and folders in the folder
files = dir(inPath);

if numel(files) == 0
    error('There is no file in the "EEG_data_collected_today" folder! Please make sure the raw files are in there.')
end

% Loop through each file/folder
for i = 1:numel(files)
    % Skip over the '.' and '..' entries, which refer to the current and parent directories
    if strcmp(files(i).name, '.') || strcmp(files(i).name, '..')
        continue;
    end
    
    % If the current file/folder is a directory, skip them
    if files(i).isdir
        % Display the name of the folder
        disp(['Found folder: ' files(i).name '. Will ignore.']);
    elseif contains(files(i).name, cfg.file_extension)
        file_to_process = fullfile(inPath, files(i).name);
        disp(['Processing: ' file_to_process])

        [pathstr, filename, ext] = fileparts(file_to_process);

        % loading data (keystrokes off)
        EEG = pop_loadset(file_to_process);
        EEG = eeg_checkset( EEG );
               
        % if need to split, then save separately
        if cfg.split_eo_ec
            % getting epoch number to separate eyes open and eyes closed
            for i = 1:numel(EEG.event)
                if contains(EEG.event(i).type, 'eye', 'IgnoreCase', true)
                    last_eye_epoch = EEG.event(i).epoch;
                end
                last_epoch = EEG.event(i).epoch;
            end

            accepted_percentage = 0.2;
            plus_minus = last_epoch/2 * accepted_percentage;

            if last_eye_epoch - plus_minus < last_epoch/2 && last_epoch/2 < last_eye_epoch + plus_minus
                epoch_number_to_cut = last_eye_epoch;
            else
                epoch_number_to_cut = round(last_epoch/2);
            end

            EEG_firsthalf = EEG;
            EEG_firsthalf = pop_selectevent( EEG, 'epoch',[1:epoch_number_to_cut] ,'deleteevents','off','deleteepochs','on','invertepochs','off');
            EEG_firsthalf = eeg_checkset( EEG_firsthalf );

            EEG_secondhalf = EEG;
            EEG_secondhalf = pop_selectevent( EEG, 'epoch',[epoch_number_to_cut:last_epoch] ,'deleteevents','off','deleteepochs','on','invertepochs','off');
            EEG_secondhalf = eeg_checkset( EEG_secondhalf );

            pop_saveset(EEG_firsthalf, 'filename', [filename, '_eo.set'],'filepath', outPath_eoec); %outPath_eo);
            pop_saveset(EEG_secondhalf, 'filename', [filename, '_ec.set'],'filepath', outPath_eoec); %outPath_ec);
        
        else
            disp('No action (file splitting) taken.')
        end

        
        STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];

    else
        disp(['Not processing: ' files(i).name])
    end

end