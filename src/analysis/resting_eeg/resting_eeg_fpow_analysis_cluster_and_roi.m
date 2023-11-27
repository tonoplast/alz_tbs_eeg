close all; clear;

% Path (repo, cap location, RELAX config file)
userprofilePath = getenv('USERPROFILE');
repoPath = [userprofilePath, filesep 'GitHub' filesep 'alz_tbs_eeg'];
analysisPath = [repoPath, filesep, 'src' filesep 'analysis'];
configPath = [repoPath, filesep 'src' filesep 'config'];
utilPath = [repoPath, filesep 'src' filesep 'utils'];
cfg_analysis = jsondecode(fileread([configPath filesep 'config_matlab.json'])).analysis_resting_eeg;
eeglab_path = [repoPath, filesep 'src' filesep 'toolbox' filesep cfg_analysis.eeglab_version];
resting_eeg_path = [analysisPath, filesep 'resting_eeg'];

% Analysis settings
active_group = {'301';'304';'306';'307';'309';'314';'315';'316';'318';'319';'321';'322';'324';'327';'330';'331';'333';'334';'337';'340';'343';'345';'347';'350';'351';'353';'356';'357';'358';'360'};
sham_group = {'302';'303';'305';'308';'310';'311';'312';'313';'317';'320';'323';'325';'326';'328';'329';'332';'335';'336';'338';'339';'341';'344';'346';'348';'349';'352';'354';'355';'359';'361'};
FOI = {[4, 7]; [8, 12]; [13, 29]; [30, 45]; [1, 45]};
FOI_name = {'theta'; 'alpha'; 'beta'; 'gamma'; 'broad'};
COI = {
    {'F1';'F3';'F5'}; 
    {'F2';'F4';'F6'}; 
    {'P1';'P3';'P5'}; 
    {'P2';'P4';'P6'};
    {'AF3'; 'AF4'; 'F7'; 'F5'; 'F3'; 'F1'; 'FZ'; 'F2'; 'F4'; 'F6'; 'F8'; 'FC5'; 'FC3'; 'FC1'; 'FCZ'; 'FC2'; 'FC4'; 'FC6'; 'C5'; 'C3'; 'C1'; 'CZ'; 'C2'; 'C4'; 'C6'; 'P7'; 'P5'; 'P3'; 'P1'; 'PZ'; 'P2'; 'P4'; 'P6'; 'P8'; 'PO3'; 'POZ'; 'PO4'; 'O1'; 'OZ'; 'O2'};
    };
COI_name = {'LF'; 'RF'; 'LP'; 'RP'; 'ALL'};

% Path (loading / saving)
inPath = [cfg_analysis.data_drive, filesep cfg_analysis.analysis_folder filesep 'Power' filesep];
outPath = [inPath, filesep 'results' filesep];
if not(isfolder(outPath))
    mkdir(outPath)
end

outPathCluster = [outPath, filesep, 'Cluster', filesep];
if not(isfolder(outPathCluster))
    mkdir(outPathCluster)
end

outPathROI = [outPath, filesep, 'ROI', filesep];
if not(isfolder(outPathROI))
    mkdir(outPathROI)
end

% initialise eeglab
cd(eeglab_path);
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;


%% load data
data_groups = {'eyesclosed_BL', 'eyesclosed_END', 'eyesopen_BL', 'eyesopen_END'};
data = struct();

for i = 1:numel(data_groups)
    current_group = data_groups{i};
    
    % Construct the file name dynamically
    file_name = fullfile(inPath, ['pow_', current_group, '.mat']);
    
    % Load the data and store it in the struct
    data.(current_group) = load(file_name, 'all_freq').all_freq;
    
    disp(['Loaded: ', current_group])
end

%% Filter data
cd(resting_eeg_path)
data_filtered = data;

% filtering data based on having both BL and END (as there are some
% drop-outs)
[data_filtered.eyesclosed_BL, data_filtered.eyesclosed_END, data_filtered.eyesclosed_BL_IDs] = filterDatasets(data.eyesclosed_BL, data.eyesclosed_END, false);
[data_filtered.eyesopen_BL, data_filtered.eyesopen_END, data_filtered.eyesopen_BL_IDs] = filterDatasets(data.eyesopen_BL, data.eyesopen_END, false);

% Split into EO and EC
data_ec = splitAndFilterDatasetsByLabels(data_filtered.eyesclosed_BL, data_filtered.eyesclosed_END, active_group, sham_group, false);
data_eo = splitAndFilterDatasetsByLabels(data_filtered.eyesopen_BL, data_filtered.eyesopen_END, active_group, sham_group, false);


%% Grand Average
grandAverages_ec = computeGrandAverages(data_ec, 'ec');
grandAverages_eo = computeGrandAverages(data_eo, 'eo');


% Adding Grand Averages for delta values (END - BL)
grandAverages_ec = operateOnGrandAverages(grandAverages_ec, 'BL', 'END', 'subtract', 'powspctrm', 'ec');
grandAverages_eo = operateOnGrandAverages(grandAverages_eo, 'BL', 'END', 'subtract', 'powspctrm', 'eo');


% plot check
% cfg = [];
% cfg.showlabels = 'yes'; 
% cfg.layout = 'quickcap64.mat'; % Provide appropriate layout
% cfg.colorbar = 'yes';
% 
% cfg.title = 'Active END';
% ft_topoplotTFR(cfg, grandAverages_ec.active_END);
% 
% cfg.title = 'Active BL';
% ft_topoplotTFR(cfg, grandAverages_ec.active_BL);
% 
% cfg.title = 'Subtraction: Active END - Active BL';
% ft_topoplotTFR(cfg, grandAverages_ec.active_delta);


%% Cluster stats & ROI (Dependent: within same group over time)



% sham EO
D1 = grandAverages_eo.sham_BL;
D2 = grandAverages_eo.sham_END;
runClusterStatisticsDepT(D1, D2, FOI, FOI_name, utilPath, outPathCluster);
runRoiAnalysisT(D1, D2, FOI, FOI_name, COI, COI_name, true, outPathROI)

% sham EC
D1 = grandAverages_ec.sham_BL;
D2 = grandAverages_ec.sham_END;
runClusterStatisticsDepT(D1, D2, FOI, FOI_name, utilPath, outPathCluster);
runRoiAnalysisT(D1, D2, FOI, FOI_name, COI, COI_name, true, outPathROI)

% active EO
D1 = grandAverages_eo.active_BL;
D2 = grandAverages_eo.active_END;
runClusterStatisticsDepT(D1, D2, FOI, FOI_name, utilPath, outPathCluster);
runRoiAnalysisT(D1, D2, FOI, FOI_name, COI, COI_name, true, outPathROI)

% active EC
D1 = grandAverages_ec.active_BL;
D2 = grandAverages_ec.active_END;
runClusterStatisticsDepT(D1, D2, FOI, FOI_name, utilPath, outPathCluster);
runRoiAnalysisT(D1, D2, FOI, FOI_name, COI, COI_name, true, outPathROI)


%% Cluster stats & ROI (Independent: Different group)

% EO BL (sham vs active)
D1 = grandAverages_eo.sham_BL;
D2 = grandAverages_eo.active_BL;
runClusterStatisticsIndepT(D1, D2, FOI, FOI_name, utilPath, outPathCluster);
runRoiAnalysisT(D1, D2, FOI, FOI_name, COI, COI_name, false, outPathROI)

% EO END (sham vs active)
D1 = grandAverages_eo.sham_END;
D2 = grandAverages_eo.active_END;
runClusterStatisticsIndepT(D1, D2, FOI, FOI_name, utilPath, outPathCluster);
runRoiAnalysisT(D1, D2, FOI, FOI_name, COI, COI_name, false, outPathROI)


% EC BL (sham vs active)
D1 = grandAverages_ec.sham_BL;
D2 = grandAverages_ec.active_BL;
runClusterStatisticsIndepT(D1, D2, FOI, FOI_name, utilPath, outPathCluster);
runRoiAnalysisT(D1, D2, FOI, FOI_name, COI, COI_name, false, outPathROI)

% EC END (sham vs active)
D1 = grandAverages_ec.sham_END;
D2 = grandAverages_ec.active_END;
runClusterStatisticsIndepT(D1, D2, FOI, FOI_name, utilPath, outPathCluster);
runRoiAnalysisT(D1, D2, FOI, FOI_name, COI, COI_name, false, outPathROI)


%% Cluster stats & ROI (Independent: Different group - Delta)

% EC (sham vs active)
D1 = grandAverages_ec.sham_delta;
D2 = grandAverages_ec.active_delta;
runClusterStatisticsIndepT(D1, D2, FOI, FOI_name, utilPath, outPathCluster);
runRoiAnalysisT(D1, D2, FOI, FOI_name, COI, COI_name, false, outPathROI)

% EO (sham vs active)
D1 = grandAverages_eo.sham_delta;
D2 = grandAverages_eo.active_delta;
runClusterStatisticsIndepT(D1, D2, FOI, FOI_name, utilPath, outPathCluster);
runRoiAnalysisT(D1, D2, FOI, FOI_name, COI, COI_name, false, outPathROI)


