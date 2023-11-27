
close all; clear;
which_drive = 'D:';
eeglab_path = [which_drive, '\tram_protocol_eeg\toolbox\eeglab2022.1'];
tgc_path = [which_drive, '\tram_protocol_eeg\tram_personal_pc\eeg_src\matlab\TGC'];
cfc_path = [which_drive, '\tram_protocol_eeg\toolbox\cfc-master\'];

addpath(genpath(tgc_path))
addpath(genpath(cfc_path))

% which_param = '1Hz_HighPass_MWF_01lostep_1bw';
which_param = '025Hz_HighPass_MWF_01lostep_2bw';

eye_status = {'eyes_open'; 'eyes_closed'};

ID = {'H301';'H302';'H303';'H304';'H305';'H306';'H307';'H308';'H309';'H310';'H311';'H312';'H313';'H314';'H315';'H316';'H317';'H318';'H319';'H320'};
Sesh = {'Ind'; 'S30'; 'S50'};
tp = {'T0'; 'T2'};

h = 5;

peak_freqs = [];

for eoec = 1:size(eye_status, 1)
    
    e_s = eye_status{eoec,1};
    if strcmp(e_s, 'eyes_open')
        eo_ec = 'eo';
    elseif strcmp(e_s, 'eyes_closed')
        eo_ec = 'ec';
    else
    end

    input_path = [which_drive, '\EEG_data_collected_today\processed_EEG\' which_param '\RELAXProcessed\Cleaned_Data\Epoched\results_matlab'];
    output_path = [input_path, filesep 'glmf'];
    mkdir(output_path)
    cd([input_path])

    for ids = 1:size(ID,1)
        for sessions = 1:size(Sesh,1)
            for timepoints = 1:size(tp,1)
                
                id_sesh_tp = [ID{ids,1},'_',Sesh{sessions,1},'_',tp{timepoints,1}];
                hh=[];hh=h;
                for i = 1:hh
                    load([id_sesh_tp, '_', eo_ec, '_GLMf_stat_repeat_' num2str(i) '.mat'])
                    load('freqs_eo.mat')    
                    moo4(:,:,:,i)=moo3;
                    clear moo3

                    load([id_sesh_tp, '_', eo_ec, '_GLMf_stat_repeat_no_mask' num2str(i) '.mat'])
                    mooX(:,:,:,i) = glmf;
                end

                hh=[];hh=h;m=[];s=1;
                for i=1:hh
                        m(:,:,s) = moo4(:,:,1,i);
                    s=s+1;
                end

                %%
                % Plotting each TGC with masking
%                 measure_names = {'GLMf';'VTK';'MI';'PLV';'dPAC'};
%                 for measure_name = 1:size(measure_names)
%                     TGC_plotter(m(:,:,measure_name),theta,gamma,[4,8],[30,60],1,measure_names{measure_name,1})
%                 end
                
                % plot different trials
%                 measure_name = 'GLMf';
%                 TGC_plotter(m,theta,gamma,[4,8],[30,60],1,measure_name)


                % Average of TGCs (or median) --> This would be the stimulation
                % frequency
%                 TGC_plotter(mean(m,3),theta,gamma,[4,8],[30,60],1,'GLMf')
%                 TGC_plotter(median(m,3),theta,gamma,[4,8],[30,60],1,'GLMf')

                recommended_frequencies = TGC_plotter(mean(squeeze(mooX),3),theta,gamma,[4,8],[30,60],1,'GLMf mean repeats');
                saveas(gcf, [output_path filesep id_sesh_tp, '_', eo_ec, '.png'])
%                 TGC_plotter(median(squeeze(mooX),3),theta,gamma,[4,8],[30,60],1,'GLMf median repeats');

                close all;
                
                peak_freqs = [peak_freqs; recommended_frequencies];
                
                end
            end
        end

        save([output_path filesep 'matlab_' eo_ec '.mat'], 'peak_freqs')
        peak_freqs = [];
  end
