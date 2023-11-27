close all; clear;

% Path (repo and cap location)
userprofilePath = getenv('USERPROFILE');
repoPath = [userprofilePath, filesep 'GitHub' filesep 'tram_protocol_eeg'];
configPath = [repoPath, filesep 'src' filesep 'config'];
cfg_prep = jsondecode(fileread([configPath filesep 'config_matlab.json'])).preparation;
cfg_post = jsondecode(fileread([configPath filesep 'config_matlab.json'])).postprocessing;
cfg_pac = jsondecode(fileread([configPath filesep 'config_matlab.json'])).pac;

% config
cfg.lo_bounds = cfg_pac.lo_bounds';
cfg.lo_step = cfg_pac.lo_step';
cfg.lo_bandwidth = cfg_pac.lo_bandwidth;
cfg.hi_bounds = cfg_pac.hi_bounds';
cfg.hi_step = cfg_pac.hi_step;
cfg.hi_bandwidth = cfg_pac.hi_bandwidth;
frontchan = cfg_pac.frontchan; %% channel of interest 1
backchan = cfg_pac.backchan; %% channel of interest 2
% frontchan = {'F3','FZ','F4'}; %% channel of interest 1
% backchan = {'P1','PZ','P3'}; %% channel of interest 2

reverse_order = cfg_pac.reverse_order;

eeglab_path = [repoPath, filesep 'src' filesep 'toolbox' filesep cfg_prep.eeglab_version];
tgc_path = [repoPath, filesep 'src' filesep 'toolbox' filesep 'cfc-secondary'];
cfc_path = [repoPath, filesep 'src' filesep 'toolbox' filesep 'cfc-master'];

caploc = [configPath filesep cfg_prep.caploc];
filterfolder = [matlabroot, '\toolbox\signal\signal\']; %% To use Matlab filter and not Fieldtrip one
    
addpath(genpath(tgc_path))
addpath(genpath(cfc_path))

cd(eeglab_path);
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;


% cfg_post.set_srate = 256 %DELETE THIS AFTER!!!

eye_status = cfg_pac.eye_status; %{'eyes_open'; 'eyes_closed'};

parpool('local')

for eoec = 1:size(eye_status, 1)
    
    e_s = eye_status{eoec,1};

    data_path = [cfg_prep.data_drive, '\EEG_data_collected_today\processed_EEG\RELAXProcessed\Cleaned_Data\Epoched\eoec'];
    output_path = [cfg_prep.data_drive, '\EEG_data_collected_today\processed_EEG\RELAXProcessed\Cleaned_Data\Epoched\results_matlab'];
 
    if strcmp(e_s, 'eyes_open')
        eo_ec = 'eo';
    elseif strcmp(e_s, 'eyes_closed')
        eo_ec = 'ec';
    end

    
    ID = {'H301';'H302';'H303';'H304';'H305';'H306';'H307';'H308';'H309';'H310';'H311';'H312';'H313';'H314';'H315';'H316';'H317';'H318';'H319';'H320'};

    if reverse_order == true
        ID = flipud(ID);
    end
        
     Sesh = {'Ind'; 'S30'; 'S50'};
     tp = {'T0'; 'T2'};
    
    
    for ids = 1:size(ID,1)
        for sessions = 1:size(Sesh,1)
            for timepoints = 1:size(tp,1)

                id_sesh_tp = [ID{ids,1},'_',Sesh{sessions,1},'_',tp{timepoints,1}];

                eeg_filename = [id_sesh_tp,'_resting_downsampled_RELAX_Epoched_', eo_ec, '.set'];

                for h=1:cfg_pac.how_many_permutations %To run Theta-gamma coupling 5 times --> Depending on your need

                    EEG = pop_loadset('filename', eeg_filename, 'filepath', data_path);
                    EEG = eeg_checkset( EEG );
                    
                    % baseline correction
                    EEG = pop_rmbase( EEG, []); 

                    % average re-referencing
                    rereference_without_these_electrodes = find(ismember({EEG.chanlocs.labels}, cfg_post.rereference_without_these_electrodes));
                    EEG = pop_reref(EEG, [],'exclude', rereference_without_these_electrodes);

                    % downsampling -- This should be removed since it is bad to do it on epoched data
                    % EEG = pop_resample(EEG, cfg_post.set_srate); %% down-sampling should reduce time it takes.
                    % [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'gui','off'); 
                    
                    label_multiplier = 1000 / EEG.srate; % to be used later for edge effect %

                    [ALLEEG, EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
                    THISCHAN=find(ismember({EEG.chanlocs.labels},backchan)); %% gamma
                    THATCHAN=find(ismember({EEG.chanlocs.labels},frontchan)); %% theta
                    [ch,pnts,eps]=size(EEG.data);


                    %% selecting epochs and shuffling %%%
                    I = 1:eps;
                    I = shuffle(I);
                    keep = [I(1:round(eps*0.75))];


                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %%%%%%%%%%%%%%%%%%%%%%%% PAC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                        try
                            %these=unique([EEG.event(:).epoch]);
                            these=keep;
                        catch
                            disp('Not enough epochs');
                            these=1:eps; % use as many as i got
                        end

                        ALLTRIALS=EEG;
                        EEG = pop_select( EEG,'trial',these);

                        %%% all trials or use above
                        % EEG=ALLTRIALS;
                        EEG = eeg_checkset( EEG );

                        clear glmf mi plv vtk
                        [ch,pnts,eps]=size(EEG.data([THISCHAN],:,:));

                        %% DATA zero padding and reshaping

                        datarange = [1:EEG.pnts];

                        data1 = EEG.data([THISCHAN],datarange,:);
                        data2 = EEG.data([THATCHAN],datarange,:);

                        %%% meaning the data if more than 1 chan
                        data1 = mean(data1,1);
                        data2 = mean(data2,1);

                        label = zeros(1,pnts,eps); %% zero padding labelling for edge effect problem
                        label(:,[1:(500/label_multiplier),end-(500/label_multiplier):end],:)=1; %% edge effect removal step (taking sampling rate into account)

                        dat1 = squeeze(data1)';
                        dat2 = squeeze(data2)';
                        label=squeeze(label)';


                        [m,n]=size(dat1);
                        data1 =reshape([dat1'],[1,m*n]);
                        data2= reshape([dat2'],[1,m*n]);
                        label= logical(reshape([label'],[1,m*n]));
                        label_ori = label;

                        % alternative label %
                        label_alt = zeros(1,length(data1));
                        label_alt(:,[1:(2000/label_multiplier),end-(2000/label_multiplier):end])=1;
                        label = logical(label_alt);

                        select=[];
                        srate=EEG.srate;

                            %% Filtering (adaptive - script attached)
                        %srate=1000
                        cfg.sr=srate;
                        [gamma,theta]=setup_adaptivefilterbands(cfg);

                        %cd(filterfolder);
                        %% PAC estimation --> calculates four different methods

                        bb=size(gamma,2);
                        cc=size(theta,2);
                        aa = size(data1,3);

                        tic

                        clear glmf mi plv vtk
                        [glmf,plv,vtk,mi] = deal(zeros(bb,cc,aa)); % -- initialize output matrices

                        for a = 1:aa
                            raw_signal = data1(1,:,a);
                            raw_signal2=data2(1,:,a);
                            label=logical(label);

                            ntimepoints = [length(raw_signal)] - [sum(label)];
                            for b = 1:bb
                                [x_gamma,y_gamma]=butter(2,[gamma(1,b) gamma(2,b)]/(srate/2),'bandpass');
                                gamma_wave= filtfilt(x_gamma,y_gamma, double(raw_signal));
                                gamma_z = hilbert(gamma_wave);
                                gamma_amp= abs(gamma_z);

                                for c = 1:cc
                                    [x_theta,y_theta]=butter(2,[theta(1,c) theta(2,c)]/(srate/2),'bandpass');
                                    theta_wave= filtfilt(x_theta,y_theta, double(raw_signal2));
                                    theta_z=hilbert(theta_wave);
                                    theta_phase=angle(theta_z);

                                    gamma_ampfilt= filtfilt(x_theta,y_theta, double(gamma_amp));

                                    gamma_amp_z=hilbert(gamma_ampfilt);
                                    gamma_amp_phase = angle(gamma_amp_z);
                                    %%%%%%%%%%

                                    thetaphase = theta_phase(~label);
                                    gammapow = gamma_amp(~label);
                                    gammapow1=gamma_ampfilt(~label);
                                    gammapowfilt =gamma_ampfilt(~label);
                                    gammapowphase=gamma_amp_phase(~label);
                                    nbins = 18;

                                    %%%% Tort's Modulation Index (Tort et al., 2010)
%                                     thetaphase_bin = ceil( tiedrank( thetaphase ) / (ntimepoints / nbins) ); % -- bin the theta phase angles into nbins -- NOTE: tiedrank also exists in eeglab toolbox; when added to path, may cause conflict
%                                     gammapow_bin = zeros(1,nbins);
%                                     for k=1:nbins
%                                         gammapow_bin(k) = squeeze(mean(gammapow(thetaphase_bin==k))); % -- compute mean gamma power in each bin
%                                     end
%                                     gammapow_bin = gammapow_bin ./ sum(gammapow_bin); % -- normalize

%                                     mi(b,c,a) = (log(nbins) + sum(gammapow_bin.*log(gammapow_bin)) ) ./ log(nbins); % -- compute MI

%                                     debias_term = mean(exp(1i*thetaphase)); % -- this is the phase clustering bias
%                                     dpac(b,c,a) = abs(mean( (exp(1i*thetaphase) - debias_term) .* gammapow)); % -- which is subtracted here

                                    %%%  -- Phase Locking Value (Cohen, 2008; Colgin et al 2009)
%                                     plv(b,c,a) = abs(mean(exp(1i*( thetaphase - angle(hilbert(detrend(gammapow))) ))));

                                    %%%     Voytek method
%                                     vtk(b,c,a) = cfc_est_voytek( thetaphase', gammapowfilt');

                                    % glm method with theta filtered (Used this method)
                                    glmf(b,c,a) = cfc_est_glm(thetaphase,gammapowfilt); %% using this at the end

                                    % For masking (GLM)
                                    [out,stat] = cfc_est_glmstats(thetaphase,gammapowfilt);
                                    glmf(b,c,a) = out;
                                    pvals=stat.stats.p;
                                    glmfP1(b,c,a)= pvals(1);
                                    glmfP2(b,c,a)= pvals(2);
                                    glmfP3(b,c,a)= pvals(3);
                                end
                                disp((b/bb)*100);
                                disp('%Still going..');
                            end

                        end
                        disp(toc);
                        disp('%Finished!!!');



                %% plot results
                    close all;

                    % just for cropping of the figure
                    theta=mean(theta,1); % makes axis for plots 
                    gamma=mean(gamma(:,:,1),1);

                    % Using glmf here  
                    TGC_plotter(glmf,theta,gamma,[4,8],[30,60],1,'GLMF')
%                     TGC_plotter(vtk,theta,gamma,[4,8],[30,60],1,'VTK');
%                     TGC_plotter(mi,theta,gamma,[4,8],[30,60],1,'MI');
%                     TGC_plotter(plv,theta,gamma,[4,8],[30,60],1,'PLV');
                    % TGC_plotter(dpac,theta,gamma,[4,8],[30,60],1,'dPAC');
                    
%                     pause;

                    %% Makes masking and saves

                    figure;
                    in=glmfP1;
                    moo=in;
                    pthresh= 0.05/(size(glmfP1,1)*size(glmfP1,2));%(31*41); % Set your threshold here.
                    moo((in<pthresh))=0;
                    moo((in>pthresh))=1;
                %     imshow(moo)

                    moo2(:,:,1)=[glmf];
                    moo2(:,:,2)=[glmfP1];
                    moo2(:,:,3)=[glmfP2];
                    moo2(:,:,4)=[glmfP3];
                    pval=logical(in>pthresh);
                    for i = 1:4
                        mi = moo2(:,:,i);     
                        mi(pval)=NaN;
                        moo3(:,:,i) = mi;
                     end

                    mkdir([output_path]);

                    cd([output_path])
                    fn=[id_sesh_tp, '_', eo_ec,  cfg_pac.output_filename_tag, '_GLMf_stat_repeat_' num2str(h)];
                    save(fn,'moo3');
                    save(['freqs_', eo_ec],'theta','gamma');

                    fn2=[id_sesh_tp, '_', eo_ec, cfg_pac.output_filename_tag, '_GLMf_stat_repeat_no_mask' num2str(h)];
                    save(fn2,'glmf');

                end

                STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[];
                disp(['Done ' id_sesh_tp])
                
                   close all;
                %% Loading saved TGCs

                cd([output_path])

                hh=[];hh=h;
                for i = 1:hh
                    load([id_sesh_tp, '_', eo_ec, cfg_pac.output_filename_tag, '_GLMf_stat_repeat_' num2str(i) '.mat'])
                    load(['freqs_', eo_ec '.mat'])    
                    moo4(:,:,:,i)=moo3;
                    clear moo3

                    load([id_sesh_tp, '_', eo_ec, cfg_pac.output_filename_tag, '_GLMf_stat_repeat_no_mask' num2str(i) '.mat'])
                    mooX(:,:,:,i) = glmf;
                end

                hh=[];hh=h;
                m=[]; s=1;
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
                measure_name = 'GLMf';
                TGC_plotter(m,theta,gamma,[4,8],[30,60],1,measure_name)


                % Average of TGCs (or median) --> This would be the stimulation
                % frequency
                TGC_plotter(mean(m,3),theta,gamma,[4,8],[30,60],1,'GLMf')
                TGC_plotter(median(m,3),theta,gamma,[4,8],[30,60],1,'GLMf')

                TGC_plotter(mean(squeeze(mooX),3),theta,gamma,[4,8],[30,60],1,'GLMf mean repeats')
                TGC_plotter(median(squeeze(mooX),3),theta,gamma,[4,8],[30,60],1,'GLMf median repeats')

                close all;

            end

        end
    end

end

delete(gcp)