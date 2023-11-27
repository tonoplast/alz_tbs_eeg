function runClusterStatisticsIndepT(D1, D2, FOI, FOI_name, utilPath, outPath)
    for f = 1:size(FOI, 1)
        freq_range = FOI{f, 1};
        
        % Load neighbours template
        load([utilPath filesep 'neighbours' filesep 'neighbours_template.mat']);
        
        % Run cluster-based permutation statistics
        cfg = [];
        cfg.channel     = {'all'};
        cfg.minnbchan        = 2; % Minimum number of channels for cluster
        cfg.clusteralpha = 0.05;
        cfg.clusterstatistic = 'maxsum';
        cfg.alpha       = 0.025; % 0.025 for two-tailed, 0.05 for one-tailed
        cfg.frequency   = freq_range;
        cfg.avgoverchan = 'no'; 
        cfg.avgoverfreq = 'yes';
        cfg.statistic   = 'indepsamplesT';
        cfg.numrandomization = 5000;
        cfg.correctm    = 'cluster';
        cfg.method      = 'montecarlo'; 
        cfg.tail        = 0;
        cfg.clustertail = 0;
        cfg.neighbours  = neighbours;

        % Enter number of participants from each dataset
        subj_D1 = size(D1.powspctrm, 1);
        subj_D2 = size(D2.powspctrm, 1);

        design = zeros(1, subj_D1 + subj_D2);
        design(1, 1:subj_D1) = 1;
        design(1, subj_D1 + 1:end) = 2;
        cfg.design = design;
        cfg.ivar  = 1; % The row number containing group identifier
        
        
        % Define variables for comparison
        [stat] = ft_freqstatistics(cfg, D1, D2);

        save([outPath, 'stats_', D1.info, '_vs_', D2.info, '_', FOI_name{f, 1}], 'stat', 'freq_range');

        cfg = [];
        cfg.alpha = 0.025;
        cfg.zparam = 'stat';
        cfg.layout = 'quickcap64.mat'; % 'easycapM11.mat'; % 'quickcap64.mat';
        cfg.gridscale = 100;
        cfg.colorbar = 'yes'; % 'yes' for on, 'no' for off
        cfg.highlightcolorpos = [0 0 1];
        cfg.highlightcolorneg = [1 1 0];
        cfg.colormap = jet;
        cfg.subplotsize = [1 1];
        cfg.zlim = [-3 3];
        cfg.highlightsymbolseries = ['*', 'x', '+', 'o', '.'];
        cfg.highlightsizeseries = [10 10 10 10 10];
        cfg.title = [D1.info, ' vs ', D2.info, ' -- ', FOI_name{f, 1}];

        
        try
            ft_clusterplot(cfg, stat);
        catch
            ft_topoplotTFR(cfg, stat);
        end
        
        fprintf('%s''stats complete: ', [D1.info, ' vs ', D2.info, ' -- ', FOI_name{f, 1}]);
        fprintf('\n');
    end
end
