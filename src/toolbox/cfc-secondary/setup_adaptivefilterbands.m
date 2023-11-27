
function [gamma,theta]=setup_adaptivefilterbands(cfg)
        if isempty(cfg)    
        cfg.sr=1000
        cfg.lo_bounds=[3 9]
        cfg.lo_step=0.1000
        cfg.lo_bandwidth=2
        cfg.hi_bounds=[20 60]
        cfg.hi_step=1
        cfg.hi_bandwidth='adaptive'
        end
        
%% Create frequency ranges
%
% Create a [2 x nsteps] matrix of low frequency bounds to scan. lo_freqs(1,:)
% and lo_freqs(2,:) contain the high and low bounds to filter on based on the
% center frequencies in lo_bounds, lo_steps and lo_bandwidth. High frequency
% bandlimits are created in a [2, n_hi_steps, n_lo_steps] matrix. This is to
% allow for an adaptive bandwidth based on the lowest low frequency (if
% requested).

n_lo_steps = (cfg.lo_bounds(2)-cfg.lo_bounds(1))/cfg.lo_step + 1;
lo_freqs = ones(2,n_lo_steps) .* repmat(cfg.lo_bounds(1):cfg.lo_step:cfg.lo_bounds(2),2,1);
lo_freqs(1,:) = lo_freqs(1,:) - cfg.lo_bandwidth/2;
lo_freqs(2,:) = lo_freqs(2,:) + cfg.lo_bandwidth/2;

n_hi_steps = floor((cfg.hi_bounds(2)-cfg.hi_bounds(1))/cfg.hi_step + 1);
hi_freqs = ones(2,n_hi_steps,n_lo_steps) .* repmat(cfg.hi_bounds(1):cfg.hi_step:cfg.hi_bounds(2),[2,1,n_lo_steps]);

if strcmp(cfg.hi_bandwidth,'adaptive')
    for idx = 1:n_lo_steps
        hi_bandwidth = lo_freqs(1,idx)+2;
        hi_freqs(1,:,idx) = hi_freqs(1,:,idx) - ones(1,n_hi_steps,1)*hi_bandwidth;
        hi_freqs(2,:,idx) = hi_freqs(2,:,idx) + ones(1,n_hi_steps,1)*hi_bandwidth;
    end
else
    size(hi_freqs)
    size(ones(n_hi_steps,n_lo_steps)*cfg.hi_bandwidth/2)
     hi_freqs(1,:,:) = hi_freqs(1,:,:) - ones(1,n_hi_steps,n_lo_steps)*cfg.hi_bandwidth/2;
     hi_freqs(2,:,:) = hi_freqs(2,:,:) + ones(1,n_hi_steps,n_lo_steps)*cfg.hi_bandwidth/2;

end
theta=lo_freqs;
gamma=hi_freqs;