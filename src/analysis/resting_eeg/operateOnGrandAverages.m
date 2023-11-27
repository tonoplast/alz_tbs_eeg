function grandAverages = operateOnGrandAverages(grandAverages, timepoint1, timepoint2, operation, parameter, tag)
    % Extract fieldnames
    groups = fieldnames(grandAverages);
    groups = cellfun(@(x) strtok(x, '_'), groups, 'UniformOutput', false);
    groups = unique(groups);

    
    % Initialize configuration for ft_math
    cfg = [];
    cfg.operation = operation;
    cfg.parameter = parameter;

    for i = 1:length(groups)
        group = groups{i};
        % Check if both time points exist in the struct
        if isfield(grandAverages, [group, '_', timepoint1]) && isfield(grandAverages, [group, '_', timepoint2])
            % Perform the operation
            result = ft_math(cfg, grandAverages.([group, '_', timepoint2]), grandAverages.([group, '_', timepoint1]));
            % Save the result in the struct with the name format 'group_delta'
            grandAverages.([group, '_delta']) = result;
            grandAverages.([group, '_delta']).info = [tag, '_', group, '_delta'];
            grandAverages.([group, '_delta']).ids = grandAverages.([group, '_', timepoint1]).ids;
        else
            warning(['Skipping group: ', group, '. One of the timepoints is missing.']);
        end
    end
end
