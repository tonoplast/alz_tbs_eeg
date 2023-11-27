function grandAverages = computeGrandAverages(data, tag)
    % Define the frequency analysis configuration (cfg)
    cfg = [];
    cfg.keepindividual = 'yes'; % Keep individual data in the grand average

    % Initialize a struct to store the grand averages for each group
    grandAverages = struct();

    % Loop through the fields of the data struct
    data_fields = fieldnames(data);
    id_fields = data_fields(contains(data_fields, 'IDs'));
    data_fields_without_ids = data_fields(~contains(data_fields, 'IDs'));

    for i = 1:numel(data_fields_without_ids)
        current_group = data_fields_without_ids{i};

        % Compute the grand average for the current group
        grandAverages.(current_group) = ft_freqgrandaverage(cfg, data.(current_group){:});
        grandAverages.(current_group).info = [tag, '_', current_group];
        
        % getting id into each grand averages
        which_group = id_fields(contains(id_fields, strtok(current_group, '_'), "IgnoreCase", true));
        grandAverages.(current_group).ids = data.(which_group{1});

        % Display a message indicating the completion of grand average computation
        disp(['Grand average computed for group: ', [tag, '_', current_group]]);
    end
end
