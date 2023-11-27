function data_struct = splitAndFilterDatasetsByLabels(data_BL, data_END, active_group_cell, sham_group_cell, use_numbers)

    if use_numbers
        BL_subject_ids = cellfun(@(x) str2double(regexp(x.info, '\d+', 'match')), data_BL, 'UniformOutput', false);
        BL_subject_ids = cellfun(@num2str, BL_subject_ids, 'UniformOutput', false)';

        END_subject_ids = cellfun(@(x) str2double(regexp(x.info, '\d+', 'match')), data_END, 'UniformOutput', false);
        END_subject_ids = cellfun(@num2str, END_subject_ids, 'UniformOutput', false)';
    else
        % Get the subject IDs for eyesclosed_BL
        BL_subject_ids = cellfun(@(x) num2str(str2double(x.info(1:3))), data_BL, 'UniformOutput', false);
    
        % Get the subject IDs for eyesclosed_END
        END_subject_ids = cellfun(@(x) num2str(str2double(x.info(1:3))), data_END, 'UniformOutput', false);
    end
    

    % Split subjects into active and sham groups
    active_BL_indices = ismember(BL_subject_ids, active_group_cell);
    sham_BL_indices = ismember(BL_subject_ids, sham_group_cell);
    
    active_END_indices = ismember(END_subject_ids, active_group_cell);
    sham_END_indices = ismember(END_subject_ids, sham_group_cell);

    % Filter BL and END datasets based on active and sham groups
    active_BL = data_BL(active_BL_indices);
    active_END = data_END(active_END_indices);

    sham_BL = data_BL(sham_BL_indices);
    sham_END = data_END(sham_END_indices);

    % adding IDs
    active_IDs = BL_subject_ids(active_BL_indices);
    sham_IDs = BL_subject_ids(sham_BL_indices);

    data_struct = struct();
    data_struct.active_BL = active_BL;
    data_struct.active_END = active_END;
    data_struct.sham_BL = sham_BL;
    data_struct.sham_END = sham_END;
    data_struct.active_IDs = active_IDs;
    data_struct.sham_IDs = sham_IDs;
end