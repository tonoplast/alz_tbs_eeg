function data_struct = splitAndFilterDatasetsByPercentage(data_BL, data_END, threshold)
    % Get the subject IDs for eyesclosed_BL
    BL_subject_ids = cellfun(@(x) str2double(x.info(1:3)), data_BL);

    % Get the subject IDs for eyesclosed_END
    END_subject_ids = cellfun(@(x) str2double(x.info(1:3)), data_END);

    % Split subjects into active and sham groups
    active_BL_indices = BL_subject_ids < threshold;
    sham_BL_indices = BL_subject_ids >= threshold;

    active_END_indices = END_subject_ids < threshold;
    sham_END_indices = END_subject_ids >= threshold;

    % Filter BL and END datasets based on active and sham groups
    active_BL = data_BL(active_BL_indices);
    active_END = data_END(active_END_indices);

    sham_BL = data_BL(sham_BL_indices);
    sham_END = data_END(sham_END_indices);

    data_struct = struct();
    data_struct.active_BL = active_BL;
    data_struct.active_END = active_END;
    data_struct.sham_BL = sham_BL;
    data_struct.sham_END = sham_END;
end