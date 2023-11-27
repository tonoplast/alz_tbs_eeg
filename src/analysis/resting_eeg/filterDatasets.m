function [filtered_BL, filtered_END, common_subject_ids] = filterDatasets(data_BL, data_END, use_numbers)
    if use_numbers
        BL_subject_ids = cellfun(@(x) str2double(regexp(x.info, '\d+', 'match')), data_BL, 'UniformOutput', false);
        BL_subject_ids = vertcat(BL_subject_ids{:});
        END_subject_ids = cellfun(@(x) str2double(regexp(x.info, '\d+', 'match')), data_END, 'UniformOutput', false);
        END_subject_ids = vertcat(END_subject_ids{:});
    else
        % Get the subject IDs for data_BL
        BL_subject_ids = cellfun(@(x) str2double(x.info(1:3)), data_BL);
        
        % Get the subject IDs for eyesclosed_END
        END_subject_ids = cellfun(@(x) str2double(x.info(1:3)), data_END);
    end

    % Find subjects that exist in both BL and END datasets
    common_subject_ids = intersect(BL_subject_ids, END_subject_ids);

    % Filter eyesclosed_BL and eyesclosed_END datasets to keep only common subjects
    filtered_BL = data_BL(ismember(BL_subject_ids, common_subject_ids));
    filtered_END = data_END(ismember(END_subject_ids, common_subject_ids));
end