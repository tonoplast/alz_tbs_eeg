function runRoiAnalysisT(D1, D2, FOI, FOI_name, COI, COI_name, paired, outPath)
       
    % Check if the lengths of FOI and FOI_name are the same
    if length(FOI) ~= length(FOI_name)
        error('FOI and FOI_name must have the same length.');
    end

    if length(COI) ~= length(COI_name)
        error('channels and COI_name must have the same length.');
    end

    numFreq = length(FOI);
    numChannelGroups = length(COI);

    % Initialize storage for averages
    avg_data1 = cell(numFreq, numChannelGroups);
    avg_data2 = cell(numFreq, numChannelGroups);
    p_values = zeros(numFreq, numChannelGroups);
   
    % Initialize an empty table
    dataTable = table();
    colNamesP = cell(numFreq * numChannelGroups, 1);

    
    % Populate the table with data from avg_data1 and avg_data2
    for f = 1:numFreq
        for c = 1:numChannelGroups
            colName1 = sprintf('%s %s %s', D1.info, COI_name{c}, FOI_name{f});
            colName2 = sprintf('%s %s %s', D2.info, COI_name{c}, FOI_name{f});
            colNamesP{(f - 1) * numChannelGroups + c} = sprintf('%s %s %s %s %s', D1.info, 'vs', D2.info, COI_name{c}, FOI_name{f});

            % Update configuration
            cfg = [];
            cfg.channel = COI{c};
            cfg.frequency = FOI{f};
    
            % Extract and average for data1
            selectedData1 = ft_selectdata(cfg, D1);
            avg_freq_data1 = mean(squeeze(mean(selectedData1.powspctrm, 3)), 2); % Average over frequency & channels
            avg_data1{f, c} = avg_freq_data1;

            % Extract and average for data2
            selectedData2 = ft_selectdata(cfg, D2);
            avg_freq_data2 = mean(squeeze(mean(selectedData2.powspctrm, 3)), 2); % Average over frequency & channels
            avg_data2{f, c} = avg_freq_data2;

            % Calculate difference in lengths
            lengthDiff = abs(length(avg_freq_data1) - length(avg_freq_data2));
            
            % Store data in the table
            dataTable.(colName1)(1:length(avg_freq_data1)) = avg_freq_data1;
            dataTable.(colName2)(1:length(avg_freq_data2)) = avg_freq_data2;
            
            % If there's a difference in lengths, replace the last 'lengthDiff' entries of the shorter data with NaN
            if length(avg_freq_data1) < length(avg_freq_data2)
                dataTable.(colName1)(end-lengthDiff+1:end) = NaN;
            elseif length(avg_freq_data2) < length(avg_freq_data1)
                dataTable.(colName2)(end-lengthDiff+1:end) = NaN;
            end

            % Paired T-test
            if paired
                [~, p_values(f, c)] = ttest(avg_freq_data1, avg_freq_data2);
            else
                [~, p_values(f, c)] = ttest2(avg_freq_data1, avg_freq_data2);
            end
        end
    end
    

    % adding ID to the data
    ID1 = D1.ids;
    ID2 = D2.ids;

    len1 = length(ID1);
    len2 = length(ID2);
    maxLength = max(len1, len2);
    
    % Ensure ID1 and ID2 are column vectors
    if isrow(ID1)
        ID1 = ID1';
    end
    
    % Pad ID1 if it's shorter
    if len1 < maxLength
        ID1 = [ID1; repmat({''}, maxLength - len1, 1)];
    end
    
    if isrow(ID2)
        ID2 = ID2';
    end
    
    % Pad ID1 if it's shorter
    if len2 < maxLength
        ID2 = [ID2; repmat({''}, maxLength - len1, 1)];
    end
    
    dataTable.ID1 = str2double(ID1);
    dataTable.ID2 = str2double(ID2);

    dataTable = [dataTable(:,end), dataTable(:,1:end-1)];
    dataTable = [dataTable(:,end), dataTable(:,1:end-1)];

    % Define the file name for the CSV file
    fileNameCSV = [outPath, filesep, D1.info, '_vs_', D2.info, '.csv'];
    
    % Use writetable to save the table as a CSV file
    writetable(dataTable, fileNameCSV);

    % Create a table using colNames and p_values_vector
    p_table = table(colNamesP, reshape(p_values',1,[])', 'VariableNames', {'Comparisons', 'P_Values'});
    p_fileNameCSV = [outPath, filesep, D1.info, '_vs_', D2.info, '_p_vals.csv'];
    writetable(p_table, p_fileNameCSV);

    % Preallocate storage
    numFreq = numel(FOI);
    numCOI = numel(COI);
    
    bar_data1 = zeros(numFreq, numCOI);
    std_data1 = zeros(numFreq, numCOI);
    bar_data2 = zeros(numFreq, numCOI);
    std_data2 = zeros(numFreq, numCOI);
    
    % Calculate bar plot data and standard deviations for avg_data1 and avg_data2
    for f = 1:numFreq
        for c = 1:numCOI
            bar_data1(f, c) = mean(avg_data1{f, c});
            std_data1(f, c) = std(avg_data1{f, c});
            
            bar_data2(f, c) = mean(avg_data2{f, c});
            std_data2(f, c) = std(avg_data2{f, c});
        end
    end
    
 
    % Set figure properties
    figure('Position', [100, 100, 1500, 1000], 'Color', 'w'); % 'Position' adjusts the size of the figure
    
    % Define custom bar colors for 'Data 1' and 'Data 2'
    barColors = {[0.4, 0.6, 0.8]; [0.8, 0.4, 0.4]};
    
    for f = 1:numFreq
        for c = 1:numCOI
            % Create a new subplot for each frequency
            subplot(numFreq, numCOI, (f - 1) * numCOI + c);
    
            % Extract data for current frequency and channel
            curr_data1 = bar_data1(f, c);
            curr_data2 = bar_data2(f, c);
            curr_std1 = std_data1(f, c);
            curr_std2 = std_data2(f, c);
    
            % Bar plot with error bars for 'Data 1' with custom color
            bar(1, curr_data1, 'FaceColor', barColors{1}, 'EdgeColor', 'none');
            hold on;
            errorbar(1, curr_data1, curr_std1, 'k.', 'LineStyle', 'none', 'LineWidth', 1.5);
    
            % Bar plot with error bars for 'Data 2' with custom color
            bar(2, curr_data2, 'FaceColor', barColors{2}, 'EdgeColor', 'none');
            errorbar(2, curr_data2, curr_std2, 'k.', 'LineStyle', 'none', 'LineWidth', 1.5);
    
            % Add significant stars if p-value < 0.05
            if p_values(f, c) < 0.05
                maxVal = max(curr_data1 + curr_std1, curr_data2 + curr_std2);
                
                % Add padding to the title
                title([FOI_name{f}, '-', COI_name{c}], 'FontSize', 12, 'FontWeight', 'normal', 'Margin', 5);
                
                % Make * red and add padding
                text(1.5, maxVal + 0.05 * abs(maxVal), '*', 'FontSize', 30, 'FontWeight', 'bold', 'Color', 'red');
            else
                % No significance, display the title without padding
                title([FOI_name{f}, '-', COI_name{c}], 'FontSize', 12, 'FontWeight', 'normal');
            end
            
            % Adjust y-axis limits
            maxY = max(curr_data1 + curr_std1, curr_data2 + curr_std2);
            minY = min(curr_data1 - curr_std1, curr_data2 - curr_std2);
            deltaY = maxY - minY;
            ylim([minY - 0.05 * abs(minY), maxY + 0.05 * abs(maxY)]);  % Extend y-axis limits by 5%
    
            % Set X-axis and title properties
            xticks([1, 2]);
            xticklabels({D1.info, D2.info});
            xtickangle(45); % Rotate the x-axis labels by 45 degrees
            if strcmp(COI_name{c}, 'ALL')
                title([FOI_name{f}, '-', COI_name{c}], 'FontSize', 12, 'FontWeight', 'normal');
            else
                title([FOI_name{f}, '-', COI_name{c}, ' (' strjoin(COI{c}, '/'), ')'], 'FontSize', 12, 'FontWeight', 'normal');
            end
            ylabel('Mean Power', 'FontSize', 10);
    
            set(gca, 'FontSize', 9, 'Box', 'off', 'LineWidth', 1);
            grid on;
            hold off;
        end
        fprintf('%s'': completed', [D1.info, ' vs ', D2.info, ' -- ', FOI_name{f, 1}]);
        fprintf('\n');
    end
    

    % Add a super title for the entire figure
    sgtitle(['ROI Analysis of ', D1.info, ' vs ', D2.info], 'FontSize', 15, 'FontWeight', 'bold');
    
    % Save the current figure as a PNG file (you can change the format)
    fileFormat = 'png'; % or 'jpeg', 'pdf', etc.
    fileName = [D1.info, '_vs_', D2.info]; % Specify the desired file name without the extension
    
    % Use saveas to save the current figure
    saveas(gcf, [outPath, filesep, fileName, '.', fileFormat]);      

end
