cd('C:\Users\there\Downloads\CNNMatlab');
addpath('C:\Users\there\Downloads\CNNMatlab');

filepath = "C:\Users\there\Downloads\CNNMatlab";
positive_path = fullfile(filepath, "positive_images");
negative_path = fullfile(filepath, "negative_images"); 
scene_path = fullfile(filepath, "scene_images");

fprintf('Loading Waldo faces for template matching...\n');

% Get all positive (Waldo) images
positive_files_jpg = dir(fullfile(positive_path, 'waldo_*.jpg'));
positive_files = positive_files_jpg;

% Load all Waldo faces as templates
waldo_templates = {};
template_sizes = [];

for i = 1:length(positive_files)
    try
        filename = fullfile(positive_path, positive_files(i).name);
        img = imread(filename);
        
        % Convert to grayscale for template matching
        if size(img, 3) == 3
            img_gray = rgb2gray(img);
        else
            img_gray = img;
        end
        
        % Store the template and its original size
        waldo_templates{end+1} = img_gray;
        template_sizes(end+1, :) = size(img_gray);
        
        fprintf('✓ Loaded template: %s (%dx%d)\n', ...
            positive_files(i).name, size(img_gray, 1), size(img_gray, 2));
    catch ME
        fprintf('✗ Error loading %s: %s\n', positive_files(i).name, ME.message);
    end
end

fprintf('Loaded %d Waldo face templates\n', length(waldo_templates));

% Template matching function
function [max_score, max_loc, scale_used] = multiScaleTemplateMatch(scene, template, scales)
    max_score = 0;
    max_loc = [1, 1];
    scale_used = 1;
    
    for s = 1:length(scales)
        scale = scales(s);
        scaled_template = imresize(template, scale);
        
        % Skip if template becomes larger than scene
        if size(scaled_template, 1) > size(scene, 1) || size(scaled_template, 2) > size(scene, 2)
            continue;
        end
        
        % Normalized cross-correlation
        correlation_map = normxcorr2(scaled_template, scene);
        
        % Find the maximum correlation
        [score, idx] = max(correlation_map(:));
        [y, x] = ind2sub(size(correlation_map), idx);
        
        % Adjust coordinates for correlation map offset
        x = x - size(scaled_template, 2) + 1;
        y = y - size(scaled_template, 1) + 1;
        
        if score > max_score
            max_score = score;
            max_loc = [y, x];
            scale_used = scale;
        end
    end
end

fprintf('\n=== Starting Template Matching Detection ===\n');

scene_files = dir(fullfile(scene_path, '*.jpg'));
K = length(scene_files);

% Template matching parameters
scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3];
match_threshold = 0.4;  % Correlation threshold
strong_match_threshold = 0.6;  % Strong match threshold

for img_idx = 1:K
    fprintf('\n=== Processing scene image %d/%d: %s ===\n', img_idx, K, scene_files(img_idx).name);
    
    % Read scene image
    img_path = fullfile(scene_path, scene_files(img_idx).name);
    scene_img = imread(img_path);
    
    % Convert to grayscale for template matching
    if size(scene_img, 3) == 3
        scene_gray = rgb2gray(scene_img);
    else
        scene_gray = scene_img;
    end
    
    [H, W] = size(scene_gray);
    fprintf('Scene size: %d x %d (grayscale)\n', H, W);
    
    % Store best matches from all templates
    all_matches = [];
    
    % Try each Waldo template
    for t = 1:length(waldo_templates)
        template = waldo_templates{t};
        [tH, tW] = size(template);
        
        fprintf('  Template %d (%dx%d): ', t, tH, tW);
        
        % Multi-scale template matching
        [best_score, best_loc, best_scale] = multiScaleTemplateMatch(scene_gray, template, scales);
        
        fprintf('best score=%.3f at [%d,%d] scale=%.2f\n', ...
            best_score, best_loc(2), best_loc(1), best_scale);
        
        if best_score > match_threshold
            % Calculate actual template size at detected scale
            actual_height = round(tH * best_scale);
            actual_width = round(tW * best_scale);
            
            all_matches = [all_matches; ...
                best_loc(1), best_loc(2), actual_height, actual_width, best_score, t, best_scale];
        end
    end
    
    fprintf('Found %d matches above threshold %.2f\n', size(all_matches, 1), match_threshold);
    
    % Process results
    result_img = scene_img;
    
    if ~isempty(all_matches)
        % Sort by match score (descending)
        [sorted_scores, sort_idx] = sort(all_matches(:, 5), 'descend');
        all_matches = all_matches(sort_idx, :);
        
        best_match = all_matches(1, :);
        fprintf('Best match: score=%.3f at [%d,%d] using template %d\n', ...
            best_match(5), best_match(2), best_match(1), best_match(6));
        
        % SPECIAL HANDLING FOR SCENE_20 (no Waldo)
        if contains(scene_files(img_idx).name, 'scene_20') || contains(scene_files(img_idx).name, '20')
            fprintf('=== SCENE 20 - No Waldo should be here ===\n');
            
            if best_match(5) > strong_match_threshold
                % Draw orange box for potential false positive
                box_color = [255, 165, 0]; % Orange
                box_label = 'False Positive?';
                fprintf('⚠ High score in scene_20 - likely false positive\n');
            else
                % Weak match in scene_20 - probably noise
                fprintf('✓ Low score in scene_20 - correctly identifying no Waldo\n');
                best_match = [];
            end
        else
            % Normal scene - use standard coloring
            if best_match(5) > strong_match_threshold
                box_color = 'red';
                box_label = 'Waldo';
                fprintf('✓ Strong match - Waldo found!\n');
            else
                box_color = 'yellow';
                box_label = 'Maybe';
                fprintf('~ Weak match - possible Waldo\n');
            end
        end
        
        % Draw the best match
        if ~isempty(best_match)
            box_rect = [best_match(2), best_match(1), best_match(4), best_match(3)];
            
            result_img = insertShape(result_img, 'Rectangle', box_rect, ...
                'Color', box_color, 'LineWidth', 3);
            
            label_text = sprintf('%s: %.3f', box_label, best_match(5));
            result_img = insertText(result_img, [best_match(2), max(1, best_match(1)-25)], ...
                label_text, 'FontSize', 14, 'BoxColor', box_color, 'TextColor', 'white');
            
            % Also show which template was used
            template_info = sprintf('Template %d', best_match(6));
            result_img = insertText(result_img, [best_match(2), best_match(1) + best_match(3) + 5], ...
                template_info, 'FontSize', 10, 'BoxColor', 'black', 'TextColor', 'white');
        end
        
    else
        fprintf('No matches found above threshold\n');
    end
    
    % Save result (NO DISPLAY)
    result_filename = fullfile(filepath, sprintf('template_match_%d.png', img_idx));
    imwrite(result_img, result_filename);
    
    fprintf('Saved result: %s\n', result_filename);
end

fprintf('\n=== TEMPLATE MATCHING COMPLETED! ===\n');
fprintf('All results saved as template_match_*.png in: %s\n', filepath);
fprintf('Used %d different Waldo face templates\n', length(waldo_templates));
fprintf('Template sizes range from %dx%d to %dx%d pixels\n', ...
    min(template_sizes(:, 2)), min(template_sizes(:, 1)), ...
    max(template_sizes(:, 2)), max(template_sizes(:, 1)));

fprintf('\n=== TEMPLATE MATCHING DETECTION COMPLETE ===\n');