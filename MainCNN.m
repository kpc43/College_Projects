cd('C:\Users\there\Downloads\CNNMatlab');
addpath('C:\Users\there\Downloads\CNNMatlab');

filepath = "C:\Users\there\Downloads\CNNMatlab";
positive_path = fullfile(filepath, "positive_images");
negative_path = fullfile(filepath, "negative_images"); 
scene_path = fullfile(filepath, "scene_images");

fprintf('Loading training images...\n');

% Get all positive (Waldo) images
positive_files_jpg = dir(fullfile(positive_path, 'waldo_*.jpg'));
positive_files = positive_files_jpg;

% Get all negative (not Waldo) images
negative_files_png = dir(fullfile(negative_path, 'not_waldo_*.png'));

% Preload all images into memory with FIXED 64x64 resizing
fprintf('Preloading training images with FIXED 64x64 resizing...\n');
train_images = [];
train_labels = [];

target_size = [64 64]; % Fixed target size for ALL images

% Load positive images (Waldo faces) - resize to exactly 64x64
for i = 1:length(positive_files)
    try
        filename = fullfile(positive_path, positive_files(i).name);
        img = imread(filename);
        
        % RESIZE TO FIXED 64x64 - no exceptions
        img_resized = imresize(img, target_size);
        
        % Store the fixed-size image
        train_images = cat(4, train_images, img_resized);
        train_labels = [train_labels; categorical({'waldo'})];

        fprintf('✓ Loaded: %s -> %dx%d\n', ...
            positive_files(i).name, target_size(1), target_size(2));
    catch ME
        fprintf('✗ Error loading %s: %s\n', positive_files(i).name, ME.message);
    end
end

% Load negative images (not Waldo) - resize to exactly 64x64
for i = 1:length(negative_files_png)
    try
        filename = fullfile(negative_path, negative_files_png(i).name);
        img = imread(filename);
        
        % RESIZE TO FIXED 64x64 - no exceptions
        img_resized = imresize(img, target_size);
        
        % Store the fixed-size image
        train_images = cat(4, train_images, img_resized);
        train_labels = [train_labels; categorical({'not_waldo'})];

        fprintf('✓ Loaded: %s -> %dx%d\n', ...
            negative_files_png(i).name, target_size(1), target_size(2));
    catch ME
        fprintf('✗ Error loading %s: %s\n', negative_files_png(i).name, ME.message);
    end
end

fprintf('Successfully loaded %d training images\n', size(train_images, 4));
fprintf('Class distribution: %d waldo, %d not_waldo\n', ...
    sum(train_labels == 'waldo'), sum(train_labels == 'not_waldo'));

% Split into training and validation
rng(42);
num_images = size(train_images, 4);
idx = randperm(num_images);

train_ratio = 0.8;
num_train = round(train_ratio * num_images);

train_idx = idx(1:num_train);
val_idx = idx(num_train+1:end);

X_train = train_images(:, :, :, train_idx);
Y_train = train_labels(train_idx);
X_val = train_images(:, :, :, val_idx);
Y_val = train_labels(val_idx);

fprintf('Training set: %d images\n', num_train);
fprintf('Validation set: %d images\n', numel(val_idx));

% Create CNN architecture
fprintf('Creating CNN architecture...\n');

layers = [
    imageInputLayer([64 64 3]) % Fixed 64x64 RGB input

    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    fullyConnectedLayer(128)
    reluLayer
    dropoutLayer(0.5)

    fullyConnectedLayer(2) % waldo vs not_waldo
    softmaxLayer
    classificationLayer
];

% Train the network
fprintf('Training CNN...\n');

options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {X_val, Y_val}, ...
    'ValidationFrequency', 5, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

net_waldo = trainNetwork(X_train, Y_train, layers, options);

fprintf('Training completed!\n');

% CORRELATION MAP TESTING - NEW SECTION
fprintf('\n=== Generating Correlation Maps for Analysis ===\n');

% Create a simple correlation map function
function heatmap = generateCorrelationMap(net, test_img, window_size)
    [H, W, ~] = size(test_img);
    heatmap = zeros(H, W);
    
    step_size = 8; % Small step for detailed heatmap
    
    for i = 1:step_size:(H - window_size + 1)
        for j = 1:step_size:(W - window_size + 1)
            % Extract window
            window = test_img(i:i+window_size-1, j:j+window_size-1, :);
            
            % Classify
            [~, scores] = classify(net, window);
            waldo_prob = scores(2);
            
            % Add probability to heatmap region
            heatmap(i:i+window_size-1, j:j+window_size-1) = ...
                heatmap(i:i+window_size-1, j:j+window_size-1) + waldo_prob;
        end
    end
    
    % Normalize heatmap
    heatmap = heatmap / max(heatmap(:));
end

% Test on first scene image to see what the network responds to
scene_files = dir(fullfile(scene_path, '*.jpg'));
if ~isempty(scene_files)
    test_scene_path = fullfile(scene_path, scene_files(1).name);
    test_scene = imread(test_scene_path);
    
    fprintf('Generating correlation map for: %s\n', scene_files(1).name);
    
    % Generate correlation map
    correlation_heatmap = generateCorrelationMap(net_waldo, test_scene, 64);
    
    % Display results
    fig = figure('Position', [100, 100, 1200, 400]);
    
    subplot(1, 3, 1);
    imshow(test_scene);
    title('Original Scene Image');
    
    subplot(1, 3, 2);
    imshow(correlation_heatmap);
    title('Raw Correlation Heatmap');
    colorbar;
    
    subplot(1, 3, 3);
    % Overlay heatmap on original image
    heatmap_resized = imresize(correlation_heatmap, [size(test_scene, 1), size(test_scene, 2)]);
    heatmap_rgb = ind2rgb(round(heatmap_resized * 255), jet(256));
    overlay_img = imfuse(test_scene, heatmap_rgb, 'blend', 'Scaling', 'joint');
    imshow(overlay_img);
    title('Heatmap Overlay (Red = High Response)');
    
    % Find peaks in correlation map
    [peaks_y, peaks_x] = find(correlation_heatmap > 0.7);
    if ~isempty(peaks_y)
        hold on;
        plot(peaks_x * (size(test_scene, 2) / size(correlation_heatmap, 2)), ...
             peaks_y * (size(test_scene, 1) / size(correlation_heatmap, 1)), ...
             'ro', 'MarkerSize', 10, 'LineWidth', 2);
        hold off;
        legend('High Response Areas');
        fprintf('Found %d high-response areas in correlation map\n', length(peaks_x));
    end
    
    % Save correlation map
    saveas(fig, fullfile(filepath, 'correlation_analysis.png'));
    fprintf('Correlation map saved as correlation_analysis.png\n');
    
    pause(3);
    close(fig);
end

% IMPROVED DETECTION WITH CORRELATION-BASED REFINEMENT
fprintf('\n=== Starting Main Detection with Correlation Refinement ===\n');

% Use stricter thresholds to reduce false positives
collection_threshold = 0.85;   % Very high threshold for candidates
display_threshold = 0.92;      % Extremely confident only for final detection

% Waldo face size in original scenes (approximate)
waldo_face_size = 40;

for img_idx = 1:length(scene_files)
    fprintf('\n=== Processing scene image %d/%d ===\n', img_idx, length(scene_files));

    % Read test image
    img_path = fullfile(scene_path, scene_files(img_idx).name);
    test_img = imread(img_path);

    [H, W, ~] = size(test_img);
    fprintf('Original image size: %d x %d\n', H, W);

    % FIRST: Generate correlation map to understand network behavior
    fprintf('  Generating correlation map...\n');
    corr_map = generateCorrelationMap(net_waldo, test_img, 64);
    
    % Find correlation peaks to guide our search
    [peak_y, peak_x] = find(corr_map > 0.6);
    fprintf('  Found %d correlation peaks above 0.6\n', length(peak_x));
    
    % Store all detections
    all_detections = [];
    
    % Use fewer, more focused scales
    scales = [0.6, 0.8, 1.0];
    fixed_window_size = 64;
    
    for scale_idx = 1:length(scales)
        scale = scales(scale_idx);
        scaled_img = imresize(test_img, scale);
        [scaled_H, scaled_W, ~] = size(scaled_img);
        
        % Calculate step size based on expected face size at this scale
        step_size = max(8, round(waldo_face_size * scale * 0.5));
        
        fprintf('  Scale %.2f: scanning %dx%d image, step %d...\n', ...
            scale, scaled_H, scaled_W, step_size);
        
        % Ensure window fits in scaled image
        if fixed_window_size > scaled_H || fixed_window_size > scaled_W
            fprintf('    Skipping - window too large\n');
            continue;
        end
        
        detection_count = 0;
        for i = 1:step_size:(scaled_H - fixed_window_size + 1)
            for j = 1:step_size:(scaled_W - fixed_window_size + 1)
                % Extract FIXED 64x64 window from scaled image
                window = scaled_img(i:i+fixed_window_size-1, j:j+fixed_window_size-1, :);
                
                % Classify using our trained CNN
                [label, scores] = classify(net_waldo, window);
                waldo_prob = scores(2);
                
                % Convert coordinates back to original image space
                orig_i = round(i / scale);
                orig_j = round(j / scale);
                
                % Use estimated face size instead of window size
                estimated_face_size = round(waldo_face_size / scale);
                
                if waldo_prob > collection_threshold
                    all_detections = [all_detections; orig_i, orig_j, estimated_face_size, waldo_prob, scale];
                    detection_count = detection_count + 1;
                end
            end
        end
        fprintf('    Found %d candidate detections\n', detection_count);
    end

    fprintf('Total candidate detections: %d\n', size(all_detections, 1));
    
    % Apply non-maximum suppression
    result_img = test_img;
    
    if ~isempty(all_detections)
        % Sort by confidence
        [sorted_probs, sort_idx] = sort(all_detections(:, 4), 'descend');
        all_detections = all_detections(sort_idx, :);
        
        fprintf('Best detection confidence: %.3f\n', sorted_probs(1));
        
        % Non-maximum suppression - keep only the best detection
        final_detections = [];
        if ~isempty(all_detections)
            strong_detections = all_detections(all_detections(:, 4) >= display_threshold, :);
            
            if ~isempty(strong_detections)
                % Take only the single highest confidence detection
                best_detection = strong_detections(1, :);
                final_detections = best_detection;
                
                fprintf('✓ SELECTED: Waldo face at [%d,%d] size %d, confidence %.3f\n', ...
                    best_detection(1), best_detection(2), best_detection(3), best_detection(4));
            else
                % Show best guess in yellow (but only if reasonably confident)
                if sorted_probs(1) > 0.7
                    best_detection = all_detections(1, :);
                    final_detections = best_detection;
                    fprintf('~ WEAK: Best guess at [%d,%d] size %d, confidence %.3f\n', ...
                        best_detection(1), best_detection(2), best_detection(3), best_detection(4));
                else
                    fprintf('~ No confident detections (best was only %.3f)\n', sorted_probs(1));
                end
            end
        end
        
        % Draw SINGLE face-focused bounding box
        if ~isempty(final_detections)
            det = final_detections;
            
            % Calculate face box centered on detection
            face_center_i = det(1) + round(fixed_window_size / (2 * det(5)));
            face_center_j = det(2) + round(fixed_window_size / (2 * det(5)));
            face_half_size = round(det(3) / 2);
            
            face_box = [face_center_j - face_half_size, face_center_i - face_half_size, det(3), det(3)];
            
            if det(4) >= display_threshold
                % High confidence - red box around FACE
                result_img = insertShape(result_img, 'Rectangle', face_box, ...
                    'Color', 'red', 'LineWidth', 3);
                
                text_str = sprintf('Waldo: %.3f', det(4));
                result_img = insertText(result_img, [face_box(1), max(1, face_box(2)-25)], ...
                    text_str, 'FontSize', 14, 'BoxColor', 'red', 'TextColor', 'white');
                
                fprintf('✓ Drawing red box around Waldo''s FACE\n');
            else
                % Low confidence - yellow box
                result_img = insertShape(result_img, 'Rectangle', face_box, ...
                    'Color', 'yellow', 'LineWidth', 2);
                
                text_str = sprintf('Maybe: %.3f', det(4));
                result_img = insertText(result_img, [face_box(1), max(1, face_box(2)-25)], ...
                    text_str, 'FontSize', 12, 'BoxColor', 'yellow', 'TextColor', 'black');
                
                fprintf('~ Drawing yellow box for best guess\n');
            end
        end
    else
        fprintf('No candidate detections found\n');
    end

    % Save and display result
    result_filename = fullfile(filepath, sprintf('waldo_detection_%d.png', img_idx));
    imwrite(result_img, result_filename);
    
    fig = figure('Visible', 'on', 'Position', [100, 100, 800, 600]);
    imshow(result_img);
    if ~isempty(all_detections) && ~isempty(final_detections)
        if final_detections(4) >= display_threshold
            title(sprintf('Waldo Face Found! (confidence: %.3f) - Image %d/%d', final_detections(4), img_idx, length(scene_files)), 'FontSize', 14);
        else
            title(sprintf('Possible Waldo Face (confidence: %.3f) - Image %d/%d', final_detections(4), img_idx, length(scene_files)), 'FontSize', 14);
        end
    else
        title(sprintf('No Waldo Detection - Image %d/%d', img_idx, length(scene_files)), 'FontSize', 14);
    end
    drawnow;
    pause(2);
    close(fig);
    
    fprintf('Saved result image\n');
end

fprintf('\n=== CORRELATION ANALYSIS COMPLETED! ===\n');
fprintf('Check correlation_analysis.png to see what the network responds to\n');
fprintf('Red areas show where the network detects "Waldo-like" patterns\n');

% Save the trained network
save(fullfile(filepath, 'waldo_face_net.mat'), 'net_waldo');
fprintf('Network saved as waldo_face_net.mat\n');

% Helper function
function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end

fprintf('\n=== WALDO DETECTION WITH CORRELATION ANALYSIS COMPLETE ===\n');