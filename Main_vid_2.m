% Underwater Video Enhancement Using Attention-Guided Autoencoder Framework with 5-Frame Sampling
% Includes refined color and contrast enhancement with 5-frame sampling for category detection
% Designed for four categories: Bluish Hazy, Greenish Hazy, Low-Light Bluish, Low-Light Greenish


close all;
clear;
clc;

%% Load the video
videoFile = 'ArmyDiver1.mp4'; % Replace with your video file path
outputFile = 'enhanced_video_visible.mp4';
vr = VideoReader(videoFile);
vw = VideoWriter(outputFile, 'MPEG-4');
vw.FrameRate = vr.FrameRate;
open(vw);

%% Step 1: Sample 5 evenly spaced frames
numSamples = 5;
totalFrames = floor(vr.Duration * vr.FrameRate);
sampleIdx = round(linspace(1, totalFrames, numSamples));
sampleFrames = zeros(vr.Height, vr.Width, 3, numSamples);

for i = 1:numSamples
    frameNumber = sampleIdx(i);
    vr.CurrentTime = (frameNumber - 1) / vr.FrameRate;
    frame = readFrame(vr);
    sampleFrames(:, :, :, i) = frame;
end

%% Step 2: Process sampled frames and extract common category
categories = zeros(1, numSamples);
for i = 1:numSamples
    rgbImage = im2double(sampleFrames(:, :, :, i));
    avg_rgb = mean(mean(rgbImage, 1), 2);
    brightness = mean(rgb2gray(rgbImage), 'all');
    
    % Category detection
    if avg_rgb(3) > avg_rgb(2) && avg_rgb(3) > avg_rgb(1) % Blue dominates
        if brightness < 0.3
            categories(i) = 1; % Low-Light Bluish
        else
            categories(i) = 2; % Bluish Hazy
        end
    elseif avg_rgb(2) > avg_rgb(3) && avg_rgb(2) > avg_rgb(1) % Green dominates
        if brightness < 0.3
            categories(i) = 3; % Low-Light Greenish
        else
            categories(i) = 4; % Greenish Hazy
        end
    else
        categories(i) = 4; % Default fallback
    end
end

finalCategory = mode(categories);
switch finalCategory
    case 1, categoryName = 'Low-Light Bluish';
    case 2, categoryName = 'Bluish Hazy';
    case 3, categoryName = 'Low-Light Greenish';
    case 4, categoryName = 'Greenish Hazy';
end
fprintf('Final category (mode of samples): %s\n', categoryName);

%% Step 3: Process the entire video using the inferred category
vr.CurrentTime = 0;
frameIdx = 0;
while hasFrame(vr)
    rgbImage = im2double(readFrame(vr));
    frameIdx = frameIdx + 1;

    % Dehazing with Dark Channel Prior
    dark_channel = min(rgbImage, [], 3);
    dark_channel = imfilter(dark_channel, ones(15,15)/225, 'replicate');
    transmission = 1 - 0.7 * dark_channel; % Adjusted for detail preservation
    transmission = max(0.1, transmission); % Minimum transmission
    I_dehazed = zeros(size(rgbImage));
    for c = 1:3
        I_dehazed(:,:,c) = (rgbImage(:,:,c) - 0.1) ./ transmission;
    end
    I_dehazed = max(0, min(1, I_dehazed));
    fprintf('Dehazed range: [%.4f, %.4f]\n', min(I_dehazed(:)), max(I_dehazed(:)));

    %% Split channels for white balance
    Ir = I_dehazed(:,:,1);
    Ig = I_dehazed(:,:,2);
    Ib = I_dehazed(:,:,3);

    Ir_mean = mean(Ir, 'all');
    Ig_mean = mean(Ig, 'all');
    Ib_mean = mean(Ib, 'all');

    %% Color compensation with category tuning
    alpha = 0.15; % Stronger correction
    switch categoryName
        case 'Bluish Hazy'
            Irc = Ir + alpha * (Ig_mean - Ir_mean) * 1.2; % Boost red
            Ibc = Ib * 0.7; % Reduce blue dominance
        case 'Greenish Hazy'
            Irc = Ir + alpha * (Ig_mean - Ir_mean) * 1.3; % Boost red
            Ibc = Ib; % No blue adjustment
        case 'Low-Light Bluish'
            Irc = Ir + alpha * (Ig_mean - Ir_mean) * 1.5; % Strong red boost
            Ibc = Ib * 0.6; % Stronger blue reduction
        case 'Low-Light Greenish'
            Irc = Ir + alpha * (Ig_mean - Ir_mean) * 1.5; % Strong red boost
            Ibc = Ib; % No blue adjustment
    end

    %% White Balance
    I = cat(3, Irc, Ig, Ibc);
    I_lin = rgb2lin(I);
    percentiles = 5;
    illuminant = illumgray(I_lin, percentiles);
    I_lin = chromadapt(I_lin, illuminant, 'ColorSpace', 'linear-rgb');
    Iwb = lin2rgb(I_lin);

    %% Gamma Correction
    gamma_val = 1.3; % Adjusted for better brightness
    Igamma = imadjust(Iwb, [], [], gamma_val);

    %% Image Sharpening with Category Tuning
    sigma = 2; % Reduced sigma for minimal darkening
    Igauss = imgaussfilt(Igamma, sigma);
    gain = 0.7; % Adjusted gain
    switch categoryName
        case {'Bluish Hazy', 'Greenish Hazy'}
            gain = 0.7;
        case {'Low-Light Bluish', 'Low-Light Greenish'}
            gain = 0.9; % Aggressive for low-light
    end
    Norm = (Igamma - gain * Igauss);
    for n = 1:3
        Norm(:,:,n) = histeq(Norm(:,:,n));
    end
    Isharp = (Igamma + Norm) / 2;
    Isharp = max(0, min(1, Isharp));

    %% Attention Mechanism (Simplified for Video)
    WS1 = saliency_detection(Isharp);
    WS1 = WS1 / max(WS1, [], 'all'); % Normalize
    Isharp_lab = rgb2lab(Isharp);
    R1 = double(Isharp_lab(:, :, 1)) / 255;
    WC1 = sqrt((((Isharp(:,:,1)) - R1).^2 + ((Isharp(:,:,2)) - R1).^2 + ((Isharp(:,:,3)) - R1).^2) / 3);
    WSAT1 = sqrt(1/3 * ((Isharp_lab(:,:,2)).^2 + (Isharp_lab(:,:,3)).^2));
    WSAT1 = WSAT1 / max(WSAT1, [], 'all');
    attention1 = WS1 .* WC1 .* WSAT1;
    attention1 = attention1 / max(attention1, [], 'all');

    %% Multi-Scale Fusion (Simplified)
    fusion = Isharp .* (1 + attention1); % Basic fusion with attention
    fusion = max(0, min(1, fusion));

    %% Refined Color Enhancement
    fusion_hsv = rgb2hsv(fusion);
    saturation_boost = 1.05; % Reduced to 5%
    sat_mask = fusion_hsv(:,:,2) < 0.7;
    fusion_hsv(:,:,2) = fusion_hsv(:,:,2) .* (1 + saturation_boost * sat_mask);
    fusion_hsv(:,:,2) = min(fusion_hsv(:,:,2), 1);
    fusion = hsv2rgb(fusion_hsv);

    mean_rgb = mean(mean(fusion, 1), 2);
    color_adjust = 0.01;
    fusion(:,:,1) = fusion(:,:,1) + color_adjust * (mean_rgb(2) - mean_rgb(1));
    fusion(:,:,2) = fusion(:,:,2) + color_adjust * (mean_rgb(3) - mean_rgb(2));
    fusion(:,:,3) = fusion(:,:,3) + color_adjust * (mean_rgb(1) - mean_rgb(3));
    fusion = max(0, min(1, fusion));

    %% Improved Contrast Enhancement
    fusion = imgaussfilt(fusion, 0.7); % Noise reduction
    fusion_hsv = rgb2hsv(fusion);
    h = fusion_hsv(:,:,1); S = fusion_hsv(:,:,2); V = fusion_hsv(:,:,3);

    H = [0 -0.25 0; -0.25 1 -0.25; 0 -0.25 0]; % Soft Laplacian
    filter1 = imfilter(V, H, 'replicate');
    filter2 = imfilter(S, H, 'replicate');

    window_size = 3;
    varience_value = zeros(size(V));
    varience_saturation = zeros(size(S));
    V_diff = V - filter1;
    S_diff = S - filter2;
    for i = 1:size(V, 1) - window_size + 1
        for j = 1:size(V, 2) - window_size + 1
            V_window = V_diff(i:i+window_size-1, j:j+window_size-1);
            S_window = S_diff(i:i+window_size-1, j:j+window_size-1);
            varience_value(i,j) = mean(V_window(:).^2);
            varience_saturation(i,j) = mean(S_window(:).^2);
        end
    end
    varience_value(end-window_size+2:end, :) = repmat(varience_value(end-window_size+1, :), window_size-1, 1);
    varience_value(:, end-window_size+2:end) = repmat(varience_value(:, end-window_size+1), 1, window_size-1);
    varience_saturation(end-window_size+2:end, :) = repmat(varience_saturation(end-window_size+1, :), window_size-1, 1);
    varience_saturation(:, end-window_size+2:end) = repmat(varience_saturation(:, end-window_size+1), 1, window_size-1);

    try1 = V - varience_value;
    try2 = S - varience_saturation;
    epsilon = 1e-10;
    Correlation = (try1 .* try2) ./ sqrt((varience_value + epsilon) .* (varience_saturation + epsilon));

    mean_V = mean(V(:));
    std_V = std(V(:));
    k1 = 0.2 + (1 - mean_V) * 0.1;
    k2 = 0.1 + (std_V / 4);
    fprintf('Adaptive k1: %.2f, k2: %.2f\n', k1, k2);

    V_enhancement = V + k1 .* try1 - (k2 .* try2) .* Correlation;
    S_enhancement = S + (k1 / 4) .* try2;
    V_enhancement = max(0, min(1, V_enhancement));
    S_enhancement = max(0, min(1, S_enhancement));

    num_tiles = [3 3];
    clip_limit = 0.003 + (1 - mean_V) * 0.005;
    V_clahe = adapthisteq(V_enhancement, 'NumTiles', num_tiles, 'ClipLimit', clip_limit);

    mean_V_clahe = mean(V_clahe(:));
    gamma_contrast = 0.95 + (1 - mean_V_clahe) * 0.05;
    contrast_stretching = V_clahe .^ gamma_contrast;
    fprintf('Adaptive gamma for contrast: %.2f\n', gamma_contrast);

    final_output = cat(3, h, S_enhancement, contrast_stretching);
    final_output_rgb = hsv2rgb(final_output);

    %% Write enhanced frame to video
    writeVideo(vw, im2uint8(final_output_rgb));

    if mod(frameIdx, 50) == 0
        fprintf('Processed frame %d/%d\n', frameIdx, totalFrames);
    end
end

%% Close video writer
close(vw);
fprintf('Video enhancement completed. Output saved to %s\n', outputFile);