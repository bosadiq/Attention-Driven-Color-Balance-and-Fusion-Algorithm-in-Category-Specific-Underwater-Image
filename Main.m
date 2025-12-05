% Underwater Image Enhancement Using Attention-Guided Autoencoder Framework
% Includes refined color and contrast enhancement
% Designed for four categories: Bluish Hazy, Greenish Hazy, Low-Light Bluish, Low-Light Greenish
%
close all;
clear;
clc;
tic;
%% Load the image and split channels
rgbImage = double(imread('31x.png')) / 255;
figure(1); imshow(rgbImage); title('Original Underwater Image');

% Validate input
if size(rgbImage, 3) ~= 3
    error('Image must be RGB.');
end

%% Step 1: Category Detection and Dehazing
% Detect dominant color and brightness for category
avg_rgb = mean(mean(rgbImage, 1), 2);
brightness = mean(rgb2gray(rgbImage), 'all');
if avg_rgb(3) > avg_rgb(2) && avg_rgb(3) > avg_rgb(1) % Blue dominates
    if brightness < 0.3
        category = 'Low-Light Bluish';
    else
        category = 'Bluish Hazy';
    end
elseif avg_rgb(2) > avg_rgb(3) && avg_rgb(2) > avg_rgb(1) % Green dominates
    if brightness < 0.3
        category = 'Low-Light Greenish';
    else
        category = 'Greenish Hazy';
    end
else
    category = 'Greenish Hazy'; % Default
end
disp(['Detected Category: ', category]);

% Dehazing with Dark Channel Prior
dark_channel = min(rgbImage, [], 3);
dark_channel = imfilter(dark_channel, ones(15,15)/225, 'replicate');
transmission = 1 - 0.7 * dark_channel; % Adjusted for better detail preservation
transmission = max(0.1, transmission); % Minimum transmission
I_dehazed = zeros(size(rgbImage));
for c = 1:3
    I_dehazed(:,:,c) = (rgbImage(:,:,c) - 0.1) ./ transmission;
end
I_dehazed = max(0, min(1, I_dehazed));
fprintf('Dehazed range: [%.4f, %.4f]\n', min(I_dehazed(:)), max(I_dehazed(:)));
figure(2); imshow(I_dehazed); title('Dehazed Image');

%% Split channels for white balance
Ir = I_dehazed(:,:,1);
Ig = I_dehazed(:,:,2);
Ib = I_dehazed(:,:,3);

Ir_mean = mean(Ir, 'all');
Ig_mean = mean(Ig, 'all');
Ib_mean = mean(Ib, 'all');

%% Color compensation with category tuning
alpha = 0.1;
switch category
    case 'Bluish Hazy'
        Irc = Ir + alpha * (Ig_mean - Ir_mean); % Boost red indirectly
        Ibc = Ib + 0 * (Ig_mean - Ib_mean);     % No blue compensation
    case 'Greenish Hazy'
        Irc = Ir + alpha * (Ig_mean - Ir_mean) * 1.2; % Boost red
        Ibc = Ib + 0 * (Ig_mean - Ib_mean);     % No blue compensation
    case 'Low-Light Bluish'
        Irc = Ir + alpha * (Ig_mean - Ir_mean) * 1.5; % Boost illumination
        Ibc = Ib + 0 * (Ig_mean - Ib_mean);     % No blue compensation
    case 'Low-Light Greenish'
        Irc = Ir + alpha * (Ig_mean - Ir_mean) * 1.5; % Boost illumination
        Ibc = Ib + 0 * (Ig_mean - Ib_mean);     % No blue compensation
end

%% White Balance
I = cat(3, Irc, Ig, Ibc);
I_lin = rgb2lin(I);
percentiles = 5;
illuminant = illumgray(I_lin, percentiles);
I_lin = chromadapt(I_lin, illuminant, 'ColorSpace', 'linear-rgb');
Iwb = lin2rgb(I_lin);
figure(3); imshow(Iwb); title('White Balance Output');

%%% Multi-Scale Fusion with Attention %%%

%% Gamma Correction
gamma = 1.5; % Adjusted for better SSIM alignment
Igamma = imadjust(Iwb, [], [], gamma);
figure(4); imshow(Igamma); title('Gamma Correction');

%% Image Sharpening with Category Tuning
sigma = 20;
Igauss = Iwb;
N = 30;
for iter = 1:N
    Igauss = imgaussfilt(Igauss, sigma);
    Igauss = min(Iwb, Igauss);
end
gain = 1;
switch category
    case {'Bluish Hazy', 'Greenish Hazy'}
        gain = 0.8; % Reduced gain for hazy images
    case {'Low-Light Bluish', 'Low-Light Greenish'}
        gain = 1.0; % Adjusted gain for low-light images
end
Norm = (Iwb - gain * Igauss);
for n = 1:3
    Norm(:,:,n) = histeq(Norm(:,:,n));
end
Isharp = (Iwb + Norm) / 2;
% figure(5); imshow(Isharp); title('Sharpened Image');

%% Attention Mechanism
% Compute saliency maps for both inputs
WS1 = saliency_detection(Isharp);
WS1 = WS1 / max(WS1, [], 'all'); % Normalize
WS2 = saliency_detection(Igamma);
WS2 = WS2 / max(WS2, [], 'all'); % Normalize

% Convert to Lab for contrast and saturation
Isharp_lab = rgb2lab(Isharp);
Igamma_lab = rgb2lab(Igamma);

% Input 1 (Sharpened): Contrast and Saturation
R1 = double(Isharp_lab(:, :, 1)) / 255;
WC1 = sqrt((((Isharp(:,:,1)) - R1).^2 + ((Isharp(:,:,2)) - R1).^2 + ((Isharp(:,:,3)) - R1).^2) / 3);
WSAT1 = sqrt(1/3 * ((Isharp_lab(:,:,2)).^2 + (Isharp_lab(:,:,3)).^2)); % Saturation from a, b channels
WSAT1 = WSAT1 / max(WSAT1, [], 'all'); % Normalize

% Input 2 (Gamma-corrected): Contrast and Saturation
R2 = double(Igamma_lab(:, :, 1)) / 255;
WC2 = sqrt((((Igamma(:,:,1)) - R2).^2 + ((Igamma(:,:,2)) - R2).^2 + ((Igamma(:,:,3)) - R2).^2) / 3);
WSAT2 = sqrt(1/3 * ((Igamma_lab(:,:,2)).^2 + (Igamma_lab(:,:,3)).^2)); % Saturation from a, b channels
WSAT2 = WSAT2 / max(WSAT2, [], 'all'); % Normalize

% Attention Map: Combine saliency, contrast, and saturation
attention1 = WS1 .* WC1 .* WSAT1; % Attention for sharpened input
attention2 = WS2 .* WC2 .* WSAT2; % Attention for gamma-corrected input
attention1 = attention1 / max(attention1, [], 'all'); % Normalize
attention2 = attention2 / max(attention2, [], 'all'); % Normalize
figure(6); imshow([attention1, attention2]); title('Attention Maps: Sharpened | Gamma-Corrected');

% Normalized weights with attention
W1 = (WC1 + WS1 + WSAT1 + attention1 + 0.1) ./ (WC1 + WS1 + WSAT1 + attention1 + WC2 + WS2 + WSAT2 + attention2 + 0.2);
W2 = (WC2 + WS2 + WSAT2 + attention2 + 0.1) ./ (WC1 + WS1 + WSAT1 + attention1 + WC2 + WS2 + WSAT2 + attention2 + 0.2);

%% Multi-Scale Fusion
img1 = Isharp;
img2 = Igamma;

level = 10;
Weight1 = gaussian_pyramid(W1, level);
Weight2 = gaussian_pyramid(W2, level);

% Laplacian pyramid for input 1
R1 = laplacian_pyramid(img1(:,:,1), level);
G1 = laplacian_pyramid(img1(:,:,2), level);
B1 = laplacian_pyramid(img1(:,:,3), level);

% Laplacian pyramid for input 2
R2 = laplacian_pyramid(img2(:,:,1), level);
G2 = laplacian_pyramid(img2(:,:,2), level);
B2 = laplacian_pyramid(img2(:,:,3), level);

% Fusion with attention-guided weights
for k = 1:level
    Rr{k} = Weight1{k} .* R1{k} + Weight2{k} .* R2{k};
    Rg{k} = Weight1{k} .* G1{k} + Weight2{k} .* G2{k};
    Rb{k} = Weight1{k} .* B1{k} + Weight2{k} .* B2{k};
end

% Reconstruct
R = pyramid_reconstruct(Rr);
G = pyramid_reconstruct(Rg);
B = pyramid_reconstruct(Rb);
fusion = cat(3, R, G, B);
fusion = max(0, min(1, fusion)); % Clip to valid range

% Post-processing for reference alignment
fusion = imadjust(fusion, [min(fusion(:)); max(fusion(:))], [0; 1]); % Contrast adjustment

%% Refined Color Enhancement
% Step 1: Selective saturation increase in HSV color space
fusion_hsv = rgb2hsv(fusion);
saturation_boost = 1.05; % Reduced to 5% to minimize artifacts
% Apply saturation boost only where it enhances without overdoing
sat_mask = fusion_hsv(:,:,2) < 0.7; % Limit to moderately saturated areas
fusion_hsv(:,:,2) = fusion_hsv(:,:,2) .* (1 + saturation_boost * sat_mask); % Selective boost
fusion_hsv(:,:,2) = min(fusion_hsv(:,:,2), 1); % Avoid oversaturation
fusion = hsv2rgb(fusion_hsv);

% Step 2: Fine-tune color balance with minimal adjustment
mean_rgb = mean(mean(fusion, 1), 2); % Compute mean RGB values
color_adjust = 0.01; % Further reduced to minimize distortion
fusion(:,:,1) = fusion(:,:,1) + color_adjust * (mean_rgb(2) - mean_rgb(1)); % Adjust red
fusion(:,:,2) = fusion(:,:,2) + color_adjust * (mean_rgb(3) - mean_rgb(2)); % Adjust green
fusion(:,:,3) = fusion(:,:,3) + color_adjust * (mean_rgb(1) - mean_rgb(3)); % Adjust blue
fusion = max(0, min(1, fusion)); % Clip to valid range
% figure(7); imshow(fusion); title('Refined Color Enhanced Image');

%% Improved Contrast Enhancement with Enhanced Noise Management
% Apply stronger Gaussian blur to reduce noise
fusion = imgaussfilt(fusion, 0.7); % Increased sigma to 0.7 for better noise reduction

% Convert to HSV for contrast enhancement
fusion_hsv = rgb2hsv(fusion);
h = fusion_hsv(:,:,1); % Hue
S = fusion_hsv(:,:,2); % Saturation
V = fusion_hsv(:,:,3); % Value

% Apply a very soft Laplacian filter to minimize edge artifacts
H = [0 -0.25 0; -0.25 1 -0.25; 0 -0.25 0]; % Even softer Laplacian kernel
filter1 = imfilter(V, H, 'replicate'); % Apply to V
filter2 = imfilter(S, H, 'replicate'); % Apply to S
combined_VS = cat(2, filter1, filter2);
% figure(8); imshow(combined_VS); title('Laplacian Filter Output (V | S)');

% Compute local variance for V and S over a 3x3 window
window_size = 3;
varience_value = zeros(size(V));
varience_saturation = zeros(size(S));
V_diff = V - filter1;
S_diff = S - filter2;
for i = 1:size(V, 1) - window_size + 1
    for j = 1:size(V, 2) - window_size + 1
        V_window = V_diff(i:i+window_size-1, j:j+window_size-1);
        S_window = S_diff(i:i+window_size-1, j:j+window_size-1);
        varience_value(i,j) = mean(V_window(:).^2); % Local variance for V
        varience_saturation(i,j) = mean(S_window(:).^2); % Local variance for S
    end
end
% Pad the remaining pixels
varience_value(end-window_size+2:end, :) = repmat(varience_value(end-window_size+1, :), window_size-1, 1);
varience_value(:, end-window_size+2:end) = repmat(varience_value(:, end-window_size+1), 1, window_size-1);
varience_saturation(end-window_size+2:end, :) = repmat(varience_saturation(end-window_size+1, :), window_size-1, 1);
varience_saturation(:, end-window_size+2:end) = repmat(varience_saturation(:, end-window_size+1), 1, window_size-1);

% figure(9); imshow(varience_value); title('Local Variance of V Component');
% figure(10); imshow(varience_saturation); title('Local Variance of S Component');

% Compute adjusted V and S components
try1 = V - varience_value; % Adjusted V
try2 = S - varience_saturation; % Adjusted S
% figure(11); imshow(try1); title('Adjusted V Component');
% figure(12); imshow(try2); title('Adjusted S Component');

% Compute correlation coefficient
epsilon = 1e-10; % Avoid division by zero
Correlation = (try1 .* try2) ./ sqrt((varience_value + epsilon) .* (varience_saturation + epsilon));
% figure(13); imshow(Correlation); title('Correlation Coefficient');

% Compute enhancement value with minimal intensity
mean_V = mean(V(:));
std_V = std(V(:));
k1 = 0.2 + (1 - mean_V) * 0.1; % Further reduced enhancement
k2 = 0.1 + (std_V / 4); % Further reduced adjustment
fprintf('Adaptive k1: %.2f, k2: %.2f\n', k1, k2);

V_enhancement = V + k1 .* try1 - (k2 .* try2) .* Correlation;
S_enhancement = S + (k1 / 4) .* try2; % Very minimal saturation enhancement
V_enhancement = max(0, min(1, V_enhancement)); % Clip to valid range
S_enhancement = max(0, min(1, S_enhancement));
% figure(14); imshow(V_enhancement); title('Enhanced V Component');

% Apply Adaptive Histogram Equalization (CLAHE) with lower clip limit
num_tiles = [5 5];
clip_limit = 0.003 + (1 - mean_V) * 0.005; % Further reduced clip limit
V_clahe = adapthisteq(V_enhancement, 'NumTiles', num_tiles, 'ClipLimit', clip_limit);
% figure(15); imshow(V_clahe); title('CLAHE Enhanced V Component');

% Perform adaptive gamma correction with conservative stretching
mean_V_clahe = mean(V_clahe(:));
gamma_contrast = 0.95 + (1 - mean_V_clahe) * 0.05; % Even more conservative gamma
contrast_stretching = V_clahe .^ gamma_contrast;
fprintf('Adaptive gamma for contrast: %.2f\n', gamma_contrast);
% figure(16); imshow(contrast_stretching); title('After Gamma Correction');

% Convert back to RGB
final_output = cat(3, h, S_enhancement, contrast_stretching);
final_output_rgb = hsv2rgb(final_output);

%% Final Output
figure(17); imshow([rgbImage, final_output_rgb]);
title('Original vs. Enhanced (With Refined Enhancements)');
imwrite(final_output_rgb, 'enhanced_final_refined.jpg');
toc;
%% Evaluation Metrics
UIQM_value = uiqm(final_output_rgb);
fprintf('UIQM: %.4f\n', ...
    UIQM_value);

% PSNR (requires reference)
PSNR_value = psnrxx('31x.png', 'enhanced_final_refined.jpg');
fprintf('PSNR: %.4f\n', PSNR_value);

% I1 = imread('image1.png');
% I2 = imread('enhanced_final_refined.jpg');
% FSIM_value = fsim_color(I2, I1);
% fprintf('FSIM: %.4f\n', FSIM_value);