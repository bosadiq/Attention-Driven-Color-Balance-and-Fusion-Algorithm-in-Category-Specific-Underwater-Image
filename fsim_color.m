function fsim_score = fsim_color(I1, I2)
% fsim_color - Computes FSIM for RGB images from scratch
%
% Usage:
%   score = fsim_color(I1, I2)
%
% Inputs:
%   I1 - Reference color image (RGB)
%   I2 - Enhanced/distorted color image (RGB)
%
% Output:
%   fsim_score - FSIM similarity score between 0 and 1

% Ensure double and normalized
if ~isa(I1, 'double'), I1 = im2double(I1); end
if ~isa(I2, 'double'), I2 = im2double(I2); end

assert(all(size(I1) == size(I2)), 'Images must be same size and RGB');

% Process each channel
fsim_R = fsim_channel(I1(:,:,1), I2(:,:,1));
fsim_G = fsim_channel(I1(:,:,2), I2(:,:,2));
fsim_B = fsim_channel(I1(:,:,3), I2(:,:,3));

% Final FSIM score (average)
fsim_score = mean([fsim_R, fsim_G, fsim_B]);

end

% -----------------------------------------------
% FSIM for one grayscale channel
% -----------------------------------------------
function score = fsim_channel(I1, I2)
T1 = 0.85;
T2 = 160;

% Phase Congruency (approximate using Gabor)
PC1 = phase_congruency_approx(I1);
PC2 = phase_congruency_approx(I2);

% Gradient Magnitude
GM1 = gradient_magnitude(I1);
GM2 = gradient_magnitude(I2);

% Similarity measures
S_pc = (2 .* PC1 .* PC2 + T1) ./ (PC1.^2 + PC2.^2 + T1);
S_gm = (2 .* GM1 .* GM2 + T2) ./ (GM1.^2 + GM2.^2 + T2);
Sim = S_pc .* S_gm;

% Weighting with max PC
PC_max = max(PC1, PC2);
score = sum(Sim(:) .* PC_max(:)) / sum(PC_max(:));
end

% -----------------------------------------------
% Compute Gradient Magnitude
% -----------------------------------------------
function GM = gradient_magnitude(I)
[Gx, Gy] = gradient(I);
GM = sqrt(Gx.^2 + Gy.^2);
end

% -----------------------------------------------
% Approximate Phase Congruency (Gabor-based)
% -----------------------------------------------
function PC = phase_congruency_approx(I)
% Create a Gabor filter bank approximation (4 orientations)
orientations = 4;
PC = zeros(size(I));
for i = 1:orientations
    theta = (i - 1) * pi / orientations;
    h = gabor_filter(size(I), theta);
    response = abs(imfilter(I, h, 'replicate'));
    PC = max(PC, response);
end
% Normalize
PC = PC / max(PC(:) + eps);
end

% -----------------------------------------------
% Gabor Filter Generator (frequency = 0.2, sigma = 2)
% -----------------------------------------------
function h = gabor_filter(sz, theta)
lambda = 5;
sigma = 2;
gamma = 0.5;
psi = 0;

[x, y] = meshgrid(-floor(sz(2)/2):floor(sz(2)/2)-1, ...
                  -floor(sz(1)/2):floor(sz(1)/2)-1);

x_theta = x * cos(theta) + y * sin(theta);
y_theta = -x * sin(theta) + y * cos(theta);

h = exp(-0.5 * (x_theta.^2 + gamma^2 * y_theta.^2) / sigma^2) ...
    .* cos(2 * pi * x_theta / lambda + psi);
end
