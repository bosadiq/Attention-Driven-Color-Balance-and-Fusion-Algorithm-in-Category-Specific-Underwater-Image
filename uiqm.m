function uiqm_score = uiqm(img)
% COMPUTE_UIQM Computes the Underwater Image Quality Measure (UIQM)
% based on colorfulness (UICM), sharpness (UISM), and contrast (UIConM).
% 
% Input:
%   img - RGB underwater image (uint8 or double)
% Output:
%   uiqm_score - scalar UIQM score

% Check input
if size(img, 3) ~= 3
    error('Input image must be RGB.');
end

% Convert to double precision if needed
if ~isfloat(img)
    img = im2double(img);
end

% Compute subcomponents
uicm = compute_uicm(img);
uism = compute_uism(img);
uiconm = compute_uiconm(img);

% Standard weights (Panetta et al. 2015)
c1 = 0.0282;
c2 = 0.2953;
c3 = 3.5753;

% Final UIQM score
uiqm_score = c1 * uicm + c2 * uism + c3 * uiconm;

end

% -----------------------------
function uicm = compute_uicm(img)
% UICM: Underwater Image Colorfulness Measure

R = img(:,:,1);
G = img(:,:,2);
B = img(:,:,3);

RG = R - G;
YB = 0.5 * (R + G) - B;

meanRG = mean(RG(:));
meanYB = mean(YB(:));
stdRG = std(RG(:));
stdYB = std(YB(:));

uicm = -0.0268 * sqrt(meanRG^2 + meanYB^2) + 0.1586 * sqrt(stdRG^2 + stdYB^2);
end

% -----------------------------
function uism = compute_uism(img)
% UISM: Underwater Image Sharpness Measure

gray = rgb2gray(img);

% Sobel edge detection
sobel_h = fspecial('sobel')';
sobel_v = fspecial('sobel');

grad_h = imfilter(gray, sobel_h, 'replicate');
grad_v = imfilter(gray, sobel_v, 'replicate');

grad_mag = sqrt(grad_h.^2 + grad_v.^2);
uism = mean(grad_mag(:));
end

% -----------------------------
function uiconm = compute_uiconm(img)
% UIConM: Underwater Image Contrast Measure

gray = rgb2gray(img);

% RMS contrast
mean_val = mean(gray(:));
rms_contrast = sqrt(mean((gray(:) - mean_val).^2));
uiconm = rms_contrast;
end
