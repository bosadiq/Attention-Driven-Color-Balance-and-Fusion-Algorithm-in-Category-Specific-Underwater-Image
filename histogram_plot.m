% Read your already enhanced image
I_enhanced = imread('enhanced_final_refined.jpg');  % Replace with your actual filename

% Set up the figure
figure('Position', [100 100 1200 400]);

% % --- (e) Display Enhanced Image ---
% subplot(1,3,1);
% imshow(I_enhanced);
% title('(e) Enhanced Image');

% --- (g) Plot RGB Channel Histograms ---
subplot(1,3,2);
hold on;
channel_names = {'Red', 'Green', 'Blue'};
colors = {'r', 'g', 'b'};
for i = 1:3
    channel = I_enhanced(:,:,i);
    histogram(double(channel(:))/255, 50, ...
        'FaceColor', colors{i}, 'EdgeColor', 'none', 'FaceAlpha', 0.6);
end
title('Histogram');
xlabel('Normalized Intensity');
ylabel('Pixel Count');
% legend(channel_names);
grid on;
sgtitle('Reference Image');

% % --- Chromaticity Map: Hue vs Saturation ---
% subplot(1,3,3);
% I_hsv = rgb2hsv(I_enhanced);
% H = I_hsv(:,:,1) * 360;  % Hue in degrees
% S = I_hsv(:,:,2);        % Saturation
% 
% % Convert to polar coordinates for circular plot
% theta = deg2rad(H(:));
% r = S(:);
% x = r .* cos(theta);
% y = r .* sin(theta);
% 
% % Scatter plot in chromaticity space
% scatter(x, y, 1, H(:), 'filled');
% axis equal;
% axis off;
% colormap(hsv);
% colorbar('Ticks',[0 60 120 180 240 300 360], 'TickLabels', ...
%     {'0°','60°','120°','180°','240°','300°','360°'});
% title('Chromaticity Map');


