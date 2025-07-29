% This MATLAB code generates the phase h5 file for a geometric phase metalens.
% https://optics.ansys.com/hc/en-us/articles/18254409091987-Large-Scale-Metalens-Ray-Propagation

%% Load and process parameters from the .mat file
optimized_lens_MatPath = 'optimized_lens.mat';

load(optimized_lens_MatPath); % Load optimized metalens and refractive lens parameters

% Convert parameters from millimeters (mm) to micrometers (um).
um = 1000;
period = meta_period * um; % Unit cell period in um
radius = meta_radius * um; % Metalens radius in um
% Calculate ai values
ai_bound = double(meta_ai_bound);
ai = meta_ai * ai_bound;

resolution = int32(radius / period); % Radius resolution
resolution = double(resolution); % Convert to double for further calculations

% Calculate x and y coordinates based on resolution and period
coordRange = ((-resolution + 0.5):1:(resolution - 0.5)) * period;
[x, y] = meshgrid(coordRange, coordRange);
r = sqrt(x.^2 + y.^2); % Calculate radius for each point

r_normalized = r / radius; % Normalize r

%% Calculate phase
phase = zeros(size(r));
exponents = 0:2:(2 * numel(ai) - 2); % Note: MATLAB indices start at 1
% Iterate over each exponent
for idx = 1:length(exponents)
    % Calculate the current exponent term
    current_term = r_normalized .^ exponents(idx);
    
    % Multiply by the corresponding ai coefficient and add to phase
    phase = phase + ai(idx) * current_term;
end

% Set phase to 0 for points outside the radius
phase(r > radius) = 0;

% Visualize the result
imagesc(phase);
axis equal tight;
colormap('jet');
colorbar;
title('2D Phase Distribution');

%% Write to h5 file
% Convert parameters to millimeters
radius_mm = radius / 1000; % Convert to mm

dx = period;
dy = period;

amplitude = ones(size(phase)) - 1e-8;

% Filename
filename = sprintf('GeometicMeta_r_%d.h5', int32(radius_mm));
saveToH5(filename, dx, dy, phase, amplitude); % Pass phase and amplitude as 2D arrays