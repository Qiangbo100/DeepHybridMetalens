function saveToH5(filename, dx, dy, phase, amplitude)
    % Calculate resolution, assuming phase is a square matrix
    resolution = size(phase, 1) / 2;
    
    % Flatten phase and amplitude into 1D arrays
    phase = reshape(phase, 1, []);
    amplitude = reshape(amplitude, 1, []);
    
    % Generate phase_index_map
    phase_index_map = reshape(1:resolution * 2 * resolution * 2, resolution * 2, resolution * 2);
    phase_index_map = int64(phase_index_map); % Convert to int64
    
    % Delete the file if it already exists
    if isfile(filename)
        delete(filename);
    end
    
    % Write data to the H5 file
    h5create(filename, '/dx', size(dx));
    h5write(filename, '/dx', dx);
    h5create(filename, '/dy', size(dy));
    h5write(filename, '/dy', dy);
    h5create(filename, '/phase', size(phase));
    h5write(filename, '/phase', phase);
    h5create(filename, '/amplitude', size(amplitude));
    h5write(filename, '/amplitude', amplitude);
    h5create(filename, '/phase_index_map', size(phase_index_map), 'Datatype', 'int64');
    h5write(filename, '/phase_index_map', phase_index_map);
    
    fprintf('File %s saved successfully.\n', filename);
end
