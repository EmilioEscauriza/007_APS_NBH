filepath = '/Users/emilioescauriza/Documents/repos/007_APS_NBH/data/A073/data/A073_rocking_curve_cropped.h5';

% Preview the file structure (optional, equivalent to what you just set up in Python)
h5disp(filepath);

% Read the full 3D array
data = h5read(filepath, '/data');  % returns 350 x 350 x 76 (MATLAB reverses HDF5 dimension order)