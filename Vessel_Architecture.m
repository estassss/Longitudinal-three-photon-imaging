% Specify the path and filename of the analysis file
PathName = '/Users/elinestas/DeepVess';
FileName = 'Analysis-223785.1.V-V_fwd.mat';

% Load the analysis file
analysisFilePath = fullfile(PathName, FileName);
load(analysisFilePath);

% Check if Skel is not empty
if isempty(Skel)
    error('Skel is empty. Please check the data file.');
else
    disp('Skel loaded successfully.');
end

% Convert V1 to logical type for skeletonization
V1_logical = V1 > 0; % Assuming that V1 contains non-zero values for vessels and zero for background

% Initialize arrays for storing the calculated properties of each vessel segment
Diameters = zeros(size(Skel, 2), 1); % For diameters
Lengths = zeros(size(Skel, 2), 1); % For lengths
Tortuosities = zeros(size(Skel, 2), 1); % For tortuosities
InterVesselDistances = zeros(size(Skel, 2), 1); % For intervessel distances

% Prepare the environment
distTransform = bwdist(~V1_logical); % Calculates distance from each foreground pixel to the nearest background pixel in the preprocessed volume

% Loop through each vessel segment identified in 'Skel'
for i = 1:size(Skel, 2)
    segmentCoords = Skel{1, i}; % coordinates of the vessel segment
    
    % Diameter Calculation
    segmentDiameters = zeros(size(segmentCoords, 1), 1);
    for j = 1:size(segmentCoords, 1)
        radius = distTransform(segmentCoords(j, 1), segmentCoords(j, 2), segmentCoords(j, 3)); %
        segmentDiameters(j) = abs(radius * 2); % Diameter = radius * 2
    end
    Diameters(i) = abs(median(segmentDiameters)); % Median diameter
    
    % Length Calculation
    distance = sqrt(sum(diff(segmentCoords).^2, 2)); % Euclidean distances between consecutive points
    Lengths(i) = abs(sum(distance)); % Total length
    
    % Tortuosity Calculation
    if size(segmentCoords, 1) > 1
        straightLineDist = sqrt(sum((segmentCoords(1,:) - segmentCoords(end,:)).^2)); % Straight-line distance
        Tortuosities(i) = abs(Lengths(i) / straightLineDist); % Path length / Straight-line distance
    else
        Tortuosities(i) = 1; % Default tortuosity for single-point segments
    end

    % Intervessel Distance Calculation
    midpoint_i = mean(segmentCoords, 1); % Calculate the midpoint of the current segment
    minDistance = inf; % Initialize the minimum distance as infinity
    
    for j = 1:size(Skel, 2)
        if i ~= j % Ensure we do not compare the segment with itself
            segmentCoords_j = Skel{1, j}; % coordinates of the other vessel segment
            midpoint_j = mean(segmentCoords_j, 1); % Calculate the midpoint of the other segment
            
            % Calculate the Euclidean distance between the two midpoints
            distance = sqrt(sum((midpoint_i - midpoint_j).^2));
            
            % Update the minimum distance if a smaller one is found
            if distance < minDistance
                minDistance = distance;
            end
        end
    end
    
    InterVesselDistances(i) = minDistance; % Store the minimum distance
end

% Density fraction Calculation
% V1 for density calculation
[numRows, numCols, numSlices] = size(V1); % Dimensions of V1

% Initialize an array to store the calculated density for each slice.
Densities = zeros(numSlices, 1);

% Loop through each slice to calculate its density.
for i = 1:numSlices
    currentSlice = V1_logical(:, :, i); % Extract the i-th slice from V1.
    vesselLength = nnz(currentSlice); % Count non-zero elements representing vessel parts.
    sliceArea = numRows * numCols; % Total area of the slice.
    Densities(i) = abs(vesselLength / sliceArea); % Density as the ratio of vessel length to slice area.
end

% Calculate the median Z-coordinate for each parameter

% Initialize arrays to store the median Z-coordinate
MedianImageNumber = zeros(size(Skel, 2), 1);

% Loop through each vessel segment identified in 'Skel'
for i = 1:size(Skel, 2)
    segmentCoords = Skel{1, i}; % coordinates of the vessel segment
    
    % Extract Z-coordinates of the current segment
    segmentZCoords = segmentCoords(:, 3); 

    % Median Z-coordinate
    MedianImageNumber(i) = median(segmentZCoords); 
    
end

% Save the results
save(fullfile(PathName, ['Vess_Architecture-', FileName]), 'im', 'Skel', 'C', 'V', 'Diameters', 'Lengths', 'Tortuosities', 'InterVesselDistances', 'Densities', 'MedianImageNumber')

% Visualize V1 in 3D
volumeViewer(V1_logical);

clear FileName;
