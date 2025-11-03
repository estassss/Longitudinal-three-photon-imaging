function process_images(plaqueChannelPath, vasculatureChannelPath)
% This function processes the images to address crosstalk issues
% It allows for either direct path inputs or interactive file selection

% Check if paths are provided, otherwise use GUI to select files
if nargin < 1 || isempty(plaqueChannelPath)
    [plaqueFileName, plaqueFolderPath] = uigetfile('*.tif', 'Select the Plaque Channel Image Stack');
    plaqueChannelPath = fullfile(plaqueFolderPath, plaqueFileName);
end
if nargin < 2 || isempty(vasculatureChannelPath)
    [vasculatureFileName, vasculatureFolderPath] = uigetfile('*.tif', 'Select the Vasculature Channel Image Stack');
    vasculatureChannelPath = fullfile(vasculatureFolderPath, vasculatureFileName);
end

% Load the image stacks
plaqueChannelStack = loadTiffStack(plaqueChannelPath);
vasculatureChannelStack = loadTiffStack(vasculatureChannelPath);

numImages = size(plaqueChannelStack, 3); % Assuming third dimension is the stack depth
outputFileName = 'processedStack.tif'; % Name of the output file

for i = 1:numImages
    plaqueImage = double(plaqueChannelStack(:,:,i));
    vasculatureImage = double(vasculatureChannelStack(:,:,i));
    
    avgBrightnessPlaque = mean(plaqueImage(:));
    avgBrightnessVasculature = mean(vasculatureImage(:));
    
    scalingFactor = avgBrightnessPlaque / avgBrightnessVasculature;
    adjustedVasculatureImage = vasculatureImage * scalingFactor;
    
    % Adjust subtraction order here: vasculature - plaques
    subtractedImage = adjustedVasculatureImage - plaqueImage;
    
    % Convert the subtractedImage to the original data type (e.g., uint8) before saving
    subtractedImage = uint8(subtractedImage);
    
    % Save each processed image to a TIFF file
    if i == 1
        imwrite(subtractedImage, outputFileName, 'Compression','none');
    else
        imwrite(subtractedImage, outputFileName, 'WriteMode', 'append', 'Compression','none');
    end
end

disp(['Processed images saved to ', outputFileName]);
end

function stack = loadTiffStack(filePath)
    info = imfinfo(filePath);
    numImages = numel(info);
    stack = zeros(info(1).Width, info(1).Height, numImages, 'uint8');
    for i = 1:numImages
        stack(:,:,i) = imread(filePath, i, 'Info', info);
    end
end

