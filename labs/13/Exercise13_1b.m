clear all
close all
clc

% add to the path the functions of PSE package
PSE_pack = 'PSE_functions';
addpath(PSE_pack)

% directory of images used for testing
path_IM = 'images\';
addpath(path_IM) % add to the path the directory of images to be used

% directory where the resulting images (saliency maps) will be stored
path_OUT = 'saliency_maps\';
mkdir(path_OUT); %In case it doesn't exist


% make a list of the images in path_IM
%Change the .jpg extension accordingly if needed
contents=dir([path_IM '*.jpg']);


% iterate for all images
for i=1:length(contents)
    
    name = contents(i).name;
    name = name(1:end-4);
    imname = strcat(path_IM,name,'.jpg');
    image_now = (imread(imname));
    
    % if image is gray-level convert it to RGB by repeating the gray level
    % image for each channel
    if length(size(image_now))==2;
        image_now = repmat(image_now,[1 1 3]);
    end
    
    % Check if the image contains any frames and exclude them
    [image_now,w] = removeframe(image_now);
    
    % perform a hierarchical saliency segmentation by using various numbers
    % of super-pixels. Fuse the resulting saliency maps using the average map
    nSuperPixelsVec = [300 600 1200];
    
    for ss = 1:length(nSuperPixelsVec)
        
        % Over-segment the image in super-pixels using the SLIC algorithm
        % Find Superpixels, LAB Mean values and superpixel indices on image boundary
        nSuperPixels = nSuperPixelsVec(ss);
        nSP = round(sqrt(numel(image_now(:,:,1)) / nSuperPixels));  
        [LMean, AMean, BMean, suppixel, boundaries,PixNum, LabelLine,width, height]=SolveSlic(image_now,nSP,20);

        % Use the image coordinates of each super-pixel to determine their
        % extended neighborhood (to be used for generating the graphs structure
        [neighbourhood,LF,max_label]=FindNeighbours(suppixel);

        % Exercise: Calculate a square matrix containing the pair-wise affinities 
        % of all super-pixels using the Heat kernel (use the mean pair-wise distance
        % value to determine the value of hyper-parameter sigma)
        Amat = ???
        
        
    
        % Assign Affinities based on the above affinity matrix and the 
        % super-pixel neighborhood information (super-pixels are similar
        % when they have similar colors and are close in the image lattice)
        H = AffinityAssign(neighbourhood,LF,Amat,max_label);

        %Estimate Possible Foreground (lower the probability of being salient
        % for super-pixels touching the boundary of the image (boundary saliency cue)
        H = UpdateDiagonal(H,boundaries);

        % Calculate Saliency Map
        SalMap_res = FindSal(H,PixNum, LabelLine,width, height);
        
        if ss==1
            SalMap = mat2gray(SalMap_res) / length(nSuperPixels);
        else
            SalMap = SalMap + mat2gray(SalMap_res) / length(nSuperPixels);
        end
    end
    
    
    %Map Back to the Original Image (if frames were excluded)
    SalMapOrig=zeros(w(1),w(2));
    SalMapOrig(w(3):w(4),w(5):w(6))=SalMap;
    
    %Discretize the Saliency Map to uint8 format
    SalMapFinal=uint8(mat2gray(SalMapOrig)*255);
    
    %Save the result
    imwrite(SalMapFinal,[path_OUT, name, '_multiResolution_HeatKernel_PSE.png']);
    imshow(SalMapFinal)

    pause
end
