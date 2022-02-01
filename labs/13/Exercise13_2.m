clear all
close all
clc


% add to the path the functions of PSE package
CKNEQCut_pack = 'CKN-EQCut_functions';
addpath(PSE_pack)

% directory of images used for testing
path_IM = 'images\';
addpath(path_IM) % add to the path the directory of images to be used

% directory where the resulting images (saliency maps) will be stored
path_OUT = 'saliency_maps\';
mkdir(path_OUT); %In case it doesn't exist


% Load Models trained for saliency detection using Convolutional Kernel
% Networks and Extended QCut
% (L2 and KL extensions follow the loss functions that the CKN was trained on)
% For different parameter values we have a different trained model
% Change the extensions to get the model with desired training
% Models exploit 150,300 and 600 superpixels (SP)
% 250 filters (F) used only - the best result reported in original paper
model150=load('model_BSD150SP_250F_LAB_L2.mat');
model150=model150.modelfinal;
model300=load('model_BSD300SP_250F_LAB_L2.mat');
model300=model300.modelfinal;
model600=load('model_BSD600SP_250F_LAB_L2.mat');
model600=model600.modelfinal;

% make a list of the images in path_IM
%Change the .jpg extension accordingly if needed
contents=dir([path_IM '*.jpg']);


for i=1:length(contents)
    name=contents(i).name;
    name=name(1:end-4);
    imname=strcat(path_IM,name,'.jpg');
    image_now=imread(imname);
    
    %if image is gray-level convert it to RGB by repeating the gray level
    %image for each channel
    if length(size(image_now))==2;
        image_now=repmat(image_now,[1 1 3]);
    end
    
    %Check if the image contains any frames and exclude them
    [image_now,w]=removeframe(image_now);

    % perform saliency segmentation by using various pre-trained saliency segmentation models
    % of super-pixels. Fuse the resulting saliency maps using the average map
    [SalMapRes1]=SCKN_QCUT(image_now,sqrt(numel(image_now(:,:,1))/150),20,model150);
    [SalMapRes2]=SCKN_QCUT(image_now,sqrt(numel(image_now(:,:,1))/300),20,model300);
    [SalMapRes3]=SCKN_QCUT(image_now,sqrt(numel(image_now(:,:,1))/600),20,model600);
    
    %Average Over Resolutions
    SalMap=mat2gray(SalMapRes1)+mat2gray(SalMapRes2)+mat2gray(SalMapRes3);
    
    %Map Back to the Original Image (if frames were excluded)
    SalMapOrig=zeros(w(1),w(2));
    SalMapOrig(w(3):w(4),w(5):w(6))=SalMap;
    
    %Discretize the Saliency Map to uint8 format
    SalMapFinal=uint8(mat2gray(SalMapOrig)*255);
    
    imwrite(SalMapFinal,[path_OUT, name, '_SCKN.png']);
    imshow(SalMapFinal)
    
    pause    
end


