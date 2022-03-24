clear all;close all;

% Load test image
I = imread('onion.png');
I = im2double(I);
figure(1)
subplot(1,3,1)
imshow(I),title('Input image')

% Spatial coordinates
[X,Y] = meshgrid(1:size(I,2),1:size(I,1));

% Exercise: Construct a feature map including RGB color features for each
% pixel. When you have the clustering up running, try adding spatial
% coordinates (variables X and Y defined above).

Z = I
Z(:,:,4) = X
Z(:,:,5) = Y
Z(:,:,4) = X/max(X(:));
Z(:,:,5) = Y/max(Y(:));
datapts = reshape(Z,[],5);
% Exercise: Use MATLAB's kmeans-function for clustering all 
% observations/pixels into 'numClust' clusters. Save the cluster indices 
% of all pixels in the variable 'cinds'.
% See 'doc kmeans' to make sure you pass the input variables correctly.
% Try changing the number of clusters and see the impact.
numClust = 10;

[cinds, C] = kmeans(datapts, numClust)
%cinds = zeros(size(I,1)*size(I,2),1); %% Replace this!

% Plot clusters
figure(1)
subplot(1,3,2)
imshow(mat2gray(reshape(double(cinds),size(X))))
title('Clusters')

% Generate cluster-based color image, where each pixel is assigned the
% average color of the cluster it belongs to.
Ifilter = zeros(size(I));
for cluster = 1:numClust
    subix = find(cinds==cluster);
    rgbval = [];
    for channel = 1:3
        tmp = I(:,:,channel);
        tmp = tmp(subix);
        rgbval(channel) = mean(tmp);
        tmp = zeros(size(I(:,:,1)));
        tmp(subix) = rgbval(channel);
        Ifilter(:,:,channel) = Ifilter(:,:,channel) + tmp;
    end
end
subplot(1,3,3)
imshow(Ifilter)
title('Colored clusters')