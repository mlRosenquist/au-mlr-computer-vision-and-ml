
% Perform Principal Component Analysis (PCA) and
% Visualize the projection matrix (EigenFaces)
clear;  clc;  close all;


% load the data
DATA_dir = 'ORL7\';
train_data = load([DATA_dir,'train_images_ORL7.mat']);  train_data = train_data.train_images;
train_lbls = load([DATA_dir,'train_labels_ORL7.mat']);  train_lbls = train_lbls.train_labels;
test_data = load([DATA_dir,'test_images_ORL7.mat']);    test_data = test_data.test_images;
test_lbls = load([DATA_dir,'test_labels_ORL7.mat']);    test_lbls = test_lbls.test_labels;
im_height = 40;  im_width = 30;


% We always apply the calculation of parameters on the training phase
% These parameters can be later used for online operation 
% (or on the test data, in order to measure performance)
% Apply Principal Component Analysis on the train_data
% Exercise: Calculate the data scatter matrix (St)
St = ???



% Calculate the eigenvectors and eigenvalues of St
% 
% V: matrix of eigenvectors
% L: diagonal matrix of eigenvalues
[V,L] = eig(St);  

% Exercise: sSort eigenvalues in a descending order (and the eigenvectors, accordingly)
% Check the eigenvalues and observe the number of the significant ones. Can
% we determine this value without even calculating the eigenanalysis of St?
Lsorted = ???
Vsorted = ???


% Keep only the eigenvectors corresponding to the maximal nEigFaces eigenvalues
% experiment with the number of nEigFaces and observe the impact on performance
nEigFaces = 5; % This is a hyper-parameter. How many are all the Eigen-Faces?
Veigen = Vsorted(:,1:nEigFaces);

% Visualize the nEigFaces Eigen-Faces
figure
title('Eigen Faces')
ii = 0;  subFigY = 1;  subFigX = 5;  % change subFigY and subFigX accordingly
for yy=1:subFigY
    for xx=1:subFigX
        ii = ii+1;
        curr_image = reshape(Veigen(:,ii),[im_height im_width]);
        subplot(subFigY,subFigX,ii), imshow(curr_image,[]); title(['EigVec',num2str(ii)])
    end
end

% Exercise: 'Project' both the training and test data to the PCA-space formed 
% by the top nEigFaces eigenvectors
Wpca = ???
Xtrain_pca = ???
Xtest_pca = ???


% classify the test data using the nearest neighbor classifier in the PCA space 
pred_lbls_knn = knnclassify(Xtest_pca', Xtrain_pca', train_lbls);

acc_knn_pca = length(find(pred_lbls_knn-test_lbls==0)) / length(test_lbls)

% Exercise: Apply Nearest Class Centroid-based classification on the test_data
pred_lbls_ncc = ???

acc_ncs_pca = length(find(pred_lbls_ncc-test_lbls==0)) / length(test_lbls) 

