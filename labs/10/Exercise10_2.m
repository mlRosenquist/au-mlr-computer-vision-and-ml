
% Perform Linear Discriminant 
% Visualize the projection matrix (FisherFaces)
clear;  clc;  close all;


% load the data
DATA_dir = 'ORL7\';
train_data = load([DATA_dir,'train_images_ORL7.mat']);  train_data = train_data.train_images;
train_lbls = load([DATA_dir,'train_labels_ORL7.mat']);  train_lbls = train_lbls.train_labels;
test_data = load([DATA_dir,'test_images_ORL7.mat']);    test_data = test_data.test_images;
test_lbls = load([DATA_dir,'test_labels_ORL7.mat']);    test_lbls = test_lbls.test_labels;
im_height = 40;  im_width = 30;
uniqueClasses = unique(train_lbls);
nClasses = length(uniqueClasses);

% We always apply the calculation of parameters on the training phase
% These parameters can be later used for online operation 
% (or on the test data, in order to measure performance)
% Apply Linear Discriminantt Analysis on the train_data
% Exercise: calculate the within-class scatter matrix (Sw)
Sw = ???

    
% Exercise: calculate the between-class scatter matrix (Sb)
Sb = ???



% Calculate the eigenvectors and eigenvalues of the matrix Sw^{-1}Sb
% 
% V: matrix of eigenvectors
% L: diagonal matrix of eigenvalues
reg = 0.001;
Sw = Sw + reg*eye(size(Sw));
[V,L] = eig(Sw\Sb);  

% Exercise: sort eigenvalues in a descending order (and the eigenvectors, accordingly)
% Check the eigenvalues and observe the number of the significant ones. Can
% we determine this value without even calculating the eigenanalysis of Sw^{-1}Sb?
Lsorted = ???
Vsorted = ???


% Keep only the eigenvectors corresponding to the maximal nFisherFaces eigenvalues
% experiment with the number of nEigFaces and observe the impact on performance
nFisherFaces = 5; % This is a hyper-parameter. How many are all the Fisher-Faces?
Veigen = Vsorted(:,1:nFisherFaces);

% Visualize the nFisherFaces Eigen-Faces
figure
title('Fisher Faces')
ii = 0;  subFigY = 1;  subFigX = 5; % change subFigY and subFigX accordingly
for yy=1:subFigY
    for xx=1:subFigX
        ii = ii+1;
        curr_image = reshape(Veigen(:,ii),[im_height im_width]);
        subplot(subFigY,subFigX,ii), imshow(curr_image,[]); title(['EigVec',num2str(ii)])
    end
end


% Exercise: 'Project' both the training and test data to the LDA-space formed
% by the top nFisherFaces eigenvectors
Wpca = ???
Xtrain_lda = ???
Xtest_lda = ???


% classify the test data using the nearest neighbor classifier in the PCA space 
pred_lbls = knnclassify(Xtest_lda', Xtrain_lda', train_lbls);

acc_knn_lda = length(find(pred_lbls-test_lbls==0)) / length(test_lbls)

% Exercise: Apply Nearest Class Centroid-based classification on the test_data
pred_lbls_ncc = ???

acc_ncs_pca = length(find(pred_lbls_ncc-test_lbls==0)) / length(test_lbls) 
