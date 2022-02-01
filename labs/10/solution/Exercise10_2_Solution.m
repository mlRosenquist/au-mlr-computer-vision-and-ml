
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
%Sw = ?
[D,N] = size(train_data);
Sw = zeros(D,D);
for cc=1:nClasses
    curr_ind = find(train_lbls==uniqueClasses(cc));
    Xc = train_data(:,curr_ind);
    m_c = mean(Xc,2);
    Nc = size(Xc,2);
    Xc_centered = Xc - m_c*ones(1,Nc);
    Sw = Sw + (Xc_centered*Xc_centered');
end
    
% Calculate the between-class scatter matrix (Sb)
%Sb = ?
Sb = zeros(D,D);
m = mean(train_data,2);
for cc=1:nClasses
    curr_ind = find(train_lbls==uniqueClasses(cc));
    Xc = train_data(:,curr_ind);
    m_c = mean(Xc,2);
    Nc = size(Xc,2);
    m_c_centered = m_c - m;
    
    Sb = Sb + Nc*(m_c_centered*m_c_centered');
end


% Calculate the eigenvectors and eigenvalues of the matrix Sw^{-1}Sb
% 
% V: matrix of eigenvectors
% L: diagonal matrix of eigenvalues
[V,L] = eig(pinv(Sw)*Sb);  


% Sort eigenvalues in a descending order (and the eigenvectors, accordingly)
% Check the eigenvalues and observe the number of the significant ones. Can
% we determine this value without even calculating the eigenanalysis of Sw^{-1}Sb?
%Lsorted = ???
%Vsorted = ???
lambda = diag(L);
[Lsorted_vals,Lindices] = sort(lambda,'descend');
Lsorted = diag(Lsorted_vals);
Vsorted = V(:,Lindices);

% Keep only the eigenvectors corresponding to the maximal nFisherFaces eigenvalues
nFisherFaces = 39; % This is a hyper-parameter. How many are all the Fisher-Faces?
%Veigen = ???
Veigen = Vsorted(:,1:nFisherFaces);

% Visualize the nFisherFaces Eigen-Faces
figure
title('Fisher Faces')
ii = 0;  subFigY = 1;  subFigX = 5;
for yy=1:subFigY
    for xx=1:subFigX
        ii = ii+1;
        curr_image = reshape(Veigen(:,ii),[im_height im_width]);
        
        % scale in the interval [0,1]
        curr_image = curr_image - min(curr_image(:));
        curr_image = curr_image ./ max(curr_image(:));
        
        subplot(subFigY,subFigX,ii), imshow(curr_image,[]); title(['EigVec',num2str(ii)])
    end
end


% 'Project' both the training and test data to the LDA-space formed by the
% top nFisherFaces eigenvectors
%Wpca = ???
%Xtrain_pca = ???
%Xtest_pca = ???
Wlda = Vsorted(:,1:nFisherFaces);
Xtrain_lda = Wlda' * train_data;
Xtest_lda = Wlda' * test_data;

% classify the test data using the nearest neighbor classifier in the PCA space 

Mdl = fitcknn(Xtrain_lda', train_lbls,'Distance','euclidean','NumNeighbors',8,'Standardize',1,'BreakTies','nearest');
pred_lbls_knn = predict(Mdl, Xtest_lda');

%pred_lbls_knn = knnclassify(Xtest_lda', Xtrain_lda', train_lbls);

acc_knn_lda = length(find(pred_lbls_knn-test_lbls==0)) / length(test_lbls)

% Exercise: Apply Nearest Class Centroid-based classification on the test_data
mean_vecs = zeros(size(Xtrain_lda,1),nClasses);
for cc=1:nClasses
    curr_ind = find(train_lbls==uniqueClasses(cc));
    Xc = Xtrain_lda(:,curr_ind);
    mean_vecs(:,cc) = mean(Xc,2);
end

Dmat = zeros(nClasses,size(Xtest_lda,2));
for ii=1:size(Xtest_lda,2)
    for cc=1:nClasses
        diff_vec = mean_vecs(:,cc) - Xtest_lda(:,ii);
        Dmat(cc,ii) = sqrt(diff_vec'*diff_vec);
    end
end
[a,pred_lbls_ncs] = min(Dmat);   pred_lbls_ncs = pred_lbls_ncs';

acc_ncc_lda = length(find(pred_lbls_ncs-test_lbls==0)) / length(test_lbls)


