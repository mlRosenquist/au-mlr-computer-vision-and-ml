
clear;  clc;

% Dataset
% datasetName = 'YTFaces\YTF_data500';
datasetName = 'PubFigLFW\pubfiglfw__gabor_lbp_hog';


% load the data
load([datasetName,'_train_data.mat'])  % vectors (representations) for the training images
load([datasetName,'_train_lbls.mat'])  % labels of the training images
load([datasetName,'_test_data.mat'])   % vectors (representations) for the test images
load([datasetName,'_test_lbls.mat'])   % labels of the training images (we need them to calculate the performance)

% standardize the data
[train_data, Xm, Xstd] = cmptSdtParams(train_data);
test_data = cmptSdtParams2(test_data, Xm, Xstd);

% training/test image representations are stored in the columns of the matrix
[D,N] = size(train_data);
[D,M] = size(test_data);
classes = unique(train_lbls);
K = length(classes);

disp(['Number of training data: ',num2str(N)])
disp(['Number of test data: ',num2str(M)])
disp(['Dimensionality of image representations: ',num2str(D)])
disp(['Number of classes: ',num2str(K)])

%%%%%% train a Regression-based classifier with varying regularization parameter value
% first create the target matrix first
%%%% Exercise: Create the target matrix T having KxN elements
T = ???

% parameter set
Cvec = [10^10 10.^(-3:3)];

for cc=1:length(Cvec)
    
    C = Cvec(cc);

    %%% training phase
    % train the classifier using the training data
    %%% Exercise: Calculate the weights of the linear regression classifier
    W = ???

    
    %%% test phase
    % predict the outputs for the test data
    % Exercise: Calculate the responses for the test data using W
    Ot = ???
    
    % classify test images using the maximum response
    [maxOt,pred_lbls] = max(Ot);
    pred_lbls = pred_lbls';
    
    % measure the performance using the classification rate
    LMS_CR(cc) = length(find(pred_lbls-test_lbls==0)) / length(test_lbls);
    
    disp(['LMS regression based classification, C: ',num2str(1/C),', CR: ',num2str(LMS_CR(cc))])
end


%%%%%% train a Random Feature Regression-based classifier with varying regularization parameter value
% Here we have two hyper-parameters:
% 1) the number of dimensions of the random mapping L
% 2) the regularization parameter C

Lvec = [100 250 500 1000 1500 2000 2500];
Cvec = [10^10 10.^(-3:3)];

for ll=1:length(Lvec)
    for cc=1:length(Cvec)
        
        L = Lvec(ll);
        C = Cvec(cc);
        
        %%% training phase
        % calculate the random mapping
        % the following two lines implement a heuristic for selecting the scale parameter values A
        pp = randperm(1000);        ref_data = train_data(:,pp);
        Dref = distance_matrix(ref_data, ref_data);  A = 2*mean(mean(Dref));
    
        gamma = 1/A;  D = size(train_data,1);
        W = sqrt(2*gamma)*randn(L,D);         bias = 2*pi*rand(L,1);
        % Exercise: Calculate the randomized data representations. Use the
        % definition in slide 75 of lecture 'Lecture8.1-2 Classification'
        Ztrain = ???
    
        % train the classifier using the training data
        % Exercise: Calculate the regression weights A
        A = ???


        %%% test phase
        % calculate the mapped test data
        Ztest = sqrt(2/L) * cos(W*test_data + repmat(bias,1,size(test_data,2)));
        
        % predict the outputs for the test data;
        Ot = A' * Ztest;

        % classify test images using the maximum response
        [maxOt,pred_lbls] = max(Ot);
        pred_lbls = pred_lbls';

        % measure the performance using the classification rate
        RFR_CR(ll,cc) = length(find(pred_lbls-test_lbls==0)) / length(test_lbls);

        disp(['RFR based classification, L: ',num2str(L),' C: ',num2str(1/C),', CR: ',num2str(RFR_CR(ll,cc))])
    end
end
    

%%%%%% train a Radial Basis Function network classifier with varying regularization parameter value
% Here we have two hyper-parameters:
% 1) the number of hidden layer neurons L
% 2) the regularization parameter C

Lvec = [100 250 500 1000 1500 2000 2500];
Cvec = [10^10 10.^(-3:3)];

for ll=1:length(Lvec)
    for cc=1:length(Cvec)
        
        L = Lvec(ll);
        C = Cvec(cc);
        
        %%% training phase
        % calculate the neuron vectors by clustering the training data
        [label, ref_data] = fkmeans(double(train_data)', L);  ref_data = ref_data';
        
        % calculate the outputs of the hidden layer for all the training data
        % Exercise: Calcualte the outputs of the network's hidden layer for
        % the training vectors. Use the definitions in slide 37 of of lecture 'Lecture8.1-2 Classification'
        Htrain = ???
        
        % train the classifier using the training data
        A = ((Htrain*Htrain' + 1/C*eye(L))\Htrain) * T';


        %%% test phase
        % calculate the outputs of the hidden layer for all the test data
        Dtest = distance_matrix(test_data, ref_data);
        Htest = exp(-Dtest/(2*sigma));
        
        % predict the outputs for the test data;
        Ot = A' * Htest;

        % classify test images using the maximum response
        [maxOt,pred_lbls] = max(Ot);
        pred_lbls = pred_lbls';

        % measure the performance using the classification rate
        RBF_CR(ll,cc) = length(find(pred_lbls-test_lbls==0)) / length(test_lbls);

        disp(['RBF based classification, L: ',num2str(L),' C: ',num2str(1/C),', CR: ',num2str(RBF_CR(ll,cc))])
    end
end

