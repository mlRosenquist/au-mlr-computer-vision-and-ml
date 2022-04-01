

clear;  clc;

% Dataset
% datasetName = 'FERdatasets\bu';
% datasetName = 'FERdatasets\jaffe';
datasetName = 'FERdatasets/kanade';

% load data
load([datasetName,'.mat'])

% split the data using the provided indices
train_data = dataMatrix(:,indxTrain(1,:));       train_lbls = classes(indxTrain(1,:));
test_data = dataMatrix(:,indxTest(1,:));         test_lbls = classes(indxTest(1,:));


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
T = zeros(K,N);
for ii=1:N
    T(train_lbls(ii),ii) = 1.0;
end

% parameter set
Cvec = [10^10 10.^(-3:3)];

for cc=1:length(Cvec)
    
    C = Cvec(cc);

    %%% training phase
    % train the classifier using the training data
    W = ((train_data*train_data' + (1/C)*eye(D))\train_data) * T';

    
    %%% test phase
    % predict the outputs for the test data;
    Ot = W' * test_data;
    
    % classify test images using the maximum response
    [maxOt,pred_lbls] = max(Ot);
    
    % measure the performance using the classification rate
    LMS_CR(cc) = length(find(pred_lbls-test_lbls==0)) / length(test_lbls);
    
    disp(['LMS regression based classification, C: ',num2str(1/C),', CR: ',num2str(LMS_CR(cc))])
end


%%%%%% train a kernel-based Regression classifier using the RBF kernel function and
%%%%%% varying regularization parameter value
% Here we have two hyper-parameters:
% 1) RBF scaling value sigma (use a scaling version of the average distance among the training samples)
% 2) the regularization parameter C

% first create the target matrix first
T = zeros(K,N);
for ii=1:N
    T(train_lbls(ii),ii) = 1.0;
end

% parameter set
Cvec = [10^10 10.^(-3:3)];
Svec = [10^10 10.^(-3:3)];

for ss=1:length(Svec)
    for cc=1:length(Cvec)
        
        S = Svec(ss);
        C = Cvec(cc);
        
        %%% training phase
        % calculate the kernel matrix for the training data
        Dtrain = distance_matrix(train_data, train_data);  sigma = mean(mean(Dtrain));
        KHtrain = exp(-Dtrain/(S*sigma));
        
        % train the classifier using the training data
        A = (K + 1/C*eye(N)) \ T';


        %%% test phase
        % calculate the kernel matrix for the test data
        Dtest = distance_matrix(test_data, train_data);
        Ktest = exp(-Dtest/(S*sigma));
        
        % predict the outputs for the test data;
        Ot = A' * Ktest;

        % classify test images using the maximum response
        [maxOt,pred_lbls] = max(Ot);
        
        % measure the performance using the classification rate
        KR_CR(ss,cc) = length(find(pred_lbls-test_lbls==0)) / length(test_lbls);

        disp(['Kernel regression-based classification, S: ',num2str(S),' C: ',num2str(S),', CR: ',num2str(KR_CR(ss,cc))])
    end
end



%%%%%% train a linear SVM classifier following the one-versus-rest scheme
%%%%%% with a varying regularization parameter value
Cvec = [10^10 10.^(-3:3)];

for cc=1:length(Cvec)
            
    C = Cvec(cc);
    
    % train a OVR binary SVM model on the training data
    cmd = ['-t 0 -b -q -c ',num2str(C),' -q'];
    model = ovrtrain(train_lbls', train_data', cmd);
    
    % classify the test data using the above model
    pred_lbls = ovrpredict(test_lbls', test_data', model);
    
    % measure the performance using the classification rate
    pred_lbls = pred_lbls';
    SVM_CR(cc) = length(find(pred_lbls-test_lbls==0)) / length(test_lbls);

    disp(['Linear SVM-based classification, C: ',num2str(S),', CR: ',num2str(SVM_CR(cc))])
end


%%%%%% train a kernel SVM-based classifier using the RBF kernel function and
%%%%%% varying regularization parameter value
% Here we have two hyper-parameters:
% 1) RBF scaling value sigma (use a scaling version of the average distance among the training samples)
% 2) the regularization parameter C

% parameter set
Cvec = [10^10 10.^(-3:3)];
Svec = [10^10 10.^(-3:3)];

for ss=1:length(Svec)
    for cc=1:length(Cvec)
        
        S = Svec(ss);
        C = Cvec(cc);
        
        %%% training phase
        % calculate the kernel matrix for the training data
        Dtrain = distance_matrix(train_data, train_data);  sigma = mean(mean(Dtrain));
        Ktrain = exp(-Dtrain/(S*sigma));
        
        % train the classifier using the training data
        cmd = ['-t 4 -b -q -c ',num2str(C),' -q'];
        model = ovrtrain_kernel(train_lbls', [(1:N)' Ktrain], cmd);
        
        %%% test phase
        % calculate the kernel matrix for the test data
        Dtest = distance_matrix(test_data, train_data);
        Ktest = exp(-Dtest/(S*sigma));
        
        % predict the outputs for the test data;
        pred_lbls = ovrpredict_kernel(test_lbls', [(1:M)' Ktest'], model);
        
        % measure the performance using the classification rate
        pred_lbls = pred_lbls';
        KSVM_CR(ss,cc) = length(find(pred_lbls-test_lbls==0)) / length(test_lbls);

        disp(['Kernel SVM-based classification, S: ',num2str(S),' C: ',num2str(S),', CR: ',num2str(KR_CR(ss,cc))])
    end
end


