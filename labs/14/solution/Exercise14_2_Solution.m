% run_experiments
clear;   clc;

Hollywood_dir = 'Hollywood2\';

% dataset parameters
nTrainVideos = 823;  
nTestVideos = 884;

% Load train and test labels
action_names = {'AnswerPhone'; 'DriveCar'; 'Eat'; 'FightPerson'; 'GetOutCar';  'HandShake'; 'HugPerson'; 'Kiss'; 'Run'; 'SitDown'; 'SitUp'; 'StandUp'};
noOfActions = size(action_names,1); 
train_labels = zeros(nTrainVideos,noOfActions);
test_labels = zeros(nTestVideos,noOfActions);
for aa=1:noOfActions
    train_txt_name = sprintf('%s\\label_files\\%s_train.txt',Hollywood_dir,action_names{aa});
    test_txt_name = sprintf('%s\\label_files\\%s_test.txt',Hollywood_dir,action_names{aa});
    fileID1 = fopen(train_txt_name);  C1 = textscan(fileID1,'%s %d');  fclose(fileID1);
    fileID2 = fopen(test_txt_name);   C2 = textscan(fileID2,'%s %d');  fclose(fileID2);
    train_labels(:,aa) = C1{1,2};     test_labels(:,aa) = C2{1,2};
    clear train_txt_name test_txt_name fileID1 fileID2 C1 C2 
end
train_labels(train_labels~=1.0) = 0.0;     Ttrain = train_labels';
test_labels(test_labels~=1.0) = 0.0;       Ttest = test_labels';


% load video representations of the training videos based on Improved Dense Trajectories 
% (Traj, HOG, HOF, MBH) and Bag of Words
train_HOG = load([Hollywood_dir,'train_avBoW_hog.mat']);    train_HOG = train_HOG.train_avBoW_hog_cel{1};
train_HOF = load([Hollywood_dir,'train_avBoW_hof.mat']);    train_HOF = train_HOF.train_avBoW_hof_cel{1};
train_MBHx = load([Hollywood_dir,'train_avBoW_mbhx.mat']);  train_MBHx = train_MBHx.train_avBoW_mbhx_cel{1};
train_MBHy = load([Hollywood_dir,'train_avBoW_mbhy.mat']);  train_MBHy = train_MBHy.train_avBoW_mbhy_cel{1};
train_Traj = load([Hollywood_dir,'train_avBoW_traj.mat']);  train_Traj = train_Traj.train_avBoW_traj_cel{1};

% calculate the RBF kernel matrices for each descriptor based on chi-square distance 
% Exercise: write a function which calculates the chi-square distance matrix
Dtrain_hog = chiSquare_distance(train_HOG,train_HOG);
Ahog = mean(Dtrain_hog(:));

% Exercise: calculate the RBF kernel matrix using the chi-square distance
%Ktrain_hog = ???
Ktrain_hog = exp( -Dtrain_hog/(2*Ahog) );
disp('Train kernel HOG computed');

% Repeat the above steps for the HOF, MBHx, MHBy and Traj video representations
Dtrain_hof = chiSquare_distance(train_HOF,train_HOF);
Ahof = mean(Dtrain_hof(:));
%Ktrain_hof = ???
Ktrain_hof = exp( -Dtrain_hof/(2*Ahof) );
disp('Train kernel HOF computed');

Dtrain_mbhx = chiSquare_distance(train_MBHx,train_MBHx);
Ambhx = mean(Dtrain_mbhx(:));
%Ktrain_mbhx = ???
Ktrain_mbhx = exp( -Dtrain_mbhx/(2*Ambhx) );
disp('Train kernel MBHx computed');

Dtrain_mbhy = chiSquare_distance(train_MBHy,train_MBHy);
Ambhy = mean(Dtrain_mbhy(:));
%Ktrain_mbhy = ???
Ktrain_mbhy = exp( -Dtrain_mbhy/(2*Ambhy) );
disp('Train kernel MBHy computed');

Dtrain_traj = chiSquare_distance(train_Traj,train_Traj);
Atraj = mean(Dtrain_traj(:));
%Ktrain_traj = ???
Ktrain_traj = exp( -Dtrain_traj/(2*Atraj) );
disp('Train kernel Traj computed');

% Fuse the information encoded in different data representation types 
% calculating the average kernel matrix (You can also test the element-wise
% kernel matrix multiplication)
Ktrain = (Ktrain_hog + Ktrain_hof + Ktrain_mbhx + Ktrain_mbhy + Ktrain_traj) / 5;
%Ktrain = Ktrain_hog .* Ktrain_hof .* Ktrain_mbhx .* Ktrain_mbhy .* Ktrain_traj;

% Calculate the parameters of a kernel Least-Means Square Regression model
% and the target vectors stored in Ttrain
%Amat = ???
Amat = pinv(Ktrain)*Ttrain';




%%%%% Test phase
% load video representations of the test videos based on Improved Dense Trajectories 
% (Traj, HOG, HOF, MBH) and Bag of Words
test_HOG = load([Hollywood_dir,'test_avBoW_hog.mat']);    test_HOG = test_HOG.test_avBoW_hog_cel{1};
test_HOF = load([Hollywood_dir,'test_avBoW_hof.mat']);    test_HOF = test_HOF.test_avBoW_hof_cel{1};
test_MBHx = load([Hollywood_dir,'test_avBoW_mbhx.mat']);  test_MBHx = test_MBHx.test_avBoW_mbhx_cel{1};
test_MBHy = load([Hollywood_dir,'test_avBoW_mbhy.mat']);  test_MBHy = test_MBHy.test_avBoW_mbhy_cel{1};
test_Traj = load([Hollywood_dir,'test_avBoW_traj.mat']);  test_Traj = test_Traj.test_avBoW_traj_cel{1};

% calculate the RBF kernel matrices for each descriptor based on chi-square distance 
Dtest_hog = chiSquare_distance(train_HOG,test_HOG);
%Ktest_hog = ???
Ktest_hog = exp( -Dtest_hog/(2*Ahog) );
disp('Test kernel HOG computed');

Dtest_hof = chiSquare_distance(train_HOF,test_HOF);
%Ktest_hof = ???
Ktest_hof = exp( -Dtest_hof/(2*Ahof) );
disp('Test kernel HOF computed');

Dtest_mbhx = chiSquare_distance(train_MBHx,test_MBHx);
%Ktest_mbhx = ???
Ktest_mbhx = exp( -Dtest_mbhx/(2*Ambhx) );
disp('Test kernel MBHx computed');

Dtest_mbhy = chiSquare_distance(train_MBHy,test_MBHy);
%Ktest_mbhy = ???
Ktest_mbhy = exp( -Dtest_mbhy/(2*Ambhy) );
disp('Test kernel MBHy computed');

Dtest_traj = chiSquare_distance(train_Traj,test_Traj);
%Ktest_traj = ???
Ktest_traj = exp( -Dtest_traj/(2*Atraj) );
disp('Test kernel Traj computed');

% Fuse the information encoded in different data representation types 
% calculating the average kernel matrix (You can also test the element-wise
% kernel matrix multiplication)
Ktest = (Ktest_hog + Ktest_hof + Ktest_mbhx + Ktest_mbhy + Ktest_traj) / 5;

% Exercise: Calculate the output of the kernel Least-Means Square Regression 
% model for the test videos
%Otest = ???
Otest = Amat' * Ktest;

% Calculate the average precision over each action class
for cc=1:size(Otest,1)
   ap(cc) = average_precision(Otest(cc,:), Ttest(cc,:), 1.0, 0.0); 
end

% Calculate the mean Average Precision performance
mAP = mean(ap)