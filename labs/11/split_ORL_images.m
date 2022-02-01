% select a subset of MNIST dataset
clear;  clc;

noOfImagesPerClass = 7;
load('orl_data.mat');
load('orl_lbls.mat');


labels = unique(lbls);
noOfLbls = length(labels);
D = size(data,1);

train_images = [];  train_labels = [];
test_images = [];   test_labels = [];
for ll=1:noOfLbls
    curr_lbl = labels(ll);
    curr_ind = find(lbls==curr_lbl);
    train_ind = curr_ind(1:noOfImagesPerClass);
    test_ind = curr_ind(noOfImagesPerClass+1:end);
    
    train_images = [train_images data(:,train_ind)];
    train_labels = [train_labels; ll*ones(length(train_ind),1)];    
    
    test_images = [test_images data(:,test_ind)];
    test_labels = [test_labels; ll*ones(length(test_ind),1)]; 
end
save('train_images_ORL7.mat','train_images');
save('train_labels_ORL7.mat','train_labels');
save('test_images_ORL7.mat','test_images');
save('test_labels_ORL7.mat','test_labels');
