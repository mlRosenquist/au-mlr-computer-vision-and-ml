% Human action recognition based on human body poses

clear;  clc;
close all;

% data directory
Weizemann_dir = 'Weizemann\';


% parameters 
train_actors = (1:6);
test_actors = (7:9);
pose_size = [32 32];
nDynemes = 20;
subFigY = 4;  subFigX = 5; % you need to change subFigY and subFibX, so that subFigY*subFigX = nDynemes
Dlda = 8;


%%%%% load the data (human body poses) and split them in training and test 
%[train_poses,test_poses,train_vidLbls,test_vidLbls,train_poseLbls,test_poseLbls] = load_weizemann_data(directory,train_actors,test_actors,pose_size);
% load the data
load([Weizemann_dir,'classification_masks.mat']);

% names of persons and actions in the database
actors = {'daria';'denis';'eli';'ido';'ira';'lena';'lyova';'moshe';'shahar'};
actions = {'bend';'jack';'jump';'pjump';'run';'side';'skip';'walk';'wave1';'wave2';'run1';'run2';'walk1';'walk2'};
action_labels = [1,2,3,4,5,6,7,8,9,9,5,5,8,8];


% load training poses
train_poses = [];
train_vidLbls = [];
train_poseLbls = [];
pose_iter = 0;
video_iter = 0;
for ii=1:length(train_actors)
    for jj=1:length(actions)

        
        % load the next video
        video_name = [actors{train_actors(ii)},'_',actions{jj}];
        field_exists = isfield(aligned_masks,video_name);
        if field_exists~=1, continue;  end

        curr_video = getfield(aligned_masks,video_name);
        [W,H,F] = size(curr_video);
        
        % store the action label for this video in train_vidLbls
        video_iter = video_iter +1;
        train_vidLbls(video_iter) = action_labels(jj);
        
        % store the poses of this video in train_poses
        for ff=1:F
            
            video_frame = curr_video(:,:,ff);
            
            % Exercise: Find the bounding box of the biggest blob in video_frame.
            % Add a border of two pixels on each side (top-down-left-right)
            %body_pose = ???
            st = regionprops(video_frame, 'BoundingBox', 'Area' );
            [maxArea, indexOfMax] = max([st.Area]);
            
            % find the coordinates of the bounding box
            bb_x = st(indexOfMax).BoundingBox(1);
            bb_y = st(indexOfMax).BoundingBox(2);
            bb_w = st(indexOfMax).BoundingBox(3);
            bb_h = st(indexOfMax).BoundingBox(4);
            
            % add a border of two pixels on each side (top-down-left-right)
            bb_x = floor(bb_x) -2;   if bb_x<0, bb_x=0; end
            bb_y = floor(bb_y) -2;   if bb_y<0, bb_y=0; end
            bb_w = floor(bb_w) +4;   bb_h = floor(bb_h) +4;

            % extract the body pose
            body_pose = video_frame(bb_y:bb_y+bb_h,bb_x:bb_x+bb_w);
            
            % Resize the pose to a fixed size (pose_size)
            pose = imresize(double(body_pose),[pose_size(1) pose_size(2)]);
            imshow(pose);
            
            % add this pose to the pose list
            pose_iter = pose_iter +1;
            pose_vector = reshape(pose,[],1);
            %pose2 = reshape(pose_vector,32,32); % if we want to obtain the pose matrix again
            train_poses(:,pose_iter) = pose_vector;
            train_poseLbls(pose_iter) = video_iter;  % we want this in order to extract all poses of a video later
        end  
        
        disp(['train video: ',video_name,'  (actor ',num2str(ii),'/',num2str(length(train_actors)),', action ',num2str(jj),'/',num2str(length(actions)),')']);
    end
end
close all;

% load test poses
test_poses = [];
test_vidLbls = [];
test_poseLbls = [];
pose_iter = 0;
video_iter = 0;
for ii=1:length(test_actors)
    for jj=1:length(actions)
        
        % load the next video
        video_name = [actors{test_actors(ii)},'_',actions{jj}];
        field_exists = isfield(aligned_masks,video_name);
        if field_exists~=1, continue;  end

        curr_video = getfield(aligned_masks,video_name);
        [W,H,F] = size(curr_video);
        
        % store the action label for this video in train_vidLbls
        video_iter = video_iter +1;
        test_vidLbls(video_iter) = action_labels(jj);
        
        % store the poses of this video in train_poses
        for ff=1:F
        
            video_frame = curr_video(:,:,ff);
            
            % Exercise: Find the bounding box of the biggest blob in video_frame.
            % Add a border of two pixels on each side (top-down-left-right)
            %body_pose = ???
            st = regionprops(video_frame, 'BoundingBox', 'Area' );
            [maxArea, indexOfMax] = max([st.Area]);
            
            % find the coordinates of the bounding box
            bb_x = st(indexOfMax).BoundingBox(1);
            bb_y = st(indexOfMax).BoundingBox(2);
            bb_w = st(indexOfMax).BoundingBox(3);
            bb_h = st(indexOfMax).BoundingBox(4);
            
            % add a border of two pixels on each side (top-down-left-right)
            bb_x = floor(bb_x) -2;   if bb_x<0, bb_x=0; end
            bb_y = floor(bb_y) -2;   if bb_y<0, bb_y=0; end
            bb_w = floor(bb_w) +4;   bb_h = floor(bb_h) +4;

            % extract the body pose
            body_pose = curr_video(bb_y:bb_y+bb_h,bb_x:bb_x+bb_w,ff);
            
            %Resize the pose to a fixed size (pose_size)
            pose = imresize(double(body_pose),[pose_size(1) pose_size(2)]);
            %imshow(pose);
            
            % add this pose to the pose list
            pose_iter = pose_iter +1;
            pose_vector = reshape(pose,[],1);
            %pose2 = reshape(pose_vector,32,32); % if we want to obtain the pose matrix again
            test_poses(:,pose_iter) = pose_vector;
            test_poseLbls(pose_iter) = video_iter;  % we want this in order to extract all poses of a video later
        end  
        
        disp(['test video: ',video_name,'  (actor ',num2str(ii),'/',num2str(length(test_actors)),', action ',num2str(jj),'/',num2str(length(actions)),')']);
    end
end
close all;
%%%%%%%%%%%%%%%%%%%%%

%%%% Cluster the poses of the training videos to obtain the dynemes
% Exercise: Cluster the poses of the training videos to obtain the dynemes
% You can use the kmeans() function of Matlab
%Dynemes = ???
[idx,Centroids] = kmeans(train_poses',nDynemes);
Dynemes = Centroids';

% Visualize the Dynemes (prototype body poses)
figure
title('Dynemes')
ii = 0;  
for yy=1:subFigY
    for xx=1:subFigX
        ii = ii+1;
        curr_image = reshape(Dynemes(:,ii),[pose_size(1) pose_size(2)]);
        subplot(subFigY,subFigX,ii), imshow(curr_image,[]); title(['Dynemes',num2str(ii)])
    end
end

%%%% Calculate the fuzzy histogram-based representation for each video
nTrainVideos = max(train_poseLbls);
train_action_vectors = zeros(nDynemes,nTrainVideos);

for ii=1:nTrainVideos
    curr_pose_indices = find(train_poseLbls==ii);
    curr_poses = train_poses(:,curr_pose_indices);
    Nt = size(curr_poses,2);
    
    % Exercise: calculate the distance matrix Dmat containing the distances 
    % of each pose to each Dyneme
    Dmat = zeros(nDynemes,Nt);
    %Dmat = ???
    %Dmat = ((sum(Dynemes'.^2,2)*ones(1,Nt))+(sum(curr_poses'.^2,2)*ones(1,nDynemes))'-(2*Dynemes'*curr_poses));
    for dd=1:nDynemes
        for kk=1:Nt
            diff = Dynemes(:,dd) - curr_poses(:,kk);
            Dmat(dd,kk) = sqrt(diff'*diff);
        end
    end
    
    % avoid the division with zeros
    eps_val = 0.0001;
    Dmat(Dmat<eps_val) = eps_val;
    
    % Fuzzy similarity matrix (uzing a fuzzification parameter value fuzz)
    fuzz = 20;
    Dmat = Dmat .^ (-fuzz);  % fuzzy similarity
    sumD = sum(Dmat);  sumD(sumD==0)=1;  Dmat = Dmat./(ones(size(Dmat,1),1)*sumD);
    
    train_action_vectors(:,ii) = mean(Dmat,2);
end


% Project the train_action_vectors to the LDA space of dimensions Dlda
% Exercise: calculate the within-class scatter matrix (Sw)
[D,N] = size(train_action_vectors);
uniqueClasses = unique(train_vidLbls);
nClasses = length(uniqueClasses);
%Sw = ???
Sw = zeros(D,D);
for cc=1:nClasses
    curr_ind = find(train_vidLbls==uniqueClasses(cc));
    Xc = train_action_vectors(:,curr_ind);
    m_c = mean(Xc,2);
    Nc = size(Xc,2);
    Xc_centered = Xc - m_c*ones(1,Nc);
    Sw = Sw + (Xc_centered*Xc_centered');
end
    
% Calculate the between-class scatter matrix (Sb)
%Sb = ???
Sb = zeros(D,D);
m = mean(train_action_vectors,2);
for cc=1:nClasses
    curr_ind = find(train_vidLbls==uniqueClasses(cc));
    Xc = train_action_vectors(:,curr_ind);
    m_c = mean(Xc,2);
    Nc = size(Xc,2);
    m_c_centered = m_c - m;
    
    Sb = Sb + Nc*(m_c_centered*m_c_centered');
end


% Calculate the eigenvectors and eigenvalues of the matrix Sw^{-1}Sb
% V: matrix of eigenvectors
% L: diagonal matrix of eigenvalues
[V,L] = eig(pinv(Sw)*Sb);  

% Sort eigenvalues in a descending order (and the eigenvectors,
% accordingly) and keep the top Dlda eigenvectors to form Wlda
lambda = diag(L);
[Lsorted_vals,Lindices] = sort(lambda,'descend');
Lsorted = diag(Lsorted_vals);
Vsorted = V(:,Lindices);
Wlda = Vsorted(:,1:Dlda);


% Exercise: Project the training data to the LDA space and find the parameters 
% of the NCC classifier
%proj_train_action_vectors = ???
%action_centers = ???
proj_train_action_vectors = Wlda' * train_action_vectors;
nClasses = length(unique(train_vidLbls));

action_centers = zeros(Dlda,nClasses);
for cc=1:nClasses
    curr_indices = find(train_vidLbls==cc);
    curr_vectors = proj_train_action_vectors(:,curr_indices);
    action_centers(:,cc) = mean(curr_vectors,2);
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Test phase
%%%% Calculate the fuzzy histogram-based representation for each video
nTestVideos = max(test_poseLbls);
test_action_vectors = zeros(nDynemes,nTestVideos);

for ii=1:nTestVideos
    curr_pose_indices = find(test_poseLbls==ii);
    curr_poses = test_poses(:,curr_pose_indices);
    Nt = size(curr_poses,2);
    
    % Exercise: calculate the distance matrix Dmat containing the distances 
    % of each pose to each Dyneme
    Dmat = zeros(nDynemes,Nt);
    %Dmat = ???
    %Dmat = ((sum(Dynemes'.^2,2)*ones(1,Nt))+(sum(curr_poses'.^2,2)*ones(1,nDynemes))'-(2*Dynemes'*curr_poses));
    for dd=1:nDynemes
        for kk=1:Nt
            diff = Dynemes(:,dd) - curr_poses(:,kk);
            Dmat(dd,kk) = sqrt(diff'*diff);
        end
    end
    
    % avoid the division with zeros
    eps_val = 0.0001;
    Dmat(Dmat<eps_val) = eps_val;
    
    % Fuzzy similarity matrix (uzing a fuzzification parameter value fuzz)
    fuzz = 20;
    Dmat = Dmat .^ (-fuzz);  % fuzzy similarity
    sumD = sum(Dmat);  sumD(sumD==0)=1;  Dmat = Dmat./(ones(size(Dmat,1),1)*sumD);
    
    test_action_vectors(:,ii) = mean(Dmat,2);
end


% Exercise: project the test action vectors to the LDA space and classify them 
% using the NCC classifier
%proj_test_action_vectors = ???
%pred_labels = ???
proj_test_action_vectors = Wlda' * test_action_vectors;
nTestSamples = size(proj_test_action_vectors,2);

DistMat = zeros(nClasses,nTestSamples);
for ii=1:nTestSamples
    for cc=1:nClasses
        diff = proj_test_action_vectors(:,ii) - action_centers(:,cc);
        DistMat(cc,ii) = sqrt(diff'*diff);
    end
end

[distances,pred_labels] = min(DistMat);

accuracy = length(find(pred_labels-test_vidLbls==0)) / length(test_vidLbls)
