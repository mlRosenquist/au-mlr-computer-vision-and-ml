clear all;close all;

% Load test image
I = imread('onion.png');
I = im2double(I);
figure(1)
subplot(1,2,1),imshow(I),title('Input image')

% Spatial coordinates
[X,Y] = meshgrid(1:size(I,2),1:size(I,1));
maxX = max(X(:));
maxY = max(Y(:));
X = X/maxX;
Y = Y/maxY;

Z = I;
Z(:,:,4) = X;
Z(:,:,5) = Y;

% Set some parameters
radius = 0.2;    % What is this?
numSeed = 100;      % Number of seed points

% Generate random seed points
seedPointsIx = randi(prod(size(X)),1,numSeed);
[seedRow,seedCol] = ind2sub(size(X),seedPointsIx);

% Define some variables for monitoring how the mean-shift tracks evolve
% over time.
terminateTrack = zeros(1,numSeed); % Set to one to terminate track
prevStepSize = zeros(1,numSeed);  % Size of last change in window position
for i = 1:numSeed
    visitList{i} = []; % Store list of pixels visited here (one list per seed point)
    seedRowlist{i} = seedRow(i); % For plotting tracks
    seedCollist{i} = seedCol(i); % For plotting tracks
end

done = 0;
iter = 1;
while not(done)
    
    % Loop over seed points
    for i = 1:numSeed
        
        % Interpolate to find feature vector at current seed position,
        % which is in general a non-integer coordinate.
        % Only consider non-terminated tracks
        if not(terminateTrack(i))
            
            % Extract image patch
            row = round(seedRow(i));
            if row<2, row=2;end
            if row>=size(I,1), row = size(I,1)-1;end
            col = round(seedCol(i));
            if col<2, col=2;end
            if col>=size(I,2), col = size(I,2)-1;end
            patch = Z(row-1:row+1,col-1:col+1,:);
            [Xi,Yi] = meshgrid(col-1:col+1,row-1:row+1);
            
            % Interpolate
            for dim = 1:5
                Zref_vec(dim) = interp2(Xi,Yi,patch(:,:,dim),seedCol(i),seedRow(i));
            end
                
            % If interpolation fails, terminate track
            if not(isempty(find(isnan(Zref_vec)))), terminateTrack(i)=1;end
        end
        
        % Do mean shift
        % Only consider non-terminated tracks
        if not(terminateTrack(i))
            
            % Task 1:
            % Calculate distance from seed point (Zref_vec) to all other pixels (Z).
            % (use Euclidian distance in feature space)
            %dist = rand(size(X))*10; % dist is set to random values. Change this!!!
            Zref = repmat(reshape(Zref_vec,[1 1 5]),[size(X) 1]);
            dist = sqrt(sum((Z-Zref).^2,3));

            % Find all pixels whose distances to the seed point are smaller
            % than a threshold (radius)
            add_to_list = find(dist<radius);
            
            if not(isempty(add_to_list))
                
                % Add new pixels to the current track
                visitList{i} = union(visitList{i},add_to_list);
                
                % Task 2:
                % Calculate the mean shift, i.e., the new coordinate of
                % the seed point.
                %new_seedRow = randi(size(I,1)); % Random value. Change this!!
                %new_seedCol = randi(size(I,2)); % Random value. Change this!!
                [row,col] = ind2sub(size(X),add_to_list);
                new_seedRow = mean(row);
                new_seedCol = mean(col);

                % Size of change in window position
                change = sqrt((new_seedRow-seedRow(i))^2+(new_seedCol-seedCol(i))^2);
                
                % If the change is larger than the previous change,
                % terminate track. This is a simple hack to avoid wild
                % fluctuations around the cluster peak. Uncomment and see
                % what happens...
                if iter > 1 & change>prevStepSize(i)
                    terminateTrack(i) = 1;
                end
                
                % Store step size
                prevStepSize(i) = change;
                
                % Apply window shift if change is larger than 0.1 pixels
                if change>0.1 & not(terminateTrack(i))
                    seedRow(i) = new_seedRow;
                    seedCol(i) = new_seedCol;
                    seedRowlist{i} = [ seedRowlist{i} new_seedRow ]; % For plotting tracks
                    seedCollist{i} = [ seedCollist{i} new_seedCol ]; % For plotting tracks
                else
                    % Terminate track if window change is too small
                    terminateTrack(i) = 1;
                end
            end
        end
    end
    
    % Terminate loop?
    iter = iter + 1;
    if iter == 100, done = 1; end
    if isempty(find(terminateTrack==0)); done = 1; end
    
    % Show tracks
    figure(1)
    subplot(1,2,2)
    imshow(I)
    hold on
    for i = 1:numSeed
        rows = seedRowlist{i};
        cols = seedCollist{i};
        plot(cols,rows,'w')
        plot(cols(end),rows(end),'r.')
    end
    hold off
    title('Mean shift tracks')
    drawnow
end
disp('Done!')

% Remove bad clusters and store each of the final seed points (=cluster
% peaks). A bad cluster is one, where the feature cannot be calculated
% using interpolation (for whatever reason), i.e., when the interpolated
% feature values are NaN (not a number).
currentCluster = 1;
clusterCenters = [];
visitList2 = {};
for i = 1:numSeed
    
    % Interpolate to find feature vector at current seed position
    row = round(seedRow(i));
    if row<2, row=2;end
    if row>=size(I,1), row = size(I,1)-1;end
    col = round(seedCol(i));
    if col<2, col=2;end
    if col>=size(I,2), col = size(I,2)-1;end
    patch = Z(row-1:row+1,col-1:col+1,:);
    [Xi,Yi] = meshgrid(col-1:col+1,row-1:row+1);
    for dim = 1:5
        Zref_vec(dim) = interp2(Xi,Yi,patch(:,:,dim),seedCol(i),seedRow(i));
    end

    % Are we good?
    if isempty(find(isnan(Zref_vec)))
        clusterCenters(currentCluster,:) = Zref_vec;
        visitList2{currentCluster} = visitList{i};
        currentCluster = currentCluster + 1;
    end    
end
numClusters = currentCluster - 1;

% Do greedy (pairwise) merging of nearby clusters
% (This is bottom-up clustering)
done = 0;
while not(done)
    
    % Calcualte inter-cluster distances
    A = [];
    for i = 1:numClusters
        for j = 1:numClusters
            A(i,j) = sqrt(sum((clusterCenters(i,:)-clusterCenters(j,:)).^2));
        end
    end
    
    % Force diagonal entries of A to high value (to avoid merging clusters
    % with themselves...)
    A = A + eye(numClusters)*max(A(:));
    
    % Find two closest clusters (indexed "i" and "j")
    [minval,minix] = min(A(:));
    [i,j] = ind2sub(size(A),minix(1));
    
    % Merge clusters "i" and "j", but only if inter-cluster distance is
    % smaller than the radius.
    if minval(1)<radius
        
        % Merge pixel lists
        pixelList = union(visitList2{i},visitList2{j}); 
        visitList2{i} = pixelList;        
        
        % Merge cluster "peaks"
        clusterCenters(i,:) = sum(clusterCenters([i j],:),1)/2; 

        % Make new copy of clusterCenters and visitList2, but discard cluster "j"
        newClusterCenters = [];
        newVisitList = {};
        cluster = 1;
        for k = 1:numClusters
            if k~=j
                newClusterCenters(cluster,:) = clusterCenters(k,:);
                newVisitList{cluster} = visitList2{k};
                cluster = cluster + 1;
            end
        end
        clusterCenters = newClusterCenters;
        visitList2 = newVisitList;
        numClusters = numClusters - 1;
    else
        done = 1;
    end
end

% Generate cluster image and display tracks on top of it
clusterImage = zeros(size(X));
for i = 1:numClusters
    pixelList = visitList2{i};
    clusterImage(pixelList) = i;
end
figure(2)
subplot(1,3,1)
imshow(mat2gray(clusterImage))
hold on
for i = 1:numSeed
    rows = seedRowlist{i};
    cols = seedCollist{i};
    plot(cols,rows,'r')
    plot(cols(end),rows(end),'r.')
end
hold off
title('Cluster image')

% Generate "colorized" cluster image and display tracks on top of it
Ifilter = zeros(size(I));
for cluster = 1:numClusters
    subix = find(clusterImage==cluster);
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
subplot(1,3,2)
imshow(Ifilter)
hold on
for i = 1:numSeed
    rows = seedRowlist{i};
    cols = seedCollist{i};
    plot(cols,rows,'w')
    plot(cols(end),rows(end),'r.')
end
hold off
title('Colorized cluster image')

% Display cluster centers
subplot(1,3,3)
imshow(Ifilter)
hold on
plot(clusterCenters(:,4)*maxX,clusterCenters(:,5)*maxY,'w.')
hold off
title('Cluster centers/peaks')
