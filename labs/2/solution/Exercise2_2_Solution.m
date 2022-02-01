clear all; close all;

VLFEATROOT = 'C:\Program Files\MATLAB\R2020b\toolbox\vlfeat-0.9.20';
VLFEATROOT = 'vlfeat-0.9.20';

addpath('C:\Program Files\MATLAB\R2020b\toolbox');
run('C:\Program Files\MATLAB\R2020b\toolbox\vlfeat-0.9.20\toolbox\vl_setup.m');

% Load image and convert to grayscale
I = imread('RSA.jpg');

% Construct scale space (Gaussian pyramid)
[frames, descrp,info] = vl_covdet(single(I),'OctaveResolution',3);
gss = info.gss;
vl_plotss(gss);

octaves = gss.lastOctave-gss.firstOctave+1;
gssSubdivs = gss.octaveLastSubdivision-gss.octaveFirstSubdivision+1;

% Show Difference of Gaussians (DoGs)
figure
DoGs = info.css;
vl_plotss(DoGs);

% Find local extrema
DoGsSubdivs = DoGs.octaveLastSubdivision-DoGs.octaveFirstSubdivision+1;
Maxima = {};
Minima = {};
MaximaNonEdges = {};
MinimaNonEdges = {};
HarrisThreshold = 0.03;
for o=1:octaves
    [H,W,~] = size(DoGs.data{o});
    for s=2:DoGsSubdivs-1
        fprintf('Calculating local max/min of octave %d/%d, subdivision %d/%d\n',o,octaves,s-1,DoGsSubdivs-2)
        data = DoGs.data{o}(:,:,s-1:s+1);
        localMax = imregionalmax(data);
        localMax = localMax(:,:,2); % Remove lower and upper subdivisions
        localMin = imregionalmin(data);
        localMin = localMin(:,:,2); % Remove lower and upper subdivisions
        
        dataMid = data(:,:,2);
        % Discard local maxima less than 10% of the global maximum
        Ithr = max([0 0.15*max(dataMid(localMax))]);
        localMax(dataMid < Ithr) = 0;
        % Discard local minima larger than 20% of the global minimum
        Ithr = min([0 0.15*min(dataMid(localMin))]);
        localMin(dataMid > Ithr) = 0;
        
        Maxima{o,s-1} = localMax;
        Minima{o,s-1} = localMin;
        
        % Remove edge responses
        [Ix,Iy] = gradient(DoGs.data{o}(:,:,s));
        Ixx = imfilter(Ix.^2,fspecial('gaussian',3,0.5));
        Iyy = imfilter(Iy.^2,fspecial('gaussian',3,0.5));
        Ixy = imfilter(Ix.*Iy,fspecial('gaussian',3,0.5));
        
        % Exercise: modify localMax and localMin by removing blobs whose
        % Harris measures are below 'HarrisThreshold'
        
        % Hint: get a list all found blobs: find(localMax == 1)
        % Hint: remove blob with command: localMax(index) = 0
        
        indsMax = find(localMax == 1);
        indsMin = find(localMin == 1);
        % Combine indices to loop through both local maxima and minima in a
        % single loop. Could also be done in two separate loops.
        inds = [indsMax;indsMin];
        for i=1:length(inds)
            M = [ Ixx(inds(i)) Ixy(inds(i))
                Ixy(inds(i)) Iyy(inds(i)) ];
            d = eig(M);
            lambda_max=max(d);
            lambda_min=min(d);
            
            R = lambda_max*lambda_min/(lambda_max+lambda_min);
            
            if R<HarrisThreshold
                if ismember(inds(i),indsMax)
                    localMax(inds(i)) = 0;
                elseif ismember(inds(i),indsMin)
                    localMin(inds(i)) = 0;
                end
            end
        end
        
        % The modified localMax and localMin are stored for each scale
        MaximaNonEdges{o,s-1} = localMax;
        MinimaNonEdges{o,s-1} = localMin;
    end
end

%% Show a single scale
octave = 2;
subdiv = 1;
% octave=1, subdiv=3 contains two remaining markers
figure
subplot(121);imagesc(gss.data{octave}(:,:,subdiv));title('Gaussian');hold on;
showBlobs(Maxima{octave,subdiv},'rx');
showBlobs(Minima{octave,subdiv},'bx');
subplot(122);imagesc(DoGs.data{octave}(:,:,subdiv));title('Difference of Gaussians');hold on;
showBlobs(Maxima{octave,subdiv},'rx');
showBlobs(Minima{octave,subdiv},'bx');

figure
subplot(121);imagesc(DoGs.data{octave}(:,:,subdiv));title('All blobs');hold on;
showBlobs(Maxima{octave,subdiv},'rx');
showBlobs(Minima{octave,subdiv},'bx');
subplot(122);imagesc(DoGs.data{octave}(:,:,subdiv));title('Edges removed');hold on;
showBlobs(MaximaNonEdges{octave,subdiv},'rx');
showBlobs(MinimaNonEdges{octave,subdiv},'bx');