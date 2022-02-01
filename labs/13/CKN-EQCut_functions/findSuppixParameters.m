function [LabelLine,suppixel,max_label,boundaries,LMean,AMean,BMean]=findSuppixParameters(image_now,m,suppix_num)

%Define parameters for SLIC
height=size(image_now,2);
width=size(image_now,1);
PixNum=height*width;
k=PixNum/(suppix_num.^2);
ImgAttr=[ height ,width, k, m, PixNum ];
[ LabelLine, LMean, AMean, BMean, k ] = SLIC( image_now(:,:,1), image_now(:,:,2), image_now(:,:,3), ImgAttr );

suppixel=reshape(LabelLine,width,height); %Reshape suppixels to image
suppixel=suppixel+1; %Labeling starts from 1 for convenience
max_label=max(max(suppixel)); %Number of Superpixels

%Boundary superpixels
boundaries=[suppixel(:,1)' suppixel(1,:) suppixel(:,end)' suppixel(end,:)];
boundaries=unique(boundaries);
SCORE=0;
% DATAMAT=[LMean AMean BMean];
% [COEFF,SCORE] = princomp(DATAMAT);
% LMean=SCORE(:,1);
% AMean=SCORE(:,2);
% BMean=SCORE(:,3);