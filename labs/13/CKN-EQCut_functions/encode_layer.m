function outmap = encode_layer(in,model)


   X=in;
   % contrast_normalization of input maps
   if model.centering
%       X=centering(X,nchannels);
   end
   if isfield(model,'median_contrast_normalization') && model.median_contrast_normalization
%       X=contrast_normalize_median(X);
   else
%       X=contrast_normalize(X);
   end

   % convolution with filters 
   outmap=model.Z' * X;
   % nonlinearities
   outmap=ones(size(model.Z,2),1)*sum(X.^2) + sum(model.Z.^2)' * ones(1,size(X,2))  -2*outmap;
   outmap=exp(-outmap/(model.sigma*model.sigma));
   outmap=diag(sqrt(model.w))*outmap;
%    outmap=bsxfun(@times,outmap,nrm);
   outmap=outmap';

% subsampling with Gaussian smoothing
% if model.subsampling > 1
%    outmap=subsampling(outmap,model.subsampling);
% end

% 
% 
% D=model.Z' * X;
% E=ones(100,1)*ones(1,3)*(X.*X);
% F=(ones(1,3)*(model.Z.*model.Z))'*ones(1,size(X,2));
% G=exp(-(E+F-2*D)./(model.sigma*model.sigma));
% K=((sqrt(model.w)*ones(1,149))).*G;
