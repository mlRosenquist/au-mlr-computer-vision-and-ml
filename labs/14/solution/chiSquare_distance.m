
function D = chiSquare_distance(X,Y)

[Dims,N] = size(X);
M = size(Y,2);

% calculate the chi-square distance matrix
D = zeros(N,M);
for ii=1:N
    for jj=1:M
        val = 0;
        for nn=1:Dims
            nume = (X(nn,ii) - Y(nn,jj))^2;
            deno = X(nn,ii) + Y(nn,jj) + eps;
            val = val + (nume/deno);
        end
        D(ii,jj) = val;        
    end
end

% % another way
% Ytemp = Y;
% Y = X;  X = Ytemp;  clear Ytemp;
% M = size(X,2);  N = size(Y,2);
% mOnes = ones(1,M); D = zeros(M,N);
% for i=1:N
%   yi = Y(:,i);      yiRep = yi(:,mOnes);
%   s = yiRep + X;    d = yiRep - X;
%   D(i,:) = sum( d.^2 ./ (s+eps), 1 );
% end