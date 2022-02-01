% You should put this function in a separate file create_filter.m
function f = create_filter(dim)

f1 = zeros(dim);
% dim should be odd
f1(:,ceil(dim/2)) = 1;

f(:,:,1) = f1; % vertical
f(:,:,2) = eye(dim); % first diagonal
f(:,:,3) = f(end:-1:1,:,2); % second diagonal;
f(:,:,4) = imrotate(f1,90);
f = f/dim;