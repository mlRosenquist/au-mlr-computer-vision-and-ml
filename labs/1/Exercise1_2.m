clear all
close all
[orig] = imread('cameraman.tif');
I = 10*randn(size(orig)) + double(orig);
I1 = [orig I];
figure(1)
imshow(I1)
f3 = create_filter(3);
f5 = create_filter(5);
f7 = create_filter(7);

for i = 1:4
    fI3(:,:,i) = ???
end
for i = 1:4
    fI5(:,:,i) = ???
end
for i = 1:4
    fI7(:,:,i) = ???
end
figure, title('With Kernel size 3')

for i = 1:4
    subplot(2,2,i), imshow(fI3(:,:,i),[]);
end
figure, title('With Kernel size 5')

for i = 1:4
    subplot(2,2,i), imshow(fI5(:,:,i),[]);
end
figure, title('With Kernel size 7')

for i = 1:4
    subplot(2,2,i), imshow(fI7(:,:,i),[]);
end
%%
bestfI3 = zeros(size(orig,1));
bestfI5 = zeros(size(orig,1));
bestfI7 = zeros(size(orig,1));
for i = 1:4
    bestfI3 = bestfI3  + fI3(:,:,i);
    bestfI5 = bestfI5  + fI5(:,:,i);
    bestfI7 = bestfI7  + fI7(:,:,i);
end
%bestfI3 = sum(fI3,3);

figure, 
subplot(2,2,1), imshow(I,[]); title('Noisy image.')
subplot(2,2,2), imshow(bestfI3,[]); title('Restored image with kernel = 3.')
subplot(2,2,3), imshow(bestfI5,[]);title('Restored image with kernel = 5.')
subplot(2,2,4), imshow(bestfI7,[]);title('Restored image with kernel = 7.')
