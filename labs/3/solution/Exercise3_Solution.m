clear all; close all;

I = imread('Edison.jpg');
I = imresize(I,0.5);

% Expand image to visualization purposes
I = [zeros(size(I,1),150,3),I,zeros(size(I,1),300,3)];

% Inline function for drawing a line
x = [1,size(I,2)]; % x-coordinates
y = @(l) (-l(3)-l(1)*x)/l(2); % y-coordinates

% Homogeneous corner points of the building
p1 = [196 220 1];
p2 = [321 137 1];
p3 = [197 314 1];
p4 = [321 303 1];
p5 = [384 114 1];
p6 = [501 181 1];
p7 = [384 305 1];
p8 = [501 311 1];
  
figure(1)
imshow(I);
hold on;
plot(p1(1),p1(2),'r*',p2(1),p2(2),'r*',p3(1),p3(2),'b*',p4(1),p4(2),'b*',...
     p5(1),p5(2),'g*',p6(1),p6(2),'g*',p7(1),p7(2),'c*',p8(1),p8(2),'c*')

% Labels corresponding to the 8 corner points, p1-p8
labels = cellstr( num2str([1:8]') );
points = [p1;p2;p3;p4;p5;p6;p7;p8];
text(points(:,1), points(:,2), labels, 'VerticalAlignment','bottom', ...
                             'HorizontalAlignment','right', 'FontSize', 14)
                         
% Step 1: compute the lines l1-l4 between the pairwise points.
% l1=[a,b,c] is the line between p1 and p2 satisfying the equation
% ax+by+c=0
% To plot a line, l1, use: plot(x,y(l1))
l1 = cross(p1,p2)
l2 = cross(p3,p4)

% Plot the lines
plot(x,y(l1),'b')
plot(x,y(l2),'b')



% Step 2: compute the first vanishing points, pv1, as the intersection
% between the two lines, l1 and l2
% To plot a point, p1, use: plot(p1(1),p1(2),'x')

% Compute a vanishing point 
pv1 = cross(l1,l2)

% Divide by homogeneous coordinate
pv1 = pv1./pv1(3)

% Plot point
plot(pv1(1),pv1(2),'yx')


% Step 3: repete step 1 and 2 for points p5-p8
l3 = cross(p5,p6)
l4 = cross(p7,p8)
pv2 = cross(l3,l4)
pv2 = pv2./pv2(3)

plot(x,y(l3),'r')
plot(x,y(l4),'r')
plot(pv2(1),pv2(2),'yx')


% Step 4: compute and visualize the horizon
lhorizon = cross(pv1,pv2);
plot(x,y(lhorizon),'g')


% Exercise: compute the height of the camera above the ground
disp('Point out the top of the measured wall using a single mouse click')
rect=getrect;pGroundTop=[rect(1);rect(2)];
disp('Point out the bottom of the measured wall using a single mouse click')
rect=getrect;pGroundBot=[rect(1);rect(2)];

% Compute the height of the wall in pixels
distWallPixels = pGroundBot(2)-pGroundTop(2);

% Compute distance between ground point and horizon in pixels
y = @(l,x) (-l(3)-l(1)*x)/l(2); % y-coordinates
distHorizonPixels = abs(pGroundBot(2)-y(lhorizon,pGroundBot(1)));

% Compute the camera height (=the height of the horizon above ground level)
cameraHeight = distHorizonPixels/distWallPixels*3.2




