function Ieq = intensityeq( I, p )
   
    HSV = rgb2hsv(I);

    V = HSV(:,:,3);
    V = histeq(V);

    HSV(:,:,3) = V;
    Ieq = hsv2rgb(HSV);

end