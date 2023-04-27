% Crop the interesting area from the image based on the bounding box


% imcrop script
contentImage = imread("C:\Users\jonas\OneDrive\03_MA_Studium\04_Semester_2\34_Projekt\04_Daten\spray\spray_clear\images\0d1f3983-1652945066407985543.png");
styleImage = imread("C:\Users\jonas\OneDrive\03_MA_Studium\04_Semester_2\34_Projekt\04_Daten\spray\spray_wet\images\1cd7100d-1652343905410987856.png");


% read image dimensions
sizeOffset = 50;
szx = size(contentImage, 2);
szy = size(contentImage, 1);

cTxt = dlmread("C:\Users\jonas\OneDrive\03_MA_Studium\04_Semester_2\34_Projekt\04_Daten\spray\spray_clear\labels\0d1f3983-1652945066407985543.txt");
sTxt = dlmread("C:\Users\jonas\OneDrive\03_MA_Studium\04_Semester_2\34_Projekt\04_Daten\spray\spray_wet\labels\1cd7100d-1652343905410987856.txt");

% get corner points
xmin = (cTxt(2) - (cTxt(4)/2))*szx - (sizeOffset/2);
ymin = (cTxt(3) - (cTxt(5)/2))*szy - (sizeOffset/2);
xmax = (cTxt(2) + (cTxt(4)/2))*szx + (sizeOffset/2);
ymax = (cTxt(3) + (cTxt(5)/2))*szy + (sizeOffset/2);

topleft     =   [xmin ymin];   % x y
topright    =   [xmax ymin];
bottomleft  =   [xmin ymax];
bottomright =   [xmax ymax];

figure(2);
imshow(styleImage);
hold on
plot(topleft(1),           topleft(2),  '*', "color", 'red');


%  [xmin ymin width height]
contentImageCrop = imcrop(contentImage, ...
    [ xmin ...    xmin = left
    ymin ...        ymin = top
    cTxt(4)*szx + sizeOffset ...                         width
    cTxt(5)*szy + sizeOffset ]);                         % heigth
figure(1);
imshow(contentImageCrop);
hold on
plot(topleft(1),           topleft(2),  '*', "color", 'red');
%plot( [topleft, topright, bottomleft, bottomright, topleft],...
%      [topleft, topright, bottomleft, bottomright, topleft], ...
%      "LineWidth", 2, "Color", "red");

%plot([x1(i), x3(i), x2(i), x4(i), x1(i)], [y1(i), y3(i), y2(i), y4(i), y1(i)], "LineWidth", 2, "color", 'red')
