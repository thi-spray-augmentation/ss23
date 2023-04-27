close all

f = figure();
f.Position(3:4) = [700 400];
subplot(2,2,[1,2]), imshow(transferImage); ...
    title(['transferImage after ', num2str(numIterations),' iterations']);
subplot(2,2,3), imshow(contentImage); title('contentImage'); ...
    xlabel('ID: 03_1a947cc8-1652945066312072152.png')
subplot(2,2,4), imshow(styleImage); title('styleImage'); ...
    xlabel('ID: 0a860de1-1652691506759178501.png')

str2 = ' 03_1a947cc8-1652945066312072152';

% automatically read current image name
currDir = 'C:\Users\Jonas\OneDrive\03_MA_Studium\04_Semester_2\34_Projekt\06_GitHub\Trainings_Set_clear\';
currImgName = strrep(currPath, currDir, ' ');
currImgName = strrep(currImgName, '.png', '');

% create folder for saving
currDate = [strrep(datestr(datetime), ':', '_'), currImgName];
mkdir('Results', currDate)

