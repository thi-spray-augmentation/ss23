%% Testscript for Neural Style Transfer
% https://se.mathworks.com/help/images/neural-style-transfer-using-deep-learning.html

%% Step 1: load style and content image
%styleImage = im2double(imread("starryNight.jpg"));
styleImage = styleImageCrop;
%im2double(imread("C:\Users\jonas\OneDrive\03_MA_Studium\04_Semester_2\34_Projekt\04_Daten\spray\spray_wet\images\0a860de1-1652691506759178501.png"));
%styleImage = im2double(imread("C:\Users\jonas\OneDrive\03_MA_Studium\04_Semester_2\34_Projekt\02_Code\01_NeuralStyleTransferNetwork\360_F_163140610_XiCmvATyPNH1RuG26pGcVr5aeM1x82iW_cut.jpg"));

%contentImage = imread("lighthouse.png");
%contentImage = contentImageCrop;
currPath = "C:\Users\Jonas\OneDrive\03_MA_Studium\04_Semester_2\34_Projekt\06_GitHub\Trainings_Set_clear\03_1a947cc8-1652945066312072152.png";
%contentImage = imread("C:\Users\Jonas\OneDrive\03_MA_Studium\04_Semester_2\34_Projekt\06_GitHub\Trainings_Set_clear\03_1a947cc8-1652945066312072152.png");
%%contentImage = contentImageCrop;
contentImage = imread(currPath);

%%contentImage = imread("C:\Users\jonas\OneDrive\03_MA_Studium\04_Semester_2\34_Projekt\06_GitHub\Trainings_Set_clear\01_0a429d27-1652966769163486864.png");
%contentImage = imread("C:\Users\jonas\OneDrive\03_MA_Studium\04_Semester_2\34_Projekt\04_Daten\spray\spray_clear\images\0d1f3983-1652945066407985543.png");
% imshow(imtile({styleImage,contentImage},BackgroundColor="w"));

%% Load feature extraction network

% get a pretrained VGG-19 network
net = vgg19;

% To make the VGG-19 network suitable for feature extraction, remove all of the fully connected layers from the network.
lastFeatureLayerIdx = 38;
layers = net.Layers;
layers = layers(1:lastFeatureLayerIdx);

% The max pooling layers of the VGG-19 network cause a fading effect. 
% To decrease the fading effect and increase the gradient flow, 
% replace all max pooling layers with average pooling layers [1].
for l = 1:lastFeatureLayerIdx
    layer = layers(l);
    if isa(layer,"nnet.cnn.layer.MaxPooling2DLayer")
        layers(l) = averagePooling2dLayer( ...
            layer.PoolSize,Stride=layer.Stride,Name=layer.Name);
    end
end

% Create a layer graph with the modified layers.
lgraph = layerGraph(layers);

% Visualize the feature extraction network in a plot.
%plot(lgraph)
%title("Feature Extraction Network")

% To train the network with a custom training loop and enable automatic differentiation, convert the layer graph to a dlnetwork object.
dlnet = dlnetwork(lgraph);

%% Preprocess Data
% Resize the style image and content image to a smaller size for faster processing.
imageSize = [384,512];
styleImg = imresize(styleImage,imageSize);
contentImg = imresize(contentImage,imageSize);

% The pretrained VGG-19 network performs classification on a channel-wise mean subtracted image. 
% Get the channel-wise mean from the image input layer, which is the first layer in the network.
imgInputLayer = lgraph.Layers(1);
meanVggNet = imgInputLayer.Mean(1,1,:);

% The values of the channel-wise mean are appropriate for images of floating point data type with pixel values in the range [0, 255].
% Convert the style image and content image to data type single with range [0, 255]. 
% Then, subtract the channel-wise mean from the style image and content image.
styleImg = rescale(single(styleImg),0,255) - meanVggNet;
contentImg = rescale(single(contentImg),0,255) - meanVggNet;

%% Initialize Transfer Image
%{
The transfer image is the output image as a result of style transfer. You can initialize the transfer 
image with a style image, content image, or any random image. Initialization with a style image or content 
image biases the style transfer process and produces a transfer image more similar to the input image. 
In contrast, initialization with white noise removes the bias but takes longer to converge on the stylized image. 
For better stylization and faster convergence, this example initializes the output transfer image as a 
weighted combination of the content image and a white noise image.
%}
noiseRatio = 0.7;
randImage = randi([-20,20],[imageSize 3]);
transferImage = noiseRatio.*randImage + (1-noiseRatio).*contentImg;

%% Define Loss Functions and Style Transfer Parameters
% (Description s. https://se.mathworks.com/help/images/neural-style-transfer-using-deep-learning.html)

% Content Loss 
styleTransferOptions.contentFeatureLayerNames = "conv4_2";
styleTransferOptions.contentFeatureLayerWeights = 1;

% Style Loss
% Specify the names of the style feature extraction layers. 
% The features extracted from these layers are used to calculate style loss.
styleTransferOptions.styleFeatureLayerNames = ["conv1_1","conv2_1","conv3_1","conv4_1","conv5_1"];

% Specify the names of the style feature extraction layers. 
% Specify small weights for simple style images and increase the weights for complex style images.
styleTransferOptions.styleFeatureLayerWeights = [0.5,1.0,1.5,3.0,4.0];

% Total Loss
% The total loss is a weighted combination of content loss and style loss. 
% α and β are weight factors for content loss and style loss, respectively.
% Specify the weight factors alpha and beta for content loss and style loss. 
% The ratio of alpha to beta should be around 1e-3 or 1e-4 [1].
styleTransferOptions.alpha = 1; 
styleTransferOptions.beta = 1e3;

%% Specify Training Options

% Train for 2500 iterations
numIterations = 250;

%{
Specify options for Adam optimization. Set the learning rate to 2 for faster convergence. 
You can experiment with the learning rate by observing your output image and losses. 
Initialize the trailing average gradient and trailing average gradient-square decay rates with [].
%}
learningRate = 2;
trailingAvg = [];
trailingAvgSq = [];
% Information output
disp('Starting Training');
dstart = datetime(now,'ConvertFrom','datenum');
fprintf('Start time: %s\n', string(dstart));
%% Train the Network

% Convert the style image, content image, and transfer image to dlarray (Deep Learning Toolbox) objects 
% with underlying type single and dimension labels "SSC".
dlStyle = dlarray(styleImg,"SSC");
dlContent = dlarray(contentImg,"SSC");
dlTransfer = dlarray(transferImage,"SSC");

% Extract the content features from the content image.
numContentFeatureLayers = numel(styleTransferOptions.contentFeatureLayerNames);
contentFeatures = cell(1,numContentFeatureLayers);
[contentFeatures{:}] = forward(dlnet,dlContent,Outputs=styleTransferOptions.contentFeatureLayerNames);

% Extract the style features from the style image
numStyleFeatureLayers = numel(styleTransferOptions.styleFeatureLayerNames);
styleFeatures = cell(1,numStyleFeatureLayers);
[styleFeatures{:}] = forward(dlnet,dlStyle,Outputs=styleTransferOptions.styleFeatureLayerNames);

%{
Train the model using a custom training loop. For each iteration:

    Calculate the content loss and style loss using the features of the content image, 
    style image, and transfer image. To calculate the loss and gradients, use the helper 
    function imageGradients (defined in the Supporting Functions section of this example).

    Update the transfer image using the adamupdate (Deep Learning Toolbox) function.

    Select the best style transfer image as the final output image.
%}
figure

minimumLoss = inf;

for iteration = 1:numIterations
    disp(iteration);
    % Evaluate the transfer image gradients and state using dlfeval and the
    % imageGradients function listed at the end of the example
    [grad,losses] = dlfeval(@imageGradients,dlnet,dlTransfer, ...
        contentFeatures,styleFeatures,styleTransferOptions);
    [dlTransfer,trailingAvg,trailingAvgSq] = adamupdate( ...
        dlTransfer,grad,trailingAvg,trailingAvgSq,iteration,learningRate);
  
    if losses.totalLoss < minimumLoss
        minimumLoss = losses.totalLoss;
        dlOutput = dlTransfer;        
    end   
    
    % Display the transfer image on the first iteration and after every 50
    % iterations. The postprocessing steps are described in the "Postprocess
    % Transfer Image for Display" section of this example
    if mod(iteration,50) == 0 || (iteration == 1)
        
        transferImage = gather(extractdata(dlTransfer));
        transferImage = transferImage + meanVggNet;
        transferImage = uint8(transferImage);
        transferImage = imresize(transferImage,size(contentImage,[1 2]));
        
        image(transferImage)
        title(["Transfer Image After Iteration ",num2str(iteration)])
        axis off image
        drawnow
    end   
    
end
% print runtime information
dend = datetime(now,'ConvertFrom','datenum');
fprintf('End time: %s\n', string(dend));
fprintf('Duration: %s\n', string(dend-dstart));
disp('done');