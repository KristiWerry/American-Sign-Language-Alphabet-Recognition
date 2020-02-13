function acc = custom_CNN(imds, imdsTest)
%custom_CNN uses a custom made Convolutional Neural Network to train
%a training set and produce an accuracy based on a test set
%   imds is the training dataset where the imdsTest is the test
%   dataset. This function creates a custom convolutional neural network
%   to predict the a classification for the images in the test set. 
%   We rotated and translate the images to make the network more robust. 

%Rotated and Translate the images for more robustness
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3]);
imds = augmentedImageDatastore([200 200 3],imds,...
    'DataAugmentation',imageAugmenter);

%determine the layers for the network
layers = [ ...
    imageInputLayer([200 200 3])
    convolution2dLayer(4,4,'Padding',0)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(4,8,'Padding',0)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(4,16,'Padding',0)
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(29)
    softmaxLayer
    classificationLayer];

%determine the options for the network
%change 'ExecutionEnvironment' to 'cpu' if you do not want parrallel computing
options = trainingOptions('sgdm', ...
    'MaxEpochs',20,...
    'Shuffle','every-epoch', ...
    'InitialLearnRate',1e-3, ...
    'MiniBatchSize',64, ...
    'Verbose',false, ...
    'ExecutionEnvironment','parallel', ... 
    'Plots','training-progress'); 

%%
%Train the network
net = trainNetwork(imds,layers,options);
savedNet = net;
%save savedNet

%%
%load savedNet
%Predict the test set
YPred = classify(net,imdsTest);
YTest = imdsTest.Labels;
%Find the accuracy
acc = sum(YPred == YTest)/numel(YTest);

end

