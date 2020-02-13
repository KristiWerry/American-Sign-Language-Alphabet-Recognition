function [acc, confMat] = HOG_SVM(imds, imdsTest)
%HOG_SVM uses extractHOGFeatures to train an SVM to produce
%accuracy.
%   imds is the training dataset where the imdsTest is the test
%   dataset. This function will use the function extractHOGFeatures
%   with a predetermined cell size to find the training features on
%   grayscale images. Then we will use Gaussian and One vs All methods
%   to train the SVM with the function fitcecoc. 
%   The function HelpExtractFeatures will extract the HOG features
%   from the test set. We will then use the function predict to find 
%   the accuracy of how well the training set could classify the test set. 
%   This will also produce a confusion matrix to show which classes where
%   classified as which class.

% Loop over the trainingSet and extract HOG features from each image. A
% similar procedure will be used to extract features from the testSet.

img = readimage(imds, 206); %Reading a random image to get the size
img = rgb2gray(img);

% Extract HOG features and HOG visualization
[hog_32x32, vis] = extractHOGFeatures(img,'CellSize',[32 32]);
cellSize = [32 32];
hogFeatureSize = length(hog_32x32);

%% 
numImages = numel(imds.Files);
trainingFeatures = zeros(numImages, hogFeatureSize, 'single');

for i = 1:numImages
    img = readimage(imds, i);
    %Convert the images to grayscale
    img = rgb2gray(img);
    %Put the HOG features in the trainingFeature set
    trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);  
end

% Get labels for each image.
trainingLabels = imds.Labels;
%%
%Train the SVM using the extracted features and paired labels
%We use a Gaussian filter and One vs All approach
t = templateSVM('KernelFunction', 'gaussian');
classifier = fitcecoc(trainingFeatures, trainingLabels, 'Coding', 'onevsall', 'Learner', t);

%%
% Extract HOG features from the test set similar to the training set extraction.
[testFeats, testLabels] = helperExtractHOGFeaturesFromImageSet(imdsTest, hogFeatureSize, cellSize);
% Get class predictions using the test features.
predictedLabels = predict(classifier, testFeats);
% Find the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);
%Get the accuracy by averaging the diagonal of the confusion matrix
acc = sum(diag(confMat))/(numel(imdsTest.Files));

end

function [features, setLabels] = helperExtractHOGFeaturesFromImageSet(imds, hogFeatureSize, cellSize)
    % Extract HOG features from an imageDatastore.
    setLabels = imds.Labels;
    numImages = numel(imds.Files);
    features  = zeros(numImages, hogFeatureSize, 'single');

    % Process each image and extract features
    for j = 1:numImages
        img = readimage(imds, j);
        img = rgb2gray(img);
        features(j, :) = extractHOGFeatures(img,'CellSize',cellSize);
    end
end

