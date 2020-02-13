%run this script to run each testing model and copare the 
%accuracies

%Note: Please make sure the folder name and file path is correct.
%first get the dataset with the folder name as the image's class
trainImagePath = fullfile('asl_alphabet_train');
imds = imageDatastore(trainImagePath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
%let imds be the training set with 2800 images per category 
%and imdsTest is the test set with the rest of the 200 images
%per category
trainingNumber = 2800;
[imds, imdsTest] = splitEachLabel(imds, trainingNumber, 'randomize');
countEachLabel(imds) %shows how many images are in each label

% Note: To test a different number of images, uncomment this
% NumToTest = 200;
% imdsTest = imageDatastore(trainImagePath, ...
%     'IncludeSubfolders',true, ...
%     'LabelSource','foldernames');
% imdsTest = splitEachLabel(imdsTest, NumToTest, 'randomize');

%initialize the accuracies
HOGacc = 0;
SURFacc = 0;
CNNacc = 0;

%WARNING: Each of the next steps will take a very long time to run.
%% 
%Extracting HOG features over the entire image then training using 
%fitcecoc
%Note: To get the same results for the HOG SVM as in our report, imds 
%should run with 1000 images per category and imdsTest should run with 
%200 test images per category.

[HOGacc, HOGconfMat] = HOG_SVM(imds, imdsTest);

%%
%Extracting SURF features using bag of features and then training using
%trainImageCategoryClassifier

[SURFacc, SURFconfMat] = SURF_SVM(imds, imdsTest);

%%
%Training a custom Convolutional Neural Network to classify images
%Note: This uses parrallel computing
CNNacc = custom_CNN(imds, imdsTest);

%%
%present the data
displayConfusionMatrix(HOGconfMat);
displayConfusionMatrix(SURFconfMat);
%Compare the accuracies
names = {'HOGAccuracy', 'SURFAccuracy', 'CNNAccuracy'};
T = table(HOGacc, SURFacc, CNNacc, 'VariableNames', names)

%%
%This function will show the confusion matrix
function displayConfusionMatrix(confMat)
% Display the confusion matrix in a formatted table.

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));
%d is for DELETE
%n for NOTHING
%s for SPACE
letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZdns';
colHeadings = arrayfun(@(x)sprintf('%c',x),letters,'UniformOutput',false);
format = repmat('%-9s',1,30);
header = sprintf(format,'Letter  |',colHeadings{:});
fprintf('\n%s\n%s\n',header,repmat('-',size(header)));
for idx = 1:numel(letters)
    fprintf('%-9s',   [letters(idx) '      |']);
    fprintf('%-9.2f', confMat(idx,:));
    fprintf('\n')
end
end


