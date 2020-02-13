function [acc, confMat] = SURF_SVM(imds, imdsTest)
%SURF_SVM uses bagOfFeatures to train an SVM to produce
%accuracy.
%   imds is the training dataset where the imdsTest is the test
%   dataset. This function will use the function bagOfFeatures
%   to find the SURF training features on grayscale images. 
%   Then we will use Gaussian and Detector methods to train the
%   SVM with the function trainImageCategoryClassifier.


%Find the SURF
bag = bagOfFeatures(imds, 'PointSelection', 'Detector');
%customize the SVM to use Gaussian
t = templateSVM('KernelFunction', 'gaussian');
%Train the images using the bag of features and Gaussian
categoryClassifier = trainImageCategoryClassifier(imds, bag, 'LearnerOptions', t);

%%
%Find the accuracy of the test set
confMat = evaluate(categoryClassifier, imdsTest);
acc = mean(diag(confMat));
%%
%Show the predicted labels of the test set
[labelIdx, scores] = predict(categoryClassifier, imdsTest);
categoryClassifier.Labels(labelIdx);


end

