

%Stacked Autoencoders TUTORIAL MATHWORKS


%% MODEL COMPARISON - with new test data
testFinal = csvread("mnist_test.csv",1,0);

% preparing dataset:
testF_label  = testFinal(:,1);
testF_label(testF_label == 0) = 10;
testF_label_dummy = dummyvar(testF_label);
inputsF = testFinal(:,2:end);  

% Transpose of everything:
inputsF = inputsF';             
testF_label = testF_label';                 
testF_label_dummy = testF_label_dummy'; 

%reshape train/test matrices into the original rectangular form of images (28x28
%matrices) with reshape

trainImages= cell(1, 31500);   
for i= 1:31500
    rect = reshape(tX(:, i), [28,28]);
    trainImages{i}= rect;
end

testImagesFinal= cell(1, 9999);   
for i= 1:9999
    rect = reshape(inputsF(:, i), [28,28]);
    testImagesFinal{i}= rect;
end

% Display some of the training images
clf
for i = 1:20
    subplot(4,5,i);
    imshow(testImagesFinal{i});
end    

%% train first encoder
%set the random number generator seed (to avoid different random weights every time the model is trained)
rng('default')

%size of the hidden layer set to 100 (should be smaller than the input size)
hiddenSize1 = 150;

%train the autoencoder, specifying the values for the regularizers
autoenc1 = trainAutoencoder(trainImages,hiddenSize1, ...
    'MaxEpochs',50);

%view autoencoder structure, where The encoder maps an input to a hidden representation, 
%and the decoder attempts to reverse this mapping to reconstruct the original input.
view(autoenc1)

%Visualizing the weights of the first autoencoder
%10x10 matrix (if the hiddensize of the encoder is 100), where each cell 
%represents a 'visual feature' capable of recognizing the form of digit
%data. It's a compressed version of the original training data
figure()
plotWeights(autoenc1);

%Returns the encoded data, feat1, for the input data using the autoencoder, autoenc1.
feat1 = encode(autoenc1,trainImages);

%possible new section:
%Reconstruct Handwritten Digit Images Using Sparse Autoencoder
XReconstructed= predict(autoenc1, testImages);

figure;
for i = 1:20
    subplot(4,5, i); 
    imshow(testImages{i});
end

figure;
for i = 1:20
    subplot(4,5, i); 
    imshow(XReconstructed{i});
end

%% train second encoder
%two differences: 
%use the features that were generated from the first autoencoder as the training data 
%decrease the size of the hidden representation  

hiddenSize2 = 50;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',200);

%watch autoencoder structure and weights
view(autoenc2)
figure()
plotWeights(autoenc2);


%extract second set of features from feature1 and second autoencoder
%from 784 dimensions in training data, to 100 features (1 encoder) to 50
%features (second encoder)
feat2 = encode(autoenc2,feat1);

%% Training the softmax layer
%Unlike the autoencoders, you train the softmax layer in a supervised fashion using labels for the training data
softnet = trainSoftmaxLayer(feat2,tY,'MaxEpochs',300);
view(softnet)
%put the three pieces together
deepnet = stack(autoenc1,autoenc2,softnet);
view(deepnet)
        
%% TEST SET
% add to the first file this variable
%Y_test_dummy = t_label_dummy(:, validation_set.test);

%visualize the results with a confusion matrix
y = deepnet(inputsF);
plotconfusion(testF_label_dummy,y);

%% Fine tuning the deep neural network
%fine tuning= performing backpropagation on the whole multilayer network
     
% Perform fine tuning
deepnet = train(deepnet,tX,tY);

%view results in a confusion matrix
y = deepnet(inputsF);
plotconfusion(testF_label_dummy,y);



