clear; clc; close all;

%% Global config
sr          = 16000;        % sampling rate
clipLenSec  = 0.5;          % Each sample is 0.5 second long.
nMels       = 40;
hopLength   = round(0.01*sr);   % 10 ms
batchSize   = 64;
maxPerClass = 600;          % A maximum of 600 entries can be selected from each category.
valRatio    = 0.15;
testRatio   = 0.3;
numEpochs   = 40;
lr          = 1e-3;

dataDirSpeech   = fullfile(pwd,'speech/');
dataDirNoSpeech = fullfile(pwd,'nospeech/');

%% 1. Loading & Building VAD Datasets
fprintf('Loading dataset and extracting PCEN features...\n');

[features, labels] = loadVADDataset( ...
    dataDirSpeech, dataDirNoSpeech, ...
    sr, clipLenSec, nMels, hopLength, maxPerClass);

% features:  H x W x 1 x N （H=nMels, W=time frames）
% labels:    N x 1  (0=no-speech, 1=speech)

numSamples = numel(labels);
idx = randperm(numSamples);
features = features(:,:,:,idx);
labels   = labels(idx);

%% 2. Train / Val / Test
nTest = round(testRatio * numSamples);
nVal  = round(valRatio  * numSamples);
nTrain = numSamples - nVal - nTest;

trainIdx = 1:nTrain;
valIdx   = nTrain+1 : nTrain+nVal;
testIdx  = nTrain+nVal+1 : numSamples;

XTrain = features(:,:,:,trainIdx);
YTrain = categorical(labels(trainIdx), [0 1], {'nospeech','speech'});

XVal   = features(:,:,:,valIdx);
YVal   = categorical(labels(valIdx), [0 1], {'nospeech','speech'});

XTest  = features(:,:,:,testIdx);
YTest  = categorical(labels(testIdx), [0 1], {'nospeech','speech'});

fprintf('Train: %d, Val: %d, Test: %d\n', numel(YTrain), numel(YVal), numel(YTest));

%% 3. Building a DS-CNN model
inputSize = size(XTrain,[1 2 3]);   % [nMels, T, 1]
numClasses = 2;

layers = buildDSCNN(inputSize, numClasses);

miniBatchSize = batchSize;
valFreq = floor(numel(YTrain)/miniBatchSize);

options = trainingOptions('adam', ...
    'InitialLearnRate', lr, ...
    'MaxEpochs', numEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',true, ...
    'ValidationData',{XVal, YVal}, ...
    'ValidationFrequency', valFreq, ...
    'ExecutionEnvironment','auto');

%% 4. Train
fprintf('Training DS-CNN model...\n');
net = trainNetwork(XTrain, YTrain, layers, options);

%% 5. Evaluate on the test set
fprintf('Evaluating on test set...\n');
[YPred, scores] = classify(net, XTest);
[acc, precision, recall, f1, cm] = computeMetrics(YTest, YPred);

disp('DS-CNN VAD Test Performance');
fprintf('Accuracy : %.4f\n', acc);
fprintf('Precision: %.4f\n', precision);
fprintf('Recall   : %.4f\n', recall);
fprintf('F1-score : %.4f\n', f1);

figure;
confusionchart(YTest, YPred);
xlabel("Predicted Class");
ylabel("True Class");
title('DS-CNN VAD Confusion Matrix');

%% 6. Simple baseline: Energy threshold VAD (from the same batch of test samples)
fprintf('Evaluating simple energy-based baseline...\n');
[baselinePred] = baseline_energy_vad( ...
    dataDirSpeech, dataDirNoSpeech, ...
    sr, clipLenSec, testIdx, idx, maxPerClass);

YTestNum = double(YTest=='speech');  % 1 or 0
baselinePredCat = categorical(baselinePred, [0 1], {'nospeech','speech'});

[accB, precB, recB, f1B, cmB] = computeMetrics( ...
    categorical(YTestNum,[0 1],{'nospeech','speech'}), baselinePredCat);

disp('Energy-based Baseline Test Performance');
fprintf('Accuracy : %.4f\n', accB);
fprintf('Precision: %.4f\n', precB);
fprintf('Recall   : %.4f\n', recB);
fprintf('F1-score : %.4f\n', f1B);

figure;
confusionchart(categorical(YTestNum,[0 1],{'nospeech','speech'}), baselinePredCat);
xlabel("Predicted Class");
ylabel("True Class");
title('Energy-based Baseline Confusion Matrix');
disp('YTest label');
tabulate(YTest)   
