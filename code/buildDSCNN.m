function layers = buildDSCNN(inputSize, numClasses)

% inputSize: [H W C]

layers = [
    imageInputLayer(inputSize,'Normalization','none','Name','input')

    convolution2dLayer(3, 32, 'Padding','same','Stride',1,'Name','conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    maxPooling2dLayer(2,'Stride',2,'Name','pool1')

    % depthwise conv
    convolution2dLayer(3, 32, 'Padding','same','Stride',1, ...
        'Name','depthwise','NumChannels',32,'DilationFactor',1,'BiasLearnRateFactor',0)
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')

    % pointwise conv
    convolution2dLayer(1, 64, 'Padding','same','Stride',1,'Name','pointwise')
    batchNormalizationLayer('Name','bn3')
    reluLayer('Name','relu3')
    maxPooling2dLayer(2,'Stride',2,'Name','pool2')

    globalAveragePooling2dLayer('Name','gap')
    fullyConnectedLayer(numClasses,'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','output')
    ];
end
