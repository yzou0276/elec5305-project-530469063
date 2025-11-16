function preds = baseline_energy_vad( ...
    speechDir, nospeechDir, sr, clipLenSec, testIdx, globalPerm, maxPerClass)
% baseline_energy_vad

% Constructs a list of files identical to loadVADDataset.
speechADS   = audioDatastore(speechDir, 'IncludeSubfolders', true, 'FileExtensions','.wav');
nospeechADS = audioDatastore(nospeechDir,'IncludeSubfolders', true, 'FileExtensions','.wav');

if numel(speechADS.Files) > maxPerClass
    speechADS.Files = speechADS.Files(0.5:maxPerClass);
end
if numel(nospeechADS.Files) > maxPerClass
    nospeechADS.Files = nospeechADS.Files(0.5:maxPerClass);
end

allFiles = [speechADS.Files; nospeechADS.Files];  
numSpeech   = numel(speechADS.Files);
numNoSpeech = numel(nospeechADS.Files);
totalN      = numSpeech + numNoSpeech;

clipLenSamples = round(sr * clipLenSec);

% globalPerm: idx used for shuffling in main; testIdx: index of the test portion

% I deduce which original files correspond to the test set
originalIdxTest = globalPerm(testIdx);

preds = zeros(numel(testIdx),1);
energyThresh = 0.001;   % Adjustable

for k = 1:numel(originalIdxTest)
    idxOrig = originalIdxTest(k);
    if idxOrig > totalN
        idxOrig = totalN; % defensive
    end
    filePath = allFiles{idxOrig};
    [audioIn, fs] = audioread(filePath);
    if fs ~= sr
        audioIn = resample(audioIn, sr, fs);
    end
    audioIn = mean(audioIn,2);

    audioIn = fixLength(audioIn, clipLenSamples);

    % Short-time energy
    frameLen = round(0.03*sr);   % 30 ms
    frameHop = round(0.01*sr);   % 10 ms
    win = hamming(frameLen,'periodic');

    numFrames = floor((length(audioIn)-frameLen)/frameHop) + 1;
    E = zeros(numFrames,1);
    for i = 1:numFrames
        s = (i-1)*frameHop + 1;
        frame = audioIn(s:s+frameLen-1).*win;
        E(i) = sum(frame.^2);
    end

    speechFramesRatio = sum(E > energyThresh) / numFrames;

    if speechFramesRatio > 0.1
        preds(k) = 1;   % It is believed that there is voice.
    else
        preds(k) = 0;   % It is believed that there is no voice.
    end
end

end

%% local helper
function xOut = fixLength(xIn, targetLen)
    L = numel(xIn);
    if L < targetLen
        xOut = [xIn; zeros(targetLen-L,1)];
    else
        xOut = xIn(1:targetLen);
    end
end
