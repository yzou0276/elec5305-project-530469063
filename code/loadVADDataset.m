function [features4D, labels] = loadVADDataset( ...
    speechDir, nospeechDir, sr, clipLenSec, nMels, hopLength, maxPerClass)
% label: 1 = speech, 0 = nospeech
speechADS   = audioDatastore(speechDir, 'IncludeSubfolders', true, 'FileExtensions','.wav');
nospeechADS = audioDatastore(nospeechDir,'IncludeSubfolders', true, 'FileExtensions','.wav');

if numel(speechADS.Files) > maxPerClass
    speechADS.Files = speechADS.Files(1:maxPerClass);
end
if numel(nospeechADS.Files) > maxPerClass
    nospeechADS.Files = nospeechADS.Files(1:maxPerClass);
end

numSpeech   = numel(speechADS.Files);
numNoSpeech = numel(nospeechADS.Files);
totalN      = numSpeech + numNoSpeech;

fprintf('Speech files: %d, Nospeech files: %d\n', numSpeech, numNoSpeech);

clipLenSamples = round(sr * clipLenSec);

% First, determine the feature size using the first speech sample.
[audioSample, fs0] = audioread(speechADS.Files{1});
if fs0 ~= sr
    audioSample = resample(audioSample, sr, fs0);
end
audioSample = mean(audioSample,2);
audioSample = fixLength(audioSample, clipLenSamples);

[featSample, ~] = extractPCENFeatures(audioSample, sr, nMels, hopLength);
[H, W] = size(featSample);

features4D = zeros(H, W, 1, totalN, 'single');
labels     = zeros(totalN, 1);  % 1=speech, 0=nospeech

idx = 1;

%%  Deal with speech 
reset(speechADS);
while hasdata(speechADS)

    [audioIn, info] = read(speechADS);
    fs = info.SampleRate;      % Take the sampling rate from info

    if fs ~= sr
        audioIn = resample(audioIn, sr, fs);
    end
    audioIn = mean(audioIn,2);
    audioIn = fixLength(audioIn, clipLenSamples);

    [feat, ~] = extractPCENFeatures(audioIn, sr, nMels, hopLength);
    features4D(:,:,1,idx) = single(feat);
    labels(idx) = 1;
    idx = idx + 1;
end

%% Deal with nospeech 
reset(nospeechADS);
while hasdata(nospeechADS)
    [audioIn, info] = read(nospeechADS);
    fs = info.SampleRate;

    if fs ~= sr
        audioIn = resample(audioIn, sr, fs);
    end
    audioIn = mean(audioIn,2);
    audioIn = fixLength(audioIn, clipLenSamples);

    [feat, ~] = extractPCENFeatures(audioIn, sr, nMels, hopLength);
    features4D(:,:,1,idx) = single(feat);
    labels(idx) = 0;
    idx = idx + 1;
end

end

%% Auxiliary function: fixed length
function xOut = fixLength(xIn, targetLen)
    L = numel(xIn);
    if L < targetLen
        xOut = [xIn; zeros(targetLen-L,1)];
    else
        xOut = xIn(1:targetLen);
    end
end
