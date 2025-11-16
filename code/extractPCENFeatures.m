function [feat, t] = extractPCENFeatures(audio, fs, nMels, hopLength)
% Extracting log-Mel features
% audio: column vector

% 1. Calculate the Mel spectrum
melSpec = melSpectrogram(audio, fs, ...
    'Window', hann(2*hopLength,'periodic'), ...
    'OverlapLength', hopLength, ...
    'FFTLength', 2*hopLength, ...
    'NumBands', nMels);

% 2. Convert to logarithmic scale (to avoid log(0))
feat = log10(melSpec + 1e-6);

% 3. Manually construct the timeline t
numFrames = size(melSpec, 2);
frameDuration = hopLength / fs;   % Interval between frames (seconds)
t = (0:numFrames-1) * frameDuration;
end
