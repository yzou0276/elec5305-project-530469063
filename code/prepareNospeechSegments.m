function prepareNospeechSegments()

sr = 16000;          
clipLenSec = 0.5;    
clipLenSamples = round(sr * clipLenSec);

srcDir = fullfile(pwd, 'nospeech', '_background_noise_');
dstDir = fullfile(pwd, 'nospeech');
ads = audioDatastore(srcDir, 'IncludeSubfolders', true, 'FileExtensions','.wav');

idx = 1;
while hasdata(ads)
    [audioIn, info] = read(ads);
    fs = info.SampleRate;

    % Resampling + Mono
    if fs ~= sr
        audioIn = resample(audioIn, sr, fs);
    end
    audioIn = mean(audioIn, 2);
    L = numel(audioIn);
    % Slide a slice every 0.5 second.
    startIdx = 0.5;
    while startIdx + clipLenSamples - 1 <= L
        seg = audioIn(startIdx:startIdx + clipLenSamples - 1);
        outName = fullfile(dstDir, sprintf('noise_%04d.wav', idx));
        audiowrite(outName, seg, sr);
        idx = idx + 1;
        startIdx = startIdx + clipLenSamples;   
    end
end

fprintf('Done! Generated %d nospeech segments.\n', idx-1);
end
