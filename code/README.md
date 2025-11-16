# Lightweight DS-CNN Voice Activity Detection (VAD)

This repository implements a **binary voice activity detection (VAD) system** that decides whether a short audio segment contains speech or not.

- `speech`: segments that contain speech (command words “yes / no / left / right / go / down”)
- `nospeech`: background noise / no speech

We use **log-Mel spectrogram features** and a **lightweight DS-CNN (Depthwise Separable CNN)** model, and compare it with a simple **energy-threshold baseline**.

This code is part of the *Speech and Audio Processing Project*.

---

## Dataset

- Speech data is taken from **Google Speech Commands v0.02**.  
  Only 6 command words are used: `yes`, `no`, `left`, `right`, `go`, `down`.
- Non-speech data is taken from the dataset’s `_background_noise_` folder and cut into 0.5-second noise segments to form the `nospeech` class.
- All audio is resampled to **16 kHz**, and each sample is **0.5 seconds** long.

> **Note**
> - Because of GitHub’s 25 MB limit, the `speech/` folder is **not** uploaded.  
> - You must download Speech Commands v0.02 yourself, and copy the six keyword folders into a local `speech/` directory.  
> - This repo contains `nospeech.zip`. Unzip it in the project root to get the `nospeech/` folder with pre-cut noise segments.

---

## File Overview

### 1. Main scripts

- `main_train_Yuting_Zou.m`  
  - Main command-line script that runs the **full experiment**:
    1. Reads audio files from `speech/` and `nospeech/`.
    2. Calls `loadVADDataset.m` to extract log-Mel features.
    3. Splits the data into training / validation / test sets.
    4. Calls `buildDSCNN.m` to build the DS-CNN model and trains it.
    5. Evaluates DS-CNN on the test set (Accuracy, Precision, Recall, F1).
    6. Calls `baseline_energy_vad.m` to evaluate the energy-based baseline on the same test set and compares both methods.
  - How to use:
    1. Set the MATLAB current folder to the project root.
    2. Make sure `speech/` and `nospeech/` exist.
    3. Run in the MATLAB command window:
       ```matlab
       main_train_Yuting_Zou
       ```

- `main_train_vad.mlx`  
  - MATLAB Live Script version of the same experiment logic as `main_train_Yuting_Zou.m`.
  - Provides formatted sections, inline outputs, training curves and confusion matrices.
  - Recommended if you want to run the project interactively inside MATLAB.
  - How to use: double-click the file in MATLAB and press **Run**.

---

### 2. Data preparation & feature extraction

- `prepareNospeechSegments.m`  
  - Reads long background noise files from `nospeech/_background_noise_/`.
  - Resamples audio to 16 kHz and converts to mono.
  - Slides a 0.5-second window along each file and saves each segment as `nospeech/noise_XXXX.wav`.
  - Typical usage:
    - First time you prepare the `nospeech/` data:
      1. Place raw background noise wav files inside `nospeech/_background_noise_/`.
      2. Run in MATLAB:
         ```matlab
         prepareNospeechSegments
         ```
      3. The script will generate multiple 0.5-second noise segments in `nospeech/`.

- `loadVADDataset.m`  
  - Builds the full VAD dataset from the `speech/` and `nospeech/` folders:
    - Reads all wav files using `audioDatastore`.
    - Optionally limits the maximum number of files per class.
    - Resamples to 16 kHz, converts to mono.
    - Trims or zero-pads each clip to 0.5 seconds.
    - Calls `extractPCENFeatures.m` to compute log-Mel features.
    - Packs everything into:
      - `features`: 4-D array [nMels × T × 1 × N]
      - `labels`: vector of length N with class labels (0 = nospeech, 1 = speech)
  - This script is called inside `main_train_Yuting_Zou.m` and normally does not need to be executed directly.

- `extractPCENFeatures.m`  
  - Computes log-Mel spectrogram features for one audio signal:
    - Uses `melSpectrogram` to compute the Mel spectrogram over time.
    - Applies `log10(melSpec + 1e-6)` to avoid `log(0)`.
    - Generates a time axis based on the hop length and sampling rate.
  - Returns:
    - `feat`: 2-D log-Mel spectrogram.
    - `t`: corresponding time axis (seconds).
  - This is called by `loadVADDataset.m` for every audio file to keep feature size consistent.

---

### 3. Model & metrics

- `buildDSCNN.m`  
  - Defines the **DS-CNN architecture** used for VAD:
    - Image input layer with no extra normalization.
    - First 2-D convolution + batch normalization + ReLU + max-pooling.
    - Depthwise convolution + batch normalization + ReLU.
    - Pointwise convolution + batch normalization + ReLU.
    - Global average pooling.
    - Fully connected layer to two output classes (`speech` / `nospeech`), followed by softmax and classification layers.
  - Used in the main script as:
    ```matlab
    inputSize  = size(XTrain, [1 2 3]);  % [nMels, T, 1]
    numClasses = 2;
    layers = buildDSCNN(inputSize, numClasses);
    ```

- `computeMetrics.m`  
  - Computes performance scores from ground-truth and predicted labels:
    - Confusion matrix `cm`
    - Accuracy
    - Precision
    - Recall
    - F1-score
  - Used after classification on the test set, for both DS-CNN and the baseline, for example:
    ```matlab
    [acc, precision, recall, f1, cm] = computeMetrics(YTest, YPred);
    ```

---

### 4. Energy-based baseline

- `baseline_energy_vad.m`  
  - Implements a **simple energy-threshold VAD baseline**:
    - Uses the same test indices as the DS-CNN experiment.
    - For each test audio clip:
      - Computes a short-time energy sequence.
      - Counts the proportion of frames whose energy is above a fixed threshold.
      - If this proportion is greater than 0.1, predicts `speech`; otherwise predicts `nospeech`.
    - Returns a vector of predicted labels for the test set.
  - The main script then feeds these predictions into `computeMetrics.m` to obtain accuracy, precision, recall, F1, and a confusion matrix.  
    This allows a direct comparison between DS-CNN and the simple energy method.

---

### 5. Zip file and missing folder

- `nospeech.zip`  
  - Contains many pre-cut 0.5-second noise segments for the `nospeech` class.
  - How to use:
    - Download `nospeech.zip`.
    - Unzip it in the project root so that a `nospeech/` folder is created.
    - If you already generated your own noise segments with `prepareNospeechSegments.m`, you do not need this zip.

- `speech` folder (not in this repo)  
  - You must create it locally with the following structure:
    ```text
    speech/
      yes/
      no/
      left/
      right/
      go/
      down/
    ```
  - Each subfolder should contain the corresponding command word wav files from Speech Commands v0.02.

---

## How to run the project

1. **Prepare data**
   - Download the Speech Commands v0.02 dataset.
   - Copy the six keyword folders (`yes`, `no`, `left`, `right`, `go`, `down`) into a local `speech/` folder in the project root.
   - Either:
     - Unzip `nospeech.zip` in the project root to get `nospeech/`, **or**
     - Place raw background noise wav files in `nospeech/_background_noise_/` and run:
       ```matlab
       prepareNospeechSegments
       ```
     - This will create many 0.5-second noise clips in `nospeech/`.

2. **Open the project in MATLAB**
   - Set the **Current Folder** to the project root (the folder containing `main_train_Yuting_Zou.m`).

3. **Run the main experiment**
   ```matlab
   main_train_Yuting_Zou

