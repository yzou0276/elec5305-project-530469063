Repo： https://github.com/yzou0276/elec5305-project-530469063

Page： https://yzou0276.github.io/elec5305-project-530469063

# Lightweight DS-CNN Voice Activity Detection (VAD)

Name: YutingZou
SID: 530469063
GitHub Username: yzou0276

This repository contains the implementation of my ELEC5305 project on **lightweight voice activity detection (VAD)** using a **Depthwise Separable CNN (DS-CNN)**.

The task is a **binary classification** problem on short audio segments (0.5 s). For each segment the system decides:

- `speech`: the segment contains a spoken command word  
- `nospeech`: the segment contains only background noise / no speech  

---

## 1. Research Question

> Can a small DS-CNN model using log-Mel features provide significantly better voice activity detection performance than a simple energy-threshold VAD, on short (0.5 s) segments from the Google Speech Commands dataset?

---

## 2. Dataset

- Based on **Google Speech Commands v0.02**.
- **Speech class**  
  - Uses six command words: `yes`, `no`, `left`, `right`, `go`, `down`.  
  - Audio is resampled to **16 kHz** and cut into **0.5 s** clips.
- **Nospeech class**  
  - Uses recordings from the `_background_noise_` folder.  
  - Long noise files are resampled to 16 kHz, converted to mono, and sliced into 0.5 s noise segments.
- Final dataset is balanced by capping the maximum number of clips per class.

> Note: Because of the GitHub 25 MB limit, the `speech/` folder is not uploaded.  
> Users must download Speech Commands v0.02 themselves and copy the six keyword folders into `speech/`.  
> The `nospeech/` folder is provided via `nospeech.zip` in this repo.

---

## 3. Methods

### 3.1 DS-CNN VAD (proposed method)

1. **Preprocessing & feature extraction**
   - Resample all audio to **16 kHz**.
   - Convert stereo to **mono**.
   - Trim or zero-pad each clip to **0.5 s**.
   - Extract **40-dim log-Mel spectrogram** features with a 10 ms hop.
2. **Model**
   - Lightweight **Depthwise Separable CNN**:
     - Initial 2-D convolution + BatchNorm + ReLU + max-pooling.  
     - Depthwise convolution + BatchNorm + ReLU.  
     - Pointwise convolution + BatchNorm + ReLU.  
     - Global average pooling + fully connected layer to 2 classes + softmax.
3. **Training**
   - Training / validation / test split (approx. 55% / 15% / 30%).  
   - Optimizer: **Adam**, 40 epochs, mini-batch size 64.  
   - Evaluation metrics: **Accuracy, Precision, Recall, F1-score** on the held-out test set.

### 3.2 Energy-Threshold VAD (baseline)

1. Use the **same test clips** as the DS-CNN evaluation.
2. For each clip, compute **short-time energy** over 30 ms frames with 10 ms hop.
3. Compute the ratio of frames whose energy is above a fixed threshold.
4. If this ratio > 0.1 → predict **speech**, otherwise predict **nospeech**.
5. Evaluate with the same metrics for a fair comparison.

---

## 4. Key Results

Test-set performance (0.5 s segments):

- **DS-CNN VAD**
  - Accuracy: **88.75%**
  - Precision: **0.8233**
  - Recall: **0.9831**
  - F1-score: **0.8962**

- **Energy-based baseline**
  - Accuracy: **46.88%**
  - Precision: **0.4795**
  - Recall: **0.8861**
  - F1-score: **0.6222**

The confusion matrices show that the DS-CNN:

- Keeps a **very low miss rate for speech** (high recall), and  
- **Greatly reduces false alarms** on background noise compared with the simple energy method.

---

## 5. Repository Contents (high level)

- `main_train_Yuting_Zou.m` – main script to run the full experiment (data loading, training, evaluation).  
- `main_train_vad.mlx` – MATLAB Live Script version with formatted output and plots.  
- `loadVADDataset.m` – builds the dataset and extracts log-Mel features for all clips.  
- `extractPCENFeatures.m` – computes log-Mel spectrogram features for one audio signal.  
- `buildDSCNN.m` – defines the DS-CNN architecture.  
- `computeMetrics.m` – computes Accuracy, Precision, Recall, F1, and confusion matrix.  
- `baseline_energy_vad.m` – runs the energy-threshold baseline VAD on the test set.  
- `prepareNospeechSegments.m` – generates 0.5 s noise clips from `_background_noise_`.  
- `nospeech.zip` – pre-generated nospeech segments (unzip to get the `nospeech/` folder).  

---

## 6. How to Run

1. **Prepare data**
   - Download Speech Commands v0.02.  
   - Create a local `speech/` folder in the project root containing:
     ```text
     speech/
       yes/
       no/
       left/
       right/
       go/
       down/
     ```
   - Unzip `nospeech.zip` in the project root to create the `nospeech/` folder  
     (or place raw noise in `nospeech/_background_noise_/` and run `prepareNospeechSegments.m` once).

2. **Run the experiment**
   - Open the project folder in MATLAB.  
   - Run:
     ```matlab
     main_train_Yuting_Zou
     ```

3. **What you will see**
   - Data loading and log-Mel feature extraction log messages.  
   - Training progress of the DS-CNN (mini-batch accuracy / validation accuracy / losses).  
   - Final DS-CNN test metrics: Accuracy, Precision, Recall, F1-score.  
   - Confusion matrix figure for the DS-CNN model.  
   - Baseline energy-VAD test metrics and confusion matrix.

---

## 7. Links

- **Code repository**: [(this GitHub repo)](https://github.com/yzou0276/elec5305-project-530469063/tree/main/code)  
- **Project report (PDF)**: 
- **Result figures**:
 https://github.com/yzou0276/elec5305-project-530469063/tree/main/Result%20Figure   
- **Demo video**: 


