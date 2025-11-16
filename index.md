## Deep Learning Voice Activity Detction in Noisy Environments


Name: YutingZou
SID: 530469063
GitHub Username: yzou0276



Repo： https://github.com/yzou0276/elec5305-project-530469063

Page： https://yzou0276.github.io/elec5305-project-530469063




# Lightweight Voice Activity Detection with DS-CNN

This project implements a simple binary voice activity detection (VAD) task
using the Google Speech Commands v0.02 dataset. We compare a lightweight
depthwise separable CNN (DS-CNN) using log-Mel spectrogram features with a
short-time energy threshold baseline.

## Research Question

Can a compact DS-CNN using log-Mel features significantly outperform a simple
short-time energy VAD baseline on 0.5-second clips derived from the Speech
Commands dataset?

## Requirements

- MATLAB R202x (tested on R20xxb)
- Deep Learning Toolbox

## Dataset

We use the Google Speech Commands v0.02 dataset.

- Official download link: [link to dataset]
- Please download and extract the dataset to: `data/speech_commands_v0.02/`

We select six command words (`yes`, `no`, `left`, `right`, `go`, `down`) as
speech examples and use the `_background_noise_` folder to generate non-speech
examples.

## How to Run

1. Download and extract the Speech Commands v0.02 dataset.
2. Open MATLAB and set the current folder to the `code/` directory.
3. Run the main live script:
   ```matlab
   main_train_vad

