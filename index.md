## Deep Learning Voice Activity Detction in Noisy Environments


Name: YutingZou
SID: 530469063
GitHub Username: yzou0276




## Project Overview

The method is divided into six parts: data preparation and annotation, feature engineering, model design, training strategy and loss, post-processing and online judgment, and evaluation and ablation. We strive to switch features, architectures, or post-processing strategies through configuration without changing the code. The overall engineering goal is end-to-end runnability and reproducibility. Each training run is completed with the configuration, random seed, dependency versions, and command line calls to ensure that the results correspond to the configuration.

The evaluation includes not only accuracy but also system metrics (RTF, memory, and size) and latency breakdown, providing a transparent trade-off between accuracy, latency, and size.

## Background & Motivation

Traditional VADs often lack transferable thresholds under conditions such as non-stationary noise, far-field reverberation, and microphone response variations, leading to missed detections. This results in downstream recognition wasting computational resources on a large number of invalid frames and amplifying word errors. Furthermore, lightweight VADs, commonly used in engineering, are prone to jitter under edge conditions.

Deep models can learn speech-like dynamic structures and harmonic patterns from time-frequency representations, maintaining stable boundaries even at low signal-to-noise ratios. Therefore, this project aims to develop a VAD baseline that can be implemented on-device using a small model and unified evaluation.


## Proposed Methodology


This project implements the entire VAD pipeline using only MATLAB. The workflow is as follows: data normalization, feature extraction, small model training, threshold calibration and post-processing, and online evaluation. First, any raw audio is normalized to mono and optionally pre-emphasized. Two main features are then extracted, and CMVN is performed on each segment. The frame context is then concatenated to form a time window vector.

Training uses Adam with learning rate scheduling and early stopping. The loss is a weighted BCE/Focal loss, with an optional temporal consistency regularization to improve boundary consistency. To enhance robustness, switchable additive noise is enabled in the waveform domain.


During the inference phase, frame-level speech is output, and segment-level decisions are then made using thresholds automatically calibrated from the validation set. Dual-threshold hysteresis is used, and for smoother decoding, two-state HMM/Viterbi decoding can be switched. To ensure real-time performance, the system provides a block interface, uses a rolling buffer for features, maintains a hidden state cache in the RNN, and employs a causal configuration for convolution. Key comparisons are repeated at least multiple times with different random seeds, and the mean ± standard deviation is reported.


## Expected Outcomes

The project will deliver a working MATLAB VAD prototype, including complete training, inference, evaluation, and plotting scripts and a unified configuration structure. This allows for a one-click reproducible "configure as experiment" workflow. The prototype outputs frame-level posteriors and segment-level start and end timestamps offline, automatically generating visualizations.

The evaluation will report two metrics—frame-level and segment-level—using a unified protocol. Baseline and ablation comparisons will also be provided.

## Timeline

Week 6, 7 Baseline connection: Configure the structure/script skeleton, normalize data and align labels, implement features and model mainline, run the first training and threshold calibration, and generate the first version of the confusion matrix.

Week 8,9 Capacity Development: Continue to supplement the main model, add data enhancement, and complete the traditional baseline vs. deep learning. Unify the plotting script output to set up comparison bar charts, frame/segment PR/ROC curves

Week 10, 11 Engineering and Optimization: Improve post-processing, implement streaming/block-based inference, measure and analyze end-to-end latency. Conduct lightweight experiments, calculate RTF, peak memory usage, and parameter count, and determine candidate final models.

Week 12, 13 Complete summary: Lock in the final configuration, replicate the experiment with multiple random seeds, and summarize the mean standard deviation and significance. Compile the report and improve the system diagram.



## Reference

Hwang, I., Park, H. M., & Chang, J. H. (2016). Ensemble of deep neural networks using acoustic environment classification for statistical model-based voice activity detection. Computer Speech & Language, 38, 1-12.

Ball, J. (2023). Voice Activity Detection (VAD) in Noisy Environments. arXiv preprint arXiv:2312.05815.

Purwins, H., Li, B., Virtanen, T., Schlüter, J., Chang, S. Y., & Sainath, T. (2019). Deep learning for audio signal processing. IEEE Journal of Selected Topics in Signal Processing, 13(2), 206-219.

Chen, G., Parada, C., & Heigold, G. (2014, May). Small-footprint keyword spotting using deep neural networks. In 2014 IEEE international conference on acoustics, speech and signal processing (ICASSP) (pp. 4087-4091). IEEE.

Lin, R., Costello, C., Jankowski, C., & Mruthyunjaya, V. (2019). Optimizing Voice Activity Detection for Noisy Conditions. In INTERSPEECH (pp. 2030-2034).

