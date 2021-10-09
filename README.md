# MultiASR

Based on [openASR](https://github.com/by2101/OpenASR) from by2101. I'm just use this for further experiment.

A pytorch based end2end speech recognition system. The main architecture is [Speech-Transformer](https://ieeexplore.ieee.org/abstract/document/8462506/).



## Features

1. **Good Performance**. The system includes advanced algorithms, such as Label Smoothing, SpecAug, LST, and achieves good performance on ASHELL1. The baseline CER on AISHELL1 test is 6.6, which is better than ESPNet.

3. **Modular Design**. We divided the system into several modules, such as trainer, metric, schedule, models. It is easy for extension and adding features.
4. **End2End**. The feature extraction and tokenization are online. The system directly processes wave file. So, the procedure is much simpified.

## Dependency

* python >= 3.5
* librosa >= 0.7.2
* pytorch >= 1.7
* torchaudio >= 0.7
* pyyaml >= 5.1
* tensorboardX for visualization. (if you do not need visualize the results, you can set TENSORBOARD_LOGGING to 0 in src/utils.py)

## Usage

We use KALDI style example organization. The example directory include top-level shell scripts, data directory, exp directory. We provide an AISHELL-1 example. The path is ROOT/egs/aishell1/s5.

### Data Preparation
The data preparation script is prep_data.sh. It will automaticlly download AISHELL-1 dataset, and format it into KALDI style data directory. Then, it will generate json files, and grapheme vocabulary. You can set `corpusdir` for storing dataset.

    bash prep_data.sh

Then, it will generate data directory and exp directory.

### Train Models
We use yaml files for parameter configuration. We provide 3 examples.

    config_base.yaml  # baseline ASR system
    config_lm_lstm.yaml  # LSTM language model
    config_lst.yaml  # training ASR with LST

Run train.sh script for training baseline system.

    bash train.sh

### Model Averaging
Average checkpoints for improving performance.

    bash avg.sh

### Decoding and Scoring
Run decode_test.sh script for decoding test set.

    bash decode_test.sh
    bash score.sh data/test/text exp/multiASR/decode_test_avg-last10

## Visualization
We provide TensorboardX based visualization. The event files are stored in $expdir/log. You can use tensorboard to visualize the training procedure.

    tensorboard --logdir=$expdir --bind_all

Then you can see procedures in browser (http://localhost:6006).

Examples:

![per token loss in batch](https://github.com/by2101/OpenASR/raw/master/figs/loss.png)

![encoder attention](https://github.com/by2101/OpenASR/raw/master/figs/enc_att.png)

![encoder-decoder attention](https://github.com/by2101/OpenASR/raw/master/figs/dec_enc_att.png)


## Acknowledgement
This system is implemented with PyTorch. We use wave reading codes from SciPy. We use SCTK software for scoring. Thanks to Dan Povey's team and their KALDI software. I learn ASR concept, and example organization from KALDI. And thanks to Google Lingvo Team. I learn the modular design from Lingvo.

## Bib
@article{bai2019learn,
  title={Learn Spelling from Teachers: Transferring Knowledge from Language Models to Sequence-to-Sequence Speech Recognition},
  author={Bai, Ye and Yi, Jiangyan and Tao, Jianhua and Tian, Zhengkun and Wen, Zhengqi},
  year={2019}
}

## References
Dong, Linhao, Shuang Xu, and Bo Xu. "Speech-transformer: a no-recurrence sequence-to-sequence model for speech recognition." 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2018.
Zhou, Shiyu, et al. "Syllable-based sequence-to-sequence speech recognition with the transformer in mandarin chinese." arXiv preprint arXiv:1804.10752 (2018).
