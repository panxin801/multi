"""
Copyright 2020 Ye Bai by1993@qq.com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import torch
import torch.nn as nn
import torchaudio.compliance.kaldi as kaldi

from complexCNN import ComplexConv as CConv
import utils


class SPLayer(nn.Module):
    def __init__(self, config, channels):
        super(SPLayer, self).__init__()
        self.config = config
        self.channels = channels
        self.feature_type = config["feature_type"]
        self.sample_rate = float(config["sample_rate"])
        self.num_mel_bins = int(config["num_mel_bins"])
        self.use_energy = config["use_energy"]
        self.spec_aug_conf = None
        if "spec_aug" in config:
            self.spec_aug_conf = {
                "freq_mask_num": config["spec_aug"]["freq_mask_num"],
                "freq_mask_width": config["spec_aug"]["freq_mask_width"],
                "time_mask_num": config["spec_aug"]["time_mask_num"],
                "time_mask_width": config["spec_aug"]["time_mask_width"],
            }

        if self.feature_type == "mfcc":
            self.num_ceps = config["num_ceps"]
        else:
            self.num_ceps = None
        if self.feature_type == "offline":
            feature_func = None
            logging.warn(
                "Use offline features. It is your duty to keep features match."
            )
        elif self.feature_type == "fbank":

            def feature_func(waveform):
                return kaldi.fbank(waveform,
                                   sample_frequency=self.sample_rate,
                                   use_energy=self.use_energy,
                                   num_mel_bins=self.num_mel_bins)
        elif self.feature_type == "mfcc":

            def feature_func(waveform):
                return kaldi.mfcc(waveform,
                                  sample_frequency=self.sample_rate,
                                  use_energy=self.use_energy,
                                  num_mel_bins=self.num_mel_bins)
        elif self.feature_type == "complex":

            def feature_func(waveform):
                return utils.complex(waveform,
                                     sample_frequency=self.sample_rate,
                                     use_energy=self.use_energy,
                                     num_mel_bins=self.num_mel_bins)
        else:
            raise ValueError("unknown feature type.")

        self.func = feature_func
        self.CConv = CConv(self.channels,
                           self.channels, (3, 5),
                           padding=(1, 2))
        self.LastCConv = CConv(self.channels, 1, (3, 5), padding=(1, 2))
        self.LastConv = nn.Conv2d(2, 1, (1, 1))

    def spec_aug(self, padded_features, feature_lengths):
        freq_means = torch.mean(padded_features, dim=-1)
        time_means = (torch.sum(padded_features, dim=1) /
                      feature_lengths[:, None].float()
                      )  # Note that features are padded with zeros.

        B, T, V = padded_features.shape
        # mask freq
        for _ in range(self.spec_aug_conf["freq_mask_num"]):
            fs = (self.spec_aug_conf["freq_mask_width"] * torch.rand(
                size=[B], device=padded_features.device,
                requires_grad=False)).long()
            f0s = ((V - fs).float() * torch.rand(size=[B],
                                                 device=padded_features.device,
                                                 requires_grad=False)).long()
            for b in range(B):
                padded_features[b, :,
                                f0s[b]:f0s[b] + fs[b]] = freq_means[b][:, None]

        # mask time
        for _ in range(self.spec_aug_conf["time_mask_num"]):
            ts = (self.spec_aug_conf["time_mask_width"] * torch.rand(
                size=[B], device=padded_features.device,
                requires_grad=False)).long()
            t0s = ((feature_lengths - ts).float() * torch.rand(
                size=[B], device=padded_features.device,
                requires_grad=False)).long()
            for b in range(B):
                padded_features[b, t0s[b]:t0s[b] +
                                ts[b], :] = time_means[b][None, :]
        return padded_features, feature_lengths

    def forward(self, wav_batch, lengths):
        if self.feature_type == "complex":
            batch_size, batch_length = wav_batch.shape[0], wav_batch.shape[2]
            if self.func is not None:
                features = []
                feature_lengths = []
                for i in range(batch_size):
                    featureLi = []
                    for chn in range(self.channels):
                        feature = self.func(wav_batch[i, chn].view(1, -1))
                        feature = feature.unsqueeze(0).permute(3, 0, 1, 2)
                        featureLi.append(feature)
                    feature_lengths.append(feature.shape[2])
                    features.append(torch.cat([ch for ch in featureLi], 1))

                # pad to max_length
                max_length = max(feature_lengths)
                padded_features = torch.zeros(batch_size, 2, self.channels,
                                              max_length,
                                              feature.shape[-1]).cuda()
                for i in range(batch_size):
                    padded_features[i, :] += features[i].cuda()
            else:
                padded_features = torch.tensor(wav_batch)
                feature_lengths = lengths

            feature_lengths = torch.tensor(feature_lengths).long().to(
                padded_features.device)

            retCC = self.CConv(padded_features)
            retCC = self.CConv(retCC)
            padded_features = self.LastCConv(retCC)
            # New added line seems dim mismatch.
            # Bug: Expected 4-dimensional input for 4-dimensional weight [1, 2, 1, 1], but got 3-dimensional input of size [2, 2818, 257] instead
            padded_features = padded_features.squeeze(2)
            # if 3 == padded_features.dim():
            #    padded_features = padded_features.unsqueeze(0)
            padded_features = self.LastConv(padded_features).squeeze(1)
            # seems like [25,736,257]

            # if self.training and self.spec_aug_conf is not None:
            #     padded_features, feature_lengths = self.spec_aug(
            #         padded_features, feature_lengths)

        else:  # feature is fbank or mfcc
            batch_size, batch_length = wav_batch.shape[0], wav_batch.shape[1]
            if self.func is not None:
                features = []
                feature_lengths = []
                for i in range(batch_size):
                    feature = self.func(wav_batch[i, :lengths[i]].view(1, -1))
                    features.append(feature)
                    feature_lengths.append(feature.shape[0])

                # pad to max_length
                max_length = max(feature_lengths)
                padded_features = torch.zeros(batch_size, max_length,
                                              feature.shape[-1]).to(
                                                  feature.device)
                for i in range(batch_size):
                    l = feature_lengths[i]
                    padded_features[i, :l, :] += features[i]
            else:
                padded_features = torch.tensor(wav_batch)
                feature_lengths = lengths

            feature_lengths = torch.tensor(feature_lengths).long().to(
                padded_features.device)

            if self.training and self.spec_aug_conf is not None:
                padded_features, feature_lengths = self.spec_aug(
                    padded_features, feature_lengths)

        return padded_features, feature_lengths
