'''
Author: Copyright @ Xin Pan

'''

import argparse
import os
import io
import logging
import subprocess
import time
import numpy as np
import torch
import torch.nn as nn
import torchaudio.compliance.kaldi as kaldi
from scipy.io import wavfile

from third_party import kaldi_io as kio
import pdb

TENSORBOARD_LOGGING = 0


def cleanup_ckpt(expdir, num_last_ckpt_keep):
    ckptlist = [
        t for t in os.listdir(expdir)
        if t.endswith(".pt") and t != "last-ckpt.pt"
    ]
    ckptlist = sorted(ckptlist)
    ckptlist_rm = ckptlist[:-num_last_ckpt_keep]
    logging.info(
        "Clean up checkpoints. Remain the last {} checkpoints.".format(
            num_last_ckpt_keep))
    for name in ckptlist_rm:
        os.remove(os.path.join(expdir, name))


def get_command_stdout(command, require_zero_status=True):
    """Executes a command and returns its stdout output as a string.  The
    command is executed with shell=True, so it may contain pipes and
    other shell constructs.

    If require_zero_stats is True, this function will raise an exception if
    the command has nonzero exit status.  If False, it just prints a warning
    if the exit status is nonzero.

    See also: execute_command, background_command
    """
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)

    stdout = p.communicate()[0]
    if p.returncode != 0:
        output = "Command exited with status {0}: {1}".format(
            p.returncode, command)
        if require_zero_status:
            raise Exception(output)
        else:
            logging.warning(output)
    return stdout


def load_wave(path, channels):
    """
    path can be wav filename or pipeline
    channels means how many channels as input for each file
    """

    # parse path
    items = path.strip().split(":", 1)
    if len(items) != 2:
        raise ValueError("Unknown path format.")
    tag = items[0]
    path = items[1]
    # now Using file only
    if tag == "file":
        basename, ext = os.path.splitext(path)
        datas = []
        if channels > 1:
            basename = basename.rsplit("_", 1)[0]
            for id in range(channels):
                readName = "%s_%02d%s" % (basename, id, ext)
                sample_rate, data = wavfile.read(readName)
                datas.append(data)
        else:
            readName = "%s%s" % (basename, ext)
            sample_rate, data = wavfile.read(readName)
            datas.append(data)
    # elif tag == "pipe":
    #     path = path[:-1]
    #     out = get_command_stdout(path, require_zero_status=True)
    #     sample_rate, data = wavfile.read(io.BytesIO(out))
    # elif tag == "ark":
    #     fn, offset = path.split(":", 1)
    #     offset = int(offset)
    #     with open(fn, 'rb') as f:
    #         f.seek(offset)
    #         sample_rate, data = wavfile.read(f, offset=offset)
    else:
        raise ValueError("Unknown file tag.")
    datas = np.array(datas, dtype=np.float32)
    return sample_rate, datas


def load_feat(path):
    items = path.strip().split(":", 1)
    if len(items) != 2:
        raise ValueError("Unknown path format.")
    tag = items[0]
    path = items[1]
    if tag == "ark":
        return kio.read_mat(path)
    else:
        raise ValueError("Unknown file tag.")


def parse_scp(fn):
    dic = {}
    with open(fn, "r", encoding="utf8") as f:
        cnt = 0
        for line in f:
            cnt += 1
            items = line.strip().split(" ", 1)
            if len(items) != 2:
                logging.warning(
                    "Wrong formated line {} in scp {}, skip it.".format(
                        cnt, fn))
                continue
            dic[items[0]] = items[1]
    return dic


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


class Timer(object):
    def __init__(self):
        self.start = 0.0

    def tic(self):
        self.start = time.time()

    def toc(self):
        return time.time() - self.start


# ==========================================
# auxilary functions for sequence
# ==========================================


def get_paddings(src, lengths):
    paddings = torch.zeros_like(src).to(src.device)
    for b in range(lengths.shape[0]):
        paddings[b, lengths[b]:, :] = 1
    return paddings


def get_paddings_by_shape(shape, lengths, device="cpu"):
    paddings = torch.zeros(shape).to(device)
    if shape[0] != lengths.shape[0]:
        raise ValueError("shape[0] does not match lengths.shape[0]:"
                         " {} vs. {}".format(shape[0], lengths.shape[0]))
    T = shape[1]
    for b in range(shape[0]):
        if lengths[b] < T:
            l = lengths[b]
            paddings[b, l:] = 1
    return paddings


def get_transformer_padding_byte_masks(B, T, lengths):
    # !!!! torch version changing: byte to bool
    masks = get_paddings_by_shape([B, T], lengths).bool()
    return masks


def get_transformer_casual_masks(T):
    masks = -torch.triu(torch.ones(T, T), diagonal=1) * 9e20
    return masks


# ==========================================
# visualization
# ==========================================
if TENSORBOARD_LOGGING == 1:
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)

    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from tensorboardX import SummaryWriter

    class Visualizer(object):
        def __init__(self):
            self.writer = None
            self.fig_step = 0

        def set_writer(self, log_dir):
            if self.writer is not None:
                raise ValueError("Dont set writer twice.")
            self.writer = SummaryWriter(log_dir)

        def add_scalar(self, tag, value, step):
            self.writer.add_scalar(tag=tag,
                                   scalar_value=value,
                                   global_step=step)

        def add_graph(self, model):
            self.writer.add_graph(model)

        def add_image(self, tag, img, data_formats):
            self.writer.add_image(tag, img, 0, dataformats=data_formats)

        def add_img_figure(self, tag, img, step=None):
            fig, axes = plt.subplots(1, 1)
            axes.imshow(img)
            self.writer.add_figure(tag, fig, global_step=step)

        def close(self):
            self.writer.close()

    visualizer = Visualizer()


# ==========================================
# extract complex-fft feature
# ==========================================
def complex(waveform: torch.Tensor,
            blackman_coeff=0.42,
            channel: int = -1,
            dither=1.0,
            energy_floor=0.0,
            frame_length=25.0,
            frame_shift=10.0,
            high_freq=0.0,
            htk_compat=False,
            low_freq=20.0,
            min_duration=0.0,
            num_mel_bins=23,
            preemphasis_coefficient=0.97,
            raw_energy=True,
            remove_dc_offset=True,
            round_to_power_of_two=True,
            sample_frequency=16000.0,
            snip_edges=True,
            subtract_mean=False,
            use_energy=False,
            use_log_fbank=True,
            use_power=True,
            vtln_high=-500.0,
            vtln_low=100.0,
            vtln_warp=1.0,
            window_type=kaldi.POVEY):
    waveform, window_shift, window_size, padded_window_size = kaldi._get_waveform_and_window_properties(
        waveform.cpu(), channel, sample_frequency, frame_shift, frame_length,
        round_to_power_of_two, preemphasis_coefficient)

    if len(waveform) < min_duration * sample_frequency:
        # signal is too short
        return torch.empty(0)

    # strided_input, size (m, padded_window_size) and signal_log_energy, size (m)
    strided_input, signal_log_energy = kaldi._get_window(
        waveform, padded_window_size, window_size, window_shift, window_type,
        blackman_coeff, snip_edges, raw_energy, energy_floor, dither,
        remove_dc_offset, preemphasis_coefficient)

    # size (m, padded_window_size // 2 + 1, 2)
    fft = torch.rfft(strided_input, 1, normalized=False,
                     onesided=True).to(waveform.device)
    # fft = fft.permute(2, 0, 1)

    return fft
