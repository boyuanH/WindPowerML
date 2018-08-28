# -*- coding: utf-8 -*-
import os
import argparse
import wave
import configparser
from datetime import datetime

import numpy as np


class STFTModule:
    def __init__(self, sample_per_frame):
        self.sample_per_frame=int(sample_per_frame)
        temp = 1
        while temp < sample_per_frame:
            temp *= 2
        self.fft_size = int(temp * 2)
        self.window = np.hanning(sample_per_frame * 2)

    def fwd(self, x):
        frame_count = int(len(x) / self.sample_per_frame) - 1
        spctr = np.zeros([frame_count, int(self.fft_size / 2)], dtype=np.complex64)
        for i in range(frame_count):
            start = i * self.sample_per_frame
            windowed_x = np.concatenate((self.window * x[start:start+self.sample_per_frame*2],
                                         np.zeros(self.fft_size - self.sample_per_frame * 2)))
            spctr[i] = np.fft.fft(windowed_x)[:int(self.fft_size / 2)]
        return spctr


def generate_feature_file(inputfilepath, outputfilepath, channel_num, sample_per_frame):
    fconv = STFTModule(sample_per_frame)
    fp = wave.open(inputfilepath, 'rb')
    inputsignal = fp.readframes(fp.getnframes() * fp.getnchannels())
    inputsignal = np.frombuffer(inputsignal, dtype="int16") / 32768.0
    inputsignal = np.reshape(inputsignal, (-1, fp.getnchannels()))[3200:, channel_num]
    feature = fconv.fwd(inputsignal)
    feature = np.power(np.absolute(feature), 2.0).astype(np.float32)
    np.save(outputfilepath, feature)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make data set by mixing signal & noise')
    parser.add_argument('-m', '--model_file', type=str, default='./data/modelwavfile.txt',
                        help='model wav file list')
    parser.add_argument('-t', '--target_file', type=str, default='./data/targetwavfile.txt',
                        help='target wav file list')
    parser.add_argument('-c', '--config_file', type=str, default='./conf/PrepareDataConfig.ini',
                        help='input config file')
    parser.add_argument('-o', '--output_dir', default='./output', type=str, help='output file path')

    # parser.add_argument('-i', nargs='+', type=str, required=True, help='input wave file path')
    # parser.add_argument('-c', '--channel_num', default=0, type=int, help='channel number to convert')
    # parser.add_argument('-n', '--sample_per_frame', default=441, type=int, help='sample per frame for fft')
    # parser.add_argument('-t', '--target_file', nargs='+', type=str, help='result file')
    args = parser.parse_args()

    # read config para from PrepareDataConfig file

    config = configparser.ConfigParser()
    if not os.path.exists(args.config_file):
        print('No config file found')
        exit()
    config.read(args.config_file)
    channel_num = int(config.get("Preparedata", "channel_num"))
    sample_per_frame = int(config.get("Preparedata", "sample_per_frame"))

    # init npylist.txt

    output_npylist_filepath = os.path.join(args.output_dir, 'npylist.txt')
    if os.path.exists(args.output_dir):
        if os.path.exists(output_npylist_filepath):
            os.remove(output_npylist_filepath)
    else:
        os.makedirs(args.output_dir)

    npyfile = open(output_npylist_filepath, "w")

    # read model file path list

    if os.path.exists(args.model_file):
        modelfile = open(args.model_file)
        modelfiles = modelfile.readlines()
        start = datetime.now()
        for fileitem in modelfiles:
            wavfile = fileitem.strip()
            if os.path.exists(wavfile):
                filesuffix = wavfile[-3:]
                if ('WAV'.find(filesuffix.upper())) == 0:
                    output_filepath = os.path.join(args.output_dir, wavfile[:-4] + ".npy")
                    generate_feature_file(wavfile, output_filepath, channel_num, sample_per_frame)
                    npyfile.write(output_filepath + '\n')
        end = datetime.now()
        print('Training data Process time ' + str(end - start))
        npyfile.close()

    # Transform target wav file to npy file
    if os.path.exists(args.target_file):
        targetfile = open(args.target_file)
        targetfiles = targetfile.readlines()
        start = datetime.now()
        for fileitem in targetfiles:
            target_file = fileitem.strip()
            if os.path.exists(target_file):
                tarfilesuffix = target_file[-3:]
                if ('WAV'.find(tarfilesuffix.upper())) == 0:
                    output_filepath = os.path.join(args.output_dir, target_file[:-4] + ".npy")
                    generate_feature_file(target_file, output_filepath, channel_num, sample_per_frame)
        end = datetime.now()
        print('Target data Process time ' + str(end - start))


