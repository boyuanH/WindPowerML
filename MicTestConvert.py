import numpy as np
import wave
import os
import math


def txt2wav(dir_path, sample_rate):
    filelist = os.listdir(dir_path)
    for file in filelist:
        if file[-3:] == 'txt':
            filename = file[0:-4]
            filelines = open(os.path.join(dir_path,file)).readlines()
            ch01 = []
            ch02 = []
            max_ch01 = 0
            max_ch02 = 0
            for line in filelines:
                line = line.strip()
                data_array = line.split()
                curr_01 = math.fabs(float(data_array[1]))
                curr_02 = math.fabs(float(data_array[2]))
                if curr_01 > max_ch01:
                    max_ch01 = curr_01
                if curr_02 > max_ch02:
                    max_ch02 = curr_02
                ch01.append(float(data_array[1]))
                ch02.append(float(data_array[2]))
            data_ch01 = np.asarray(ch01)
            data_ch02 = np.asarray(ch02)
            data_s_ch01 = data_ch01 / max_ch01 * sample_rate
            data_s_ch02 = data_ch02 / max_ch02 * sample_rate
            wave_ch01 = data_s_ch01.astype(np.short)
            wave_ch02 = data_s_ch02.astype(np.short)
            f_01 = wave.open(os.path.join(dir_path, filename+'_01.wav'), 'wb')
            f_01.setnchannels(1)
            f_01.setframerate(sample_rate)
            f_01.setsampwidth(2)
            f_01.writeframes(wave_ch01.tostring())
            f_01.close()
            f_02 = wave.open(os.path.join(dir_path, filename+'_02.wav'), 'wb')
            f_02.setnchannels(1)
            f_02.setframerate(sample_rate)
            f_02.setsampwidth(2)
            f_02.writeframes(wave_ch02.tostring())
            f_02.close()


if __name__ == '__main__':
    filepath = './data/MicTest'
    txt2wav(filepath, 16384)