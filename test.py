import wave
import contextlib
import os
import re

from pydub import AudioSegment




def get_wavfile_length(filepath):
    with contextlib.closing(wave.open(filepath, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration * 1000


def cut_file(filepath):
    if os.path.exists(filepath):
        if filepath[-3:] == 'wav':
            total_time = get_wavfile_length(filepath)
            train_time = int(total_time * 0.7)
            print("Total time is " + str(total_time))
            train_file_name = filepath[:-4] + '_train.wav'
            valid_file_name = filepath[:-4] + '_valid.wav'
            wavfile = AudioSegment.from_wav(filepath)
            wavfile[:train_time].export(train_file_name, format='wav')
            wavfile[train_time:].export(valid_file_name, format='wav')


'''
def cut_file_with_span(filepath, starttime, endtime, radname):
    cut_file_name = "20181107_32_" + filepath[-11:-4] + '_rad' + str(radname) + '.wav'
    wavfile = AudioSegment.from_wav(dir + filepath)
    wavfile[start_time:end_time].export(cut_file_name, format='wav')
    print(cut_file_name+ "    " + str(get_wavfile_length(dir + cut_file_name)))
    return 0
'''


if __name__ == '__main__':
    count = 0
    wav_dir = "C:\\tmp\\Audio\\old_45#\\45#\\"
    for each in os.listdir(wav_dir):
        full_file_path = wav_dir + each
        if os.path.exists(full_file_path):
            print(each)
            cut_file(full_file_path)


'''
filepath = 'C:/Work/ZKLF/AudioData20181107/20181107_32_aigo002_952rad.wav'
filepathA = 'C:/Work/ZKLF/AudioData20181107/20181107_32_aigo002_952radA.wav'
filepathB = 'C:/Work/ZKLF/AudioData20181107/20181107_32_aigo002_952radB.wav'

with contextlib.closing(wave.open(filepath, 'r')) as f:

    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)
    used = duration * 0.7
    print(duration)
    print(int(used))


wavfile = AudioSegment.from_wav(filepath)
wavfile[:213000].export(filepathA, format='wav')
wavfile[213000:].export(filepathB, format='wav')
'''