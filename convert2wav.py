import os
import numpy as np


def get_txt_data(path):
    if os.path.exists(path) is False:
        return
    contentlines = open(path).readlines()
    ch1 = []
    ch2 = []
    mictime = []
    for line in contentlines:
        line = line.strip()
        items = line.split()
        mictime.append(items[0])
        ch1.append(items[1])
        ch2.append(items[2])
    return mictime, ch1, ch2


if __name__ == '__main__':
    filepath = 'C:/Work/ZKLF/MicTest/record/002.MB_256.txt'
    get_txt_data(filepath)
