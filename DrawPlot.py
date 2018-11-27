import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import os


def drawplotimage(csvfilepath, outputfilepath):
    plt.cla()
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
    x, y = np.loadtxt(csvfilepath, dtype=str, delimiter=',', unpack=True)
    y_data = y[0:].astype(np.float32)
    x_data = range(0, len(y_data))
    plt.plot(x_data, y_data, 'o', color='red', label="")
    plt.xlabel('time')
    plt.ylabel('score')
    plt.legend()
    plt.savefig(outputfilepath)


def list_all_file(rootpath):
    _files = []
    file_list = os.listdir(rootpath)

    for i in range(0, len(file_list)):
        path = os.path.join(rootpath, file_list[i])
        if os.path.isdir(path):
            _files.extend(list_all_file(path))
        if os.path.isfile(path):
            if path[-3:] == "csv":
                _files.append(path)
    return _files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='result score')
    parser.add_argument('-i', '--csv_filepath', required=True, type=str, help='1')
    parser.add_argument('-o', '--outputfilepath', default='./output/result.png', type=str, help='2')
    args = parser.parse_args()
    c1 = "C:\\Work\\ZKLF\\AudioData20181107\\Result\\32\\aigo005\\C\\32-aigo005-C_20181107_32_aigo005_121rad_valid.csv"
    c2 = "C:\\Work\\ZKLF\\AudioData20181107\\Result\\32\\aigo005\\C\\32-aigo005-C_20181107_32_aigo005_121rad_valid.png"
    '''
    drawplotimage(c1, c2)
    '''
    
    datapath = "C:\\Work\\ZKLF\\AudioData20181107\\Result\\"
    csvfiles = list_all_file(datapath)
    i = 0
    print(len(csvfiles))
    for csvfile in csvfiles:
        filename = os.path.split(csvfile)
        output_path = os.path.abspath(os.path.join(csvfile, ".."))
        output_path = os.path.join(output_path, filename[1][:-4])
        output_path = output_path + ".png"
        print(str(i)+" : " + filename[1])
        i = i + 1
        drawplotimage(csvfile, output_path)


