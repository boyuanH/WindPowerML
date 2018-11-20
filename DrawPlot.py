import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse


def drawplotimage(csvfilepath, outputfilepath):
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
    x, y = np.loadtxt(csvfilepath, delimiter=',', unpack=True)
    plt.plot(x, y, 'o', color='red')
    plt.xlabel('time')
    plt.ylabel('score')
    plt.legend()
    #plt.show()
    plt.savefig(outputfilepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='result score')
    parser.add_argument('-i', '--csv_filepath', required=True, type=str, help='1')
    parser.add_argument('-o', '--outputfilepath', default='./output/result.png', type=str, help='2')
    args = parser.parse_args()
    drawplotimage(args.csv_filepath, args.outputfilepath)