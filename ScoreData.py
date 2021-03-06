import argparse
import numpy as np
import configparser
from datetime import datetime
from datetime import timedelta
import cms_cvae
import csv
import os


def create_datelist(date_length, file_name_date, frame):
    u""" create timestamp data list for score list
    the first element is got from npy file name, and add frame*10 msec to the following elements
    """
    # set frame*10 msec to input_millisecond
    input_millisecond = frame * 10
    datelist = []
    count = 0
    while count < date_length:
        if count == 0:
            count += 1
            # put npy file name time & date infomation into the first element
            result_dt = file_name_date
        else:
            count += 1
            # add input_millisecond to the previous one and put it to the following elements
            result_dt += timedelta(milliseconds=input_millisecond)
        # datetime type convert to string as "YYYY-MM-DD HH:MM:SS.sss" and add to datelist
        datelist.append(result_dt.isoformat(sep=' ', timespec='milliseconds'))

    return datelist


def calc_score(input_npy_filepath, model_filepath, preporocessor_filepath, postprocessor_filepath, output_csv_filepath, file_name_date, gpu, dim, frame):
    detection_params = {'gpu': gpu, 'dim': dim, 'frame': frame, 'model_filepath': model_filepath, 'preprocessor_filepath':preporocessor_filepath, 'postprocessor_filepath':postprocessor_filepath}
    # check parameters
    detector = cms_cvae.Detector(detection_params)
    # calculate score
    x = np.load(input_npy_filepath)
    score = detector.calc_score(x)
    # get score list length
    score_length = score.shape[0]
    # create timestamp data list for score list
    time = create_datelist(score_length, file_name_date, frame)
    # stack timestamp and score list
    result = np.vstack((time, score))
    # transpose columns and rows
    result = result.transpose()
    # save result to csv file
    f = open(output_csv_filepath, 'w')
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(result)
    f.close()


def main(input_npy_filepath, model_filepath, preporocessor_filepath, postprocessor_filepath, output_csv_filepath, file_name_date, gpu, dim, frame):
    calc_score(input_npy_filepath, model_filepath, preporocessor_filepath, postprocessor_filepath, output_csv_filepath, file_name_date, gpu, dim, frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='result score')
    parser.add_argument('-i', '--targetwavfile_filepath', required=True, type=str, help='1')
    parser.add_argument('-m', '--model_filepath', type=str, help='2')
    parser.add_argument('-o', '--output_csv_filepath', default='./output/output.csv', type=str, help='5')
    parser.add_argument('-f', '--file_name_date', type=str, help='6')
    parser.add_argument('-c', '--config_file', type=str, default='./conf/ScoreDataConfig.ini', help='6')

    args = parser.parse_args()

    config = configparser.ConfigParser()
    if not os.path.exists(args.config_file):
        print('No config file found')
        exit()
    config.read(args.config_file)
    gpu = int(config.get("ScoreData", "gpu"))
    dim = int(config.get("ScoreData", "dim"))
    frame = int(config.get("ScoreData", "frame"))

    modelsrcpath = ""
    if args.model_filepath is None:
        modelsrcpath = os.path.abspath(os.path.join(args.targetwavfile_filepath, '..'))
    else:
        modelsrcpath = args.model_filepath

    model_filepath = os.path.join(modelsrcpath, 'network_final.npz')
    preporocessor_filepath = os.path.join(modelsrcpath, 'preprocessor.pickle')
    postprocessor_filepath = os.path.join(modelsrcpath, 'postprocessor.pickle')
    if os.path.exists(args.targetwavfile_filepath):
        targetfiles = open(args.targetwavfile_filepath).readlines()
        for targetfile in targetfiles:
            targetfilepath = targetfile.strip()
            if os.path.exists(targetfilepath):
                npyfilename = os.path.split(targetfilepath)[-1]
                modelsrcpathcombin = modelsrcpath.split("\\")

                csvfilename = modelsrcpathcombin[-3] + "-" + modelsrcpathcombin[-2] + "-" + modelsrcpathcombin[-1] + "_" + npyfilename[:-4] + '.csv'

                outputcsvfile = os.path.join(modelsrcpath, modelsrcpathcombin[-3] + "-" + modelsrcpathcombin[-2]
                                             + "-" + modelsrcpathcombin[-1] + "_" + x[:-4] + '.csv')
                datetimesrc = outputcsvfile[-18:-4]
                date_dt = datetime.strptime(datetimesrc[0:4] + '-' + datetimesrc[4:6] + '-'
                                            + datetimesrc[6:8] + '_' + datetimesrc[8:10] + '-'
                                            + datetimesrc[10:12] + '-' + datetimesrc[12:14] + '-' + '000',
                                            '%Y-%m-%d_%H-%M-%S-%f')
                start = datetime.now()
                print("Scoring " + npyfilename + '...')
                main(targetfilepath, model_filepath, preporocessor_filepath, postprocessor_filepath,
                     outputcsvfile, date_dt, gpu, dim, frame)
                end = datetime.now()
                print(end - start)
            else:
                print('Target File list' + targetfilepath + ' not exit')
    else:
        print('Target File list' + args.target_file + ' not exit')




