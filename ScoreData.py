import argparse
import numpy as np
from datetime import datetime
from datetime import timedelta
import cms_cvae
import csv


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
    parser.add_argument('input_npy_filepath', type=str, help='1')
    parser.add_argument('model_filepath', type=str, help='2')
    parser.add_argument('preporocessor_filepath', type=str, help='3')
    parser.add_argument('postprocessor_filepath', type=str, help='4')
    parser.add_argument('output_csv_filepath', type=str, help='5')
    parser.add_argument('file_name_date', type=str, help='6')
    parser.add_argument('gpu', type=int, help='7')
    parser.add_argument('dim', type=int, help='8')
    parser.add_argument('frame', type=int, help='9')

    args = parser.parse_args()

    date_dt = datetime.strptime(args.file_name_date, '%Y-%m-%d_%H-%M-%S-%f')
    start = datetime.now()
    main(args.input_npy_filepath, args.model_filepath, args.preporocessor_filepath, args.postprocessor_filepath,
         args.output_csv_filepath, date_dt, args.gpu, args.dim, args.frame)
    end = datetime.now()
    print(end - start)
