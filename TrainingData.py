import sys
import argparse
import pickle
import os
import numpy as np
from datetime import datetime
import chainer
import normalizer
import cms_cvae as net

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))


def generate_preprocessor(dim, traindata_filepath_list, normalize_is_binwise):
    stdgauss_normalizer = normalizer.StdGaussNormalizer(dim, binwise=normalize_is_binwise)
    for traindata_filepath in traindata_filepath_list:
        x = np.load(traindata_filepath)
        stdgauss_normalizer.update(x)
    return stdgauss_normalizer


def generate_postprocessor(filepath_list, network, preprocessor):
    stdgauss_normalizer = normalizer.StdGaussNormalizer(1)
    for filepath in filepath_list:
        score = network.test(filepath, preprocessor=preprocessor, k=5)
        stdgauss_normalizer.update(score)
    return stdgauss_normalizer


def main(traindata_filepath_list, output_dir, gpu, batch_size, epoch, validdata_filepath_list, n_filters, k_size, dim, frame, warmup, binwise):
    os.makedirs(output_dir, exist_ok=True)
    preprocessor = generate_preprocessor(dim, traindata_filepath_list, binwise)
    network = net.ConvolutionalVariationalAutoEncoderNetWork(gpu=gpu, n_filters=n_filters, ksize=k_size, dim=dim, frame=frame)
    trained_network = network.train(traindata_filepath_list, output_dir, fileloader=np.load, preprocessor=preprocessor, batch_size=batch_size, epoch=epoch, validdata_filepath_list=validdata_filepath_list, warmup=warmup)
    postprocessor = generate_postprocessor(traindata_filepath_list, network, preprocessor)
    chainer.serializers.save_npz(os.path.join(output_dir, 'network_final.npz'), trained_network)
    with open(os.path.join(output_dir, 'preprocessor.pickle'), 'wb') as f:
        pickle.dump(preprocessor, f)
    with open(os.path.join(output_dir, 'postprocessor.pickle'), 'wb') as f:
        pickle.dump(postprocessor, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train DNN')
    parser.add_argument('traindata_listpath', type=str, help='list file path of train data file paths')
    parser.add_argument('output_dir', type=str, help='directory to output model/log/etc...')
    parser.add_argument('-g', '--gpu', type=int, default=-1, help='GPU Number (-1:CPU)')
    parser.add_argument('-b', '--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('-e', '--epoch', type=int, default=100, help='epoch')
    parser.add_argument('-v', '--validdata_listpath', type=str, default=None, help='list file path of validation data file paths')
    parser.add_argument('-n', '--n_filters', type=int, default=8, help='n_filters')
    parser.add_argument('-k', '--k_size', type=int, default=5, help='k_size')
    parser.add_argument('-d', '--dim', type=int, default=256, help='dim')
    parser.add_argument('-f', '--frame', type=int, default=32, help='count of frames')
    parser.add_argument('-w', '--warmup', type=int, default=10, help='warm-up epoch')
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--binwise', dest='binwise', action='store_true')
    group2.add_argument('--no-binwise', dest='binwise', action='store_false')
    parser.set_defaults(binwise=False)

    args = parser.parse_args()

    start = datetime.now()

    traindata_filepath_list = [filepath.strip() for filepath in open(args.traindata_listpath, 'r')]
    if args.validdata_listpath:
        validdata_filepath_list = [filepath.strip() for filepath in open(args.validdata_listpath, 'r')]
    else:
        validdata_filepath_list = None

    main(traindata_filepath_list, args.output_dir, args.gpu, args.batch_size, args.epoch, validdata_filepath_list,
         args.n_filters, args.k_size, args.dim, args.frame, args.warmup, args.binwise)

    end = datetime.now()
    print(end - start)
