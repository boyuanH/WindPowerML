# -*- coding: utf-8 -*-

import six
import numpy as np
import pickle
import inspect

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import training
from chainer.training import extensions
from chainer import variable
from chainer import cuda
from chainer import reporter as reporter_module

import utility
import normalizer


class ConvolutionalVariationalAutoencoder(chainer.Chain):
    def __init__(self, n_channels, n_filters=8, ksize=5, task_is_train=True):
        self.dtype = np.float32
        W = chainer.initializers.HeNormal(1 / np.sqrt(2), self.dtype)
        bias = chainer.initializers.Zero(self.dtype)
        self.pooling_ksize = 2
        self.pooling_stride = 2
        self.h1_size = None
        self.h2_size = None
        self.h3_size = None
        super(ConvolutionalVariationalAutoencoder, self).__init__(
            # encoder
            enc_conv1=L.Convolution2D(n_channels, out_channels=n_filters, ksize=ksize, pad=ksize // 2,
                                      initialW=W, initial_bias=bias),
            enc_bn1=L.BatchNormalization(n_filters),
            enc_conv2=L.Convolution2D(n_filters, n_filters // 2, ksize=ksize, pad=ksize // 2,
                                      initialW=W, initial_bias=bias),
            enc_bn2=L.BatchNormalization(n_filters // 2),
            enc_conv3=L.Convolution2D(n_filters // 2, n_filters // 2, ksize=ksize, pad=ksize // 2,
                                      initialW=W, initial_bias=bias),
            enc_bn3=L.BatchNormalization(n_filters // 2),
            enc_conv4_mu=L.Convolution2D(n_filters // 2, n_filters // 2, ksize=ksize, pad=ksize // 2,
                                         initialW=W, initial_bias=bias),
            enc_conv4_ln_var=L.Convolution2D(n_filters // 2, n_filters // 2, ksize=ksize, pad=ksize // 2,
                                             initialW=W, initial_bias=bias),
            # decoder
            dec_conv1=L.Convolution2D(n_filters // 2, n_filters // 2, ksize=ksize, pad=ksize // 2,
                                      initialW=W, initial_bias=bias),
            dec_bn1=L.BatchNormalization(n_filters // 2),
            dec_conv2=L.Convolution2D(n_filters // 2, n_filters // 2, ksize=ksize, pad=ksize // 2,
                                      initialW=W, initial_bias=bias),
            dec_bn2=L.BatchNormalization(n_filters // 2),
            dec_conv3=L.Convolution2D(n_filters // 2, n_filters, ksize=ksize, pad=ksize // 2,
                                      initialW=W, initial_bias=bias),
            dec_bn3=L.BatchNormalization(n_filters),
            dec_conv4_mu=L.Convolution2D(n_filters, n_channels, ksize=ksize, pad=ksize // 2,
                                         initialW=W, initial_bias=bias),
            dec_conv4_ln_var=L.Convolution2D(n_filters, n_channels, ksize=ksize, pad=ksize // 2,
                                             initialW=W, initial_bias=bias)
        )
        self.p1 = F.MaxPooling2D(2, 2)
        self.p2 = F.MaxPooling2D(2, 2)
        self.p3 = F.MaxPooling2D(2, 2)
        self.train = task_is_train

    def __call__(self, x):
        mu_e, ln_var_e = self.encode(x)
        z = F.gaussian(mu_e, ln_var_e)
        mu_d, ln_var_d = self.decode(z)
        return mu_d, ln_var_d

    def encode(self, x):
        with chainer.using_config('train', self.train):
            h = F.relu(self.enc_bn1(self.enc_conv1(x)))
            self.h1_size = h.shape
            with chainer.using_config('use_cudnn', 'never'):
                h = self.p1(h)
            h = F.relu(self.enc_bn2(self.enc_conv2(h)))
            self.h2_size = h.shape
            with chainer.using_config('use_cudnn', 'never'):
                h = self.p2(h)
            h = F.tanh(self.enc_bn3(self.enc_conv3(h)))
            self.h3_size = h.shape
            with chainer.using_config('use_cudnn', 'never'):
                h = self.p3(h)
        return self.enc_conv4_mu(h), self.enc_conv4_ln_var(h)

    def decode(self, z):
        with chainer.using_config('train', self.train):
            h = F.relu(self.dec_bn1(self.dec_conv1(z)))
            h = F.upsampling_2d(h, self.p3.indexes, self.p3.kh, self.p3.sy, self.p3.ph, self.h3_size[2:])
            h = F.relu(self.dec_bn2(self.dec_conv2(h)))
            h = F.upsampling_2d(h, self.p2.indexes, self.p2.kh, self.p2.sy, self.p2.ph, self.h2_size[2:])
            h = F.tanh(self.dec_bn3(self.dec_conv3(h)))
            h = F.upsampling_2d(h, self.p1.indexes, self.p1.kh, self.p1.sy, self.p1.ph, self.h1_size[2:])
        return self.dec_conv4_mu(h), self.dec_conv4_ln_var(h)


class FileWiseDataset(chainer.dataset.DatasetMixin):
    def __init__(self, filepath_list, dim, frame, fileloader, preprocessor):
        self._x = []
        self._data_num = []
        self._frame = frame
        for i, filepath in enumerate(filepath_list):
            data = fileloader(filepath)
            if preprocessor:
                data = preprocessor(data)
            data = np.reshape(data[:data.shape[0] // frame * frame], (-1, 1, frame, data.shape[1]))
            self._x.extend(data)

    def __len__(self):
        return len(self._x)

    def get_example(self, i):
        return self._x[i], self._x[i]


class WUUpdater(training.StandardUpdater):
    def __init__(self, train_iter, optimizer, warmup=0, device=None):
        super(WUUpdater, self).__init__(train_iter, optimizer, device=device)
        self.warmup = warmup

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)

        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target

        beta = 1.0
        if self.warmup > 0 and self.epoch < self.warmup:
            beta = float(self.epoch) / float(self.warmup)

        if isinstance(in_arrays, tuple):
            in_vars = tuple(variable.Variable(x) for x in in_arrays)
            optimizer.update(loss_func, *in_vars, beta=beta)
        elif isinstance(in_arrays, dict):
            in_vars = {key: variable.Variable(x)
                       for key, x in six.iteritems(in_arrays)}
            optimizer.update(loss_func, **in_vars, beta=beta)
        else:
            in_var = variable.Variable(in_arrays)
            optimizer.update(loss_func, in_var, beta=beta)


class CVAEReconstructor(chainer.link.Chain):
    def __init__(self, predictor, k=1):
        super(CVAEReconstructor, self).__init__(predictor=predictor)
        self.loss = None
        self.k = k

    def __call__(self, *args, beta=1.0):
        assert len(args) >= 2
        x = args[:-1]
        t = args[-1]
        mu_e, ln_var_e = self.predictor.encode(*x)
        batchsize = len(mu_e.data)
        rec_loss = 0
        for l in six.moves.range(self.k):
            z = F.gaussian(mu_e, ln_var_e)
            mu_d, ln_var_d = self.predictor.decode(z)
            rec_loss += F.gaussian_nll(t, mu_d, ln_var_d) / (self.k * batchsize)
        kl_loss = beta * F.gaussian_kl_divergence(mu_e, ln_var_e) / batchsize
        self.loss = rec_loss + kl_loss
        reporter_module.report({'loss': self.loss}, self)
        return self.loss


class ConvolutionalVariationalAutoEncoderNetWork:
    def __init__(self, model_filepath=None, gpu=-1, n_filters=16, ksize=3, dim=512, frame=32, task_is_train=True):
        self.dim = dim
        self.frame = frame
        self.network = ConvolutionalVariationalAutoencoder(n_channels=1, n_filters=n_filters, ksize=ksize,
                                                           task_is_train=task_is_train)
        if model_filepath:
            chainer.serializers.load_npz(model_filepath, self.network)

        self.model = CVAEReconstructor(self.network)
        self.model.compute_accuracy = False

        self.gpu = gpu
        if self.gpu >= 0:
            chainer.cuda.get_device(self.gpu).use()
            self.model.to_gpu()

    def train(self, traindata_filepath_list, output_dir,
              fileloader=np.load, preprocessor=None, batch_size=20, epoch=100, validdata_filepath_list=None, warmup=0):
        train_dataset = FileWiseDataset(traindata_filepath_list, self.dim, self.frame, fileloader, preprocessor)
        train_iter = chainer.iterators.SerialIterator(train_dataset, batch_size)
        optimizer = optimizers.Adam()
        optimizer.setup(self.model)
        updater = WUUpdater(train_iter, optimizer, warmup=warmup, device=self.gpu)
        trainer = training.Trainer(updater, (epoch, 'epoch'), out=output_dir)

        evaluator = None
        if validdata_filepath_list:
            valid_dataset = FileWiseDataset(validdata_filepath_list, self.dim, self.frame, fileloader, preprocessor)
            valid_iter = chainer.iterators.SerialIterator(valid_dataset, batch_size, repeat=False, shuffle=False)

            valid_model = self.model.copy()
            valid_rnn = valid_model.predictor
            valid_rnn.train = False

            evaluator = extensions.Evaluator(valid_iter, valid_model, device=self.gpu)

        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.ProgressBar(update_interval=10))
        if evaluator:
            trainer.extend(evaluator)
            trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))
        else:
            trainer.extend(extensions.PrintReport(['epoch', 'main/loss']))

        trainer.run()

        return self.network

    def test(self, test_filepath, fileloader=np.load, preprocessor=None, postprocessor=None, k=1):
        x = fileloader(test_filepath)
        if preprocessor:
            x = preprocessor(x)
        x = np.reshape(x[:x.shape[0] // self.frame * self.frame], (-1, 1, self.frame, x.shape[1]))
        if self.gpu >= 0:
            x = chainer.cuda.to_gpu(x, device=self.gpu)
        mu_e, ln_var_e = self.network.encode(x)
        batchsize = len(mu_e.data)
        loss = 0
        for l in range(k):
            z = F.gaussian(mu_e, ln_var_e)
            mu_d, ln_var_d = self.network.decode(z)
            loss += (F.gaussian_nll(chainer.Variable(x), mu_d, ln_var_d, reduce='no') / k).data
        xp = cuda.get_array_module(loss)
        loss = xp.sum(loss, axis=(1, 2, 3))
        loss = cuda.to_cpu(loss)
        if postprocessor:
            loss = postprocessor(loss)
        return loss

    def output_raw(self, test_filepath, fileloader=np.load, preprocessor=None):
        x = fileloader(test_filepath)
        if preprocessor:
            x = preprocessor(x)
        reshaped_x = np.reshape(x[:x.shape[0] // self.frame * self.frame], (-1, 1, self.frame, x.shape[1]))
        if self.gpu >= 0:
            reshaped_x = chainer.cuda.to_gpu(reshaped_x, device=self.gpu)
        mu_e, ln_var_e = self.network.encode(reshaped_x)
        z = F.gaussian(mu_e, ln_var_e)
        mu_d, ln_var_d = self.network.decode(z)

        xp = cuda.get_array_module(mu_d.data)
        mu_d = xp.reshape(mu_d.data, (-1, x.shape[1]))
        ln_var_d = xp.reshape(ln_var_d.data, (-1, x.shape[1]))
        loss = F.gaussian_nll(chainer.Variable(reshaped_x.reshape((-1, x.shape[1]))),
                              chainer.Variable(mu_d),
                              chainer.Variable(ln_var_d), reduce='no')

        return x, chainer.cuda.to_cpu(mu_d), chainer.cuda.to_cpu(ln_var_d), chainer.cuda.to_cpu(loss.data)

    def encode(self, test_filepath, fileloader=np.load, preprocessor=None):
        x = fileloader(test_filepath)
        if preprocessor:
            x = preprocessor(x)
        x = np.reshape(x[:x.shape[0] // self.frame * self.frame], (-1, 1, self.frame, x.shape[1]))
        if self.gpu >= 0:
            x = chainer.cuda.to_gpu(x, device=self.gpu)
        mu_e, ln_var_e = self.network.encode(x)

        return mu_e.data, ln_var_e.data

    def reconst(self, x, preprocessor=None, postprocessor=None, k=1):
        if preprocessor:
            x = preprocessor(x)
        x = np.reshape(x[:x.shape[0] // self.frame * self.frame], (-1, 1, self.frame, x.shape[1]))
        if self.gpu >= 0:
            x = cuda.to_gpu(x, device=self.gpu)
        xp = cuda.get_array_module(x)
        loss = xp.zeros(len(x))
        # ここはメモリ節約のためにわざわざ分割計算している
        for i in np.arange(len(x)):
            x_batch = xp.reshape(x[i], (-1, 1, self.frame, self.dim))
            mu_e, ln_var_e = self.network.encode(x_batch)
            mu_e.unchain_backward()
            ln_var_e.unchain_backward()
            for l in range(k):
                z = F.gaussian(mu_e, ln_var_e)
                z.unchain_backward()
                mu_d, ln_var_d = self.network.decode(z)
                mu_d.unchain_backward()
                ln_var_d.unchain_backward()
                loss[i] += (F.gaussian_nll(chainer.Variable(x_batch), mu_d, ln_var_d) / k).data
        loss = cuda.to_cpu(loss)
        if postprocessor:
            loss = postprocessor(loss)
        return loss


class Detector:
    def __init__(self, params):
        params = self.__init_params_check(params)

        self.gpu = params['gpu']
        self.dim = params['dim']
        self.frame = params['frame']
        self.model_filepath = params['model_filepath']
        self.preprocessor_filepath = params['preprocessor_filepath']
        self.postprocessor_filepath = params['postprocessor_filepath']

        self.preprocessor = None
        if self.preprocessor_filepath:
            with open(self.preprocessor_filepath, 'rb') as f:
                self.preprocessor = pickle.load(f)

        self.postprocessor = None
        if self.postprocessor_filepath:
            with open(self.postprocessor_filepath, 'rb') as f:
                self.postprocessor = pickle.load(f)

        self.net = ConvolutionalVariationalAutoEncoderNetWork(model_filepath=self.model_filepath, gpu=self.gpu,
                                                              n_filters=8, ksize=5, dim=self.dim, frame=self.frame,
                                                              task_is_train=False)

    def __init_params_check(self, params):
        if 'gpu' not in params:
            params['gpu'] = -1
        if 'dim' not in params:
            raise ValueError(utility.error_message('{}.{}()'.format(self.__class__.__name__,
                                                                    inspect.getframeinfo(inspect.currentframe())[2]),
                                                   'dim', 'empty',
                                                   'must set dimension value'))
        if 'frame' not in params:
            raise ValueError(utility.error_message('{}.{}()'.format(self.__class__.__name__,
                                                                    inspect.getframeinfo(inspect.currentframe())[2]),
                                                   'frame', 'empty',
                                                   'must set frame value'))
        if 'model_filepath' not in params:
            raise ValueError(utility.error_message('{}.{}()'.format(self.__class__.__name__,
                                                                    inspect.getframeinfo(inspect.currentframe())[2]),
                                                   'model_filepath', 'empty',
                                                   'must set model filepath'))
        if 'preprocessor_filepath' not in params:
            params['preprocessor_filepath'] = None
        if 'postprocessor_filepath' not in params:
            params['postprocessor_filepath'] = None
        if not isinstance(params['gpu'], int):
            raise TypeError(utility.error_message('{}.{}()'.format(self.__class__.__name__,
                                                                   inspect.getframeinfo(inspect.currentframe())[2]),
                                                  'gpu', params['gpu'],
                                                  'expected type is int'))
        if not isinstance(params['dim'], int):
            raise TypeError(utility.error_message('{}.{}()'.format(self.__class__.__name__,
                                                                   inspect.getframeinfo(inspect.currentframe())[2]),
                                                  'dim', params['dim'],
                                                  'expected type is int'))
        if not isinstance(params['frame'], int):
            raise TypeError(utility.error_message('{}.{}()'.format(self.__class__.__name__,
                                                                   inspect.getframeinfo(inspect.currentframe())[2]),
                                                  'frame', params['frame'],
                                                  'expected type is int'))
        if not isinstance(params['model_filepath'], str):
            raise TypeError(utility.error_message('{}.{}()'.format(self.__class__.__name__,
                                                                   inspect.getframeinfo(inspect.currentframe())[2]),
                                                  'model_filepath', params['model_filepath'],
                                                  'expected type is str'))
        if params['preprocessor_filepath'] and (not isinstance(params['preprocessor_filepath'], str)):
            raise TypeError(utility.error_message('{}.{}()'.format(self.__class__.__name__,
                                                                   inspect.getframeinfo(inspect.currentframe())[2]),
                                                  'preprocessor_filepath', params['preprocessor_filepath'],
                                                  'expected type is str'))
        if params['postprocessor_filepath'] and (not isinstance(params['postprocessor_filepath'], str)):
            raise TypeError(utility.error_message('{}.{}()'.format(self.__class__.__name__,
                                                                   inspect.getframeinfo(inspect.currentframe())[2]),
                                                  'postprocessor_filepath', params['postprocessor_filepath'],
                                                  'expected type is str'))
        if params['dim'] < 1:
            raise ValueError(utility.error_message('{}.{}()'.format(self.__class__.__name__,
                                                                    inspect.getframeinfo(inspect.currentframe())[2]),
                                                   'dim', params['dim'],
                                                   'expected value >= 1'))
        if params['frame'] < 1:
            raise ValueError(utility.error_message('{}.{}()'.format(self.__class__.__name__,
                                                                    inspect.getframeinfo(inspect.currentframe())[2]),
                                                   'frame', params['frame'],
                                                   'expected value >= 1'))
        return params

    def calc_score(self, x):
        score = self.net.reconst(x, preprocessor=self.preprocessor, postprocessor=self.postprocessor, k=1)
        return score
