说明：

执行环境：
   python≥3.6
   numpy latest
   chainer = 2.0.2

将代码重写整合为3个部分，数据准备，模型训练，结果输出。分别对应3个py文件
PrepareData.py
TrainingData.py
ScoreData.py
依次执行3个文件

1. 数据准备
wav文件转为npy文件
参数列表：
-i 需要用于模型训练的wav文件，可跟多个
-o npy文件输出目录，默认值当前目录
-c 麦克风阵列通道，默认值0
-n 帧采样率，默认值441
-t 目标处理文件，可以为空，可以多个
ex:
 python PrepareData.py -i 001.wav 002.wav -t 003.wav

2. 模型训练
读取参数列表文件npylist.txt，寻找对应的npy文件进行模型训练
参数列表:
traindata_listpath  npylist.txt的path
output_dir          模型文件的输出目录
ex:
 python TrainingData.py ./npylist.txt .

3. 结果输出
对目标文件进行处理，得到输出结果
参数列表：
input_npy_filepath      目标npy文件path
model_filepath          模型文件path
preporocessor_filepath  preprocessor文件path
postprocessor_filepath  postprocessor文件path
output_csv_filepath     结果文件
file_name_date          输入日期，格式为 年-月-日_时-分-秒-毫秒
gpu                     -1
dim                     256
frame                   32

ex:
python ScoreData.py 009.npy network_final.npz preprocessor.pickle postprocessor.pickle outputtest.csv 2018-11-12_13-14-15-321 -1 256 32