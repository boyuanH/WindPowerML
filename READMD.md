˵����

ִ�л�����
   python��3.6
   numpy latest
   chainer = 2.0.2

��������д����Ϊ3�����֣�����׼����ģ��ѵ�������������ֱ��Ӧ3��py�ļ�
PrepareData.py
TrainingData.py
ScoreData.py
����ִ��3���ļ�

1. ����׼��
wav�ļ�תΪnpy�ļ�
�����б�
-i ��Ҫ����ģ��ѵ����wav�ļ����ɸ����
-o npy�ļ����Ŀ¼��Ĭ��ֵ��ǰĿ¼
-c ��˷�����ͨ����Ĭ��ֵ0
-n ֡�����ʣ�Ĭ��ֵ441
-t Ŀ�괦���ļ�������Ϊ�գ����Զ��
ex:
 python PrepareData.py -i 001.wav 002.wav -t 003.wav

2. ģ��ѵ��
��ȡ�����б��ļ�npylist.txt��Ѱ�Ҷ�Ӧ��npy�ļ�����ģ��ѵ��
�����б�:
traindata_listpath  npylist.txt��path
output_dir          ģ���ļ������Ŀ¼
ex:
 python TrainingData.py ./npylist.txt .

3. ������
��Ŀ���ļ����д����õ�������
�����б�
input_npy_filepath      Ŀ��npy�ļ�path
model_filepath          ģ���ļ�path
preporocessor_filepath  preprocessor�ļ�path
postprocessor_filepath  postprocessor�ļ�path
output_csv_filepath     ����ļ�
file_name_date          �������ڣ���ʽΪ ��-��-��_ʱ-��-��-����
gpu                     -1
dim                     256
frame                   32

ex:
python ScoreData.py 009.npy network_final.npz preprocessor.pickle postprocessor.pickle outputtest.csv 2018-11-12_13-14-15-321 -1 256 32