# coding=gbk
import preprocess.data_clean.reuters.clean_data as cd
import preprocess.data_clean.reuters.import_data as id
import preprocess.transfer_vector.voca_dict.voca_data as vd
import preprocess.transfer_vector.generate_vector.bdc as bdc
import preprocess.transfer_vector.generate_vector.transfer_vector as tv
import pandas as pd
"""
   �Ժ������е��ã����沿����Ҫ�Ƕ�·͸�����Ͽ���д���
"""
sourcefile = "../data/reuters/reuter_all.xml"   #·͸�����Ͽ�
train_csv = "../data/reuters/reuter_train.csv"  #�ֿ���ѵ����
test_csv = "../data/reuters/reuter_test.csv"   #�ֿ��Ĳ��Լ�
voca_csv = "../data/reuters/voca_dict.csv"   #�ֵ�
feature_voca_csv = "../data/reuters/feature_voca_dict.csv"

class_num = 8
pd_train,pd_test = id.getTrainAndTest(sourcefile,class_num)
#����ѵ����
cd.clean_data(pd_train)
#������Լ�
cd.clean_data(pd_test)

#����ѵ��������
#pd_train = pd_train.head(100)
#voca_dict = vd.getRelativeValue(pd_train,vd.getUniqueVocabulary(pd_train),8)   #getUniqueVocabulary�ȽϺ�ʱ���洢��csv��
#���ʵ�д��csv�ļ���
#voca_dict.to_csv(voca_csv)

#�����ֵ����BDCֵ����Ҫָ��index
voca_dict = pd.read_csv(voca_csv,index_col=0)   #ע�⣬������֮��list������ַ�����ʹ�õ�ʱ����evalת��һ��
bdc.getBDCVector(voca_dict,class_num)   #����bdcֵ
bdc.getTotalVoca(pd_test,voca_dict)  #�����Լ��е��������뵽�ʵ���
print(voca_dict)
voca_dict.to_csv(feature_voca_csv)

#��ѵ���Ͳ����ĵ�תΪvsm��vsm�����vector��
tv.changeToFeatureVector(pd_train,voca_dict,"bdc")
tv.changeToFeatureVector(pd_test,voca_dict,"bdc")
print(pd_train)
print(pd_test)
#д��ѵ���ļ���
pd_train.to_csv(train_csv)
#д������ļ���
pd_test.to_csv(test_csv)