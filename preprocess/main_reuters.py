# coding=gbk
import preprocess.data_clean.reuters.clean_data as cd
import preprocess.data_clean.reuters.import_data as id
import preprocess.transfer_vector.voca_dict.voca_data as vd
import preprocess.transfer_vector.generate_vector.bdc as bdc
import preprocess.transfer_vector.generate_vector.transfer_vector as tv
import pandas as pd
"""
   对函数进行调用，下面部分主要是对路透社语料库进行处理
"""
sourcefile = "../data/reuters/reuter_all.xml"   #路透社语料库
train_csv = "../data/reuters/reuter_train.csv"  #分开的训练集
test_csv = "../data/reuters/reuter_test.csv"   #分开的测试集
voca_csv = "../data/reuters/voca_dict.csv"   #字典
feature_voca_csv = "../data/reuters/feature_voca_dict.csv"

class_num = 8
pd_train,pd_test = id.getTrainAndTest(sourcefile,class_num)
#处理训练集
cd.clean_data(pd_train)
#处理测试集
cd.clean_data(pd_test)

#控制训练集个数
#pd_train = pd_train.head(100)
#voca_dict = vd.getRelativeValue(pd_train,vd.getUniqueVocabulary(pd_train),8)   #getUniqueVocabulary比较耗时，存储在csv中
#将词典写入csv文件中
#voca_dict.to_csv(voca_csv)

#根据字典计算BDC值，需要指定index
voca_dict = pd.read_csv(voca_csv,index_col=0)   #注意，都出来之后list变成了字符串，使用的时候用eval转换一下
bdc.getBDCVector(voca_dict,class_num)   #计算bdc值
bdc.getTotalVoca(pd_test,voca_dict)  #将测试集中的特征加入到词典中
print(voca_dict)
voca_dict.to_csv(feature_voca_csv)

#将训练和测试文档转为vsm，vsm存放在vector中
tv.changeToFeatureVector(pd_train,voca_dict,"bdc")
tv.changeToFeatureVector(pd_test,voca_dict,"bdc")
print(pd_train)
print(pd_test)
#写入训练文件中
pd_train.to_csv(train_csv)
#写入测试文件中
pd_test.to_csv(test_csv)