# coding=gbk
import pandas as pd
from classification.knn.knn import *
"""
    ������KNN�ĵ��ú���
"""
train_csv = "../../data/reuters/reuter_train.csv"  #�ֿ���ѵ����
test_csv = "../../data/reuters/reuter_test.csv"   #�ֿ��Ĳ��Լ�
evaluation_csv = "../../data/reuters/evaluation_csv.csv"   #�����
k_list = [1,2,3,4,5,6,7,8,9,10]
class_num = 8


#����ѵ�����Ͳ��Լ�����
pd_train = pd.read_csv(train_csv)
pd_test = pd.read_csv(test_csv)

#����ѵ�����Ͳ��Լ�����
pd_test = pd_test.head(50)

if __name__ == '__main__':
    #����KNN
    result = KNN(pd_train,pd_test,k_list)
    evaluation_result = pd.DataFrame(evaluation(result,class_num,k_list),columns=['macro_f1','micro_f1'])
    evaluation_result.to_csv(evaluation_csv)
    print(evaluation_result)