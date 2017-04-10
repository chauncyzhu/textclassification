# coding=gbk
import pandas as pd
import knn
"""
    ������KNN�ĵ��ú���
"""
train_csv = "../../data/reuters/reuter_train.csv"  #�ֿ���ѵ����
test_csv = "../../data/reuters/reuter_test.csv"   #�ֿ��Ĳ��Լ�
k_list = [50,60,70,80,90,100]
class_num = 8


#����ѵ�����Ͳ��Լ�����
pd_train = pd.read_csv(train_csv)
pd_test = pd.read_csv(test_csv)

#����ѵ�����Ͳ��Լ�����
pd_test = pd_test.head(50)

if __name__ == '__main__':
    #����KNN
    result = knn.KNN(pd_train,pd_test,k_list)
    evaluation_result = knn.evaluation(result,class_num,k_list)
    print(evaluation_result)