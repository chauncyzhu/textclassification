# coding=gbk
import pandas as pd
from classification.knn.knn import *
from classification.knn.evaluation import *
"""
    ������KNN�ĵ��ú���
"""
#Ĭ�ϲ���
TRAIN_CSV = "../../data/reuters/reuter_train.csv"  #�ֿ���ѵ����
TEST_CSV = "../../data/reuters/reuter_test.csv"   #�ֿ��Ĳ��Լ�
EVALUATION_MULTI_CSV = "../../data/reuters/evaluation_multi_csv.csv"   #���������
EVALUATION_BINARY_CSV = "../../data/reuters/evaluation_binary_csv.csv"   #����������
K_LIST = [50,60,70,80,90,100]
CLASS_NUM = 8   #�������ĸ���
CONFIRM_POS_CLASS = 0  #ָ���������������

#���ݶ�ȡ
PD_TRAIN = pd.read_csv(TRAIN_CSV)
PD_TEST = pd.read_csv(TEST_CSV)

#����������
PD_TEST = PD_TEST.head(50)

#��ö��������
def getMultiClassData():
    return (PD_TRAIN,PD_TEST)  #������ʵû�ж��������Ķ�

#��ö��������ݣ�ֻ��Ҫ�Ķ�class�о���
def getBinaryClassData():
    def f(x):
        if eval(x)[CONFIRM_POS_CLASS] == 1:
            return '[1,0]'  #��������
        else:
            return '[0,1]'  #���ڸ���
    PD_TRAIN['class'] = PD_TRAIN['class'].apply(f)  #apply���ص���һ���µ�Dataframe
    PD_TEST['class'] = PD_TEST['class'].apply(f)
    print(PD_TRAIN)
    print(PD_TEST)
    return (PD_TRAIN,PD_TEST)


if __name__ == '__main__':
    #����KNN
    #��ö���������
    # pd_multi_train,pd_multi_test = getMultiClassData()
    # multi_result = knn(pd_multi_train,pd_multi_test,K_LIST)
    # evaluation_multi_result = pd.DataFrame(evaluation_multiclass(multi_result,CLASS_NUM,K_LIST),columns=['macro_f1','micro_f1'])
    # evaluation_multi_result.to_csv(EVALUATION_MULTI_CSV)

    #��ö����������
    pd_binary_train, pd_binary_test = getBinaryClassData()
    binary_result = knn(pd_binary_train, pd_binary_test, K_LIST)
    evaluation_binary_result = pd.DataFrame(evaluation_binaryclass(binary_result, K_LIST), columns=['precision','recall','f1'])
    evaluation_binary_result.to_csv(EVALUATION_BINARY_CSV)
    print("evaluation_binary_result:",evaluation_binary_result)