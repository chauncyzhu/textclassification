# coding=gbk
import pandas as pd
from classification.knn.knn import *
from classification.knn.evaluation import *
import utils.reuters_path as path
"""
    ������KNN�ĵ��ú���
"""
#Ĭ�ϲ���
K_LIST = [50,60,70,80,90,100]
CLASS_NUM = 8   #�������ĸ���

#��ö��������
def get_multi_class_data():
    # ���ݶ�ȡ
    pd_train = pd.read_csv(path.TRAIN_MULTI_BDC_CSV)
    pd_test = pd.read_csv(path.TEST_MULTI_BDC_CSV)

    #df bdc
    #pd_train = pd.read_csv(path.TRAIN_MULTI_DF_BDC_CSV)
    #pd_test = pd.read_csv(path.TEST_MULTI_DF_BDC_CSV)

    # ����������
    pd_test = pd_test.head(50)
    return (pd_train,pd_test)  #������ʵû�ж��������Ķ�

#��ö��������ݣ�ֻ��Ҫ�Ķ�class�о���
def get_binary_class_data():
    # ���ݶ�ȡ
    #pd_train = pd.read_csv(path.TRAIN_BINARY_BDC_CSV)
    #pd_test = pd.read_csv(path.TEST_BINARY_BDC_CSV)

    # #df bdc
    pd_train = pd.read_csv(path.TRAIN_BINARY_DF_BDC_CSV)
    pd_test = pd.read_csv(path.TEST_BINARY_DF_BDC_CSV)

    #����ָ��������ı�class


    # ����������
    pd_test = pd_test.head(10)
    return (pd_train,pd_test)

if __name__ == '__main__':
    #��ö���������
    feature_name = "bdc"
    #feature_name = "df_bdc"
    pd_multi_train,pd_multi_test = get_multi_class_data()
    multi_result = knn(pd_multi_train,pd_multi_test,K_LIST,feature_name)
    evaluation_multi_result = pd.DataFrame(evaluation_multiclass(multi_result,CLASS_NUM,K_LIST),columns=['macro_f1','micro_f1'])
    #д���ļ���
    evaluation_multi_result.to_csv(path.EVALUATION_MULTI_BDC_CSV)
    #evaluation_multi_result.to_csv(path.EVALUATION_MULTI_DF_BDC_CSV)
    print("evaluation_multi_result:", evaluation_multi_result)

    #���������
    multi_result = pd.DataFrame(multi_result)
    multi_result.to_csv(path.MULTI_BDC_CSV)

    #��ö����������
    # feature_name = "bdc"
    # #feature_name = "df_bdc"
    # pd_binary_train, pd_binary_test = get_binary_class_data()
    # binary_result = knn(pd_binary_train, pd_binary_test, K_LIST,feature_name)
    # evaluation_binary_result = pd.DataFrame(evaluation_binaryclass(binary_result, K_LIST), columns=['precision','recall','f1'])
    # #д���ļ�
    # evaluation_binary_result.to_csv(path.EVALUATION_BIANRY_BDC_CSV)
    # #evaluation_binary_result.to_csv(path.EVALUATION_BINARY_DF_BDC_CSV)
    # print("evaluation_binary_result:",evaluation_binary_result)