# coding=gbk
import pandas as pd
from classification.knn.knn import *
from classification.knn.evaluation import *
"""
    ������KNN�ĵ��ú���
"""
#Ĭ�ϲ���
EVALUATION_MULTI_BDC_CSV = "../../data/reuters/evaluation_multi_csv.csv"   #���������
EVALUATION_BINARY_DF_BDC_CSV = "../../data/reuters/evaluation_binary_csv.csv"   #����������
K_LIST = [50,60,70,80,90,100]
CLASS_NUM = 8   #�������ĸ���
CONFIRM_POS_CLASS = 0  #ָ���������������

#��ö��������
def get_multi_class_data():
    train_bdc_csv = "../data/reuters/multiclass/reuter_train_multiclass_bdc.csv"  # �ֿ���ѵ����
    test_bdc_csv = "../data/reuters/multiclass/reuter_test_multiclass_bdc.csv"  # �ֿ��Ĳ��Լ�
    train_df_bdc_csv = "../data/reuters/multiclass/reuter_train_multiclass_df_bdc.csv"  # �ֿ���ѵ����
    test_df_bdc_csv = "../data/reuters/multiclass/reuter_test_multiclass_df_bdc.csv"  # �ֿ��Ĳ��Լ�

    # ���ݶ�ȡ
    pd_train = pd.read_csv(train_bdc_csv)
    pd_test = pd.read_csv(test_bdc_csv)

    # ����������
    pd_test = pd_test.head(50)
    return (pd_train,pd_test)  #������ʵû�ж��������Ķ�

#��ö��������ݣ�ֻ��Ҫ�Ķ�class�о���
def get_binary_class_data():
    train_bdc_csv = "../data/reuters/binaryclass/reuter_train_binaryclass_bdc.csv"  # �ֿ���ѵ����
    test_bdc_csv = "../data/reuters/binaryclass/reuter_test_binaryclass_bdc.csv"  # �ֿ��Ĳ��Լ�
    train_df_bdc_csv = "../data/reuters/binaryclass/reuter_train_binaryclass_df_bdc.csv"  # �ֿ���ѵ����
    test_df_bdc_csv = "../data/reuters/binaryclass/reuter_test_binaryclass_df_bdc.csv"  # �ֿ��Ĳ��Լ�

    # ���ݶ�ȡ
    pd_train = pd.read_csv(train_bdc_csv)
    pd_test = pd.read_csv(test_bdc_csv)

    # ����������
    pd_test = pd_test.head(50)
    return (pd_train,pd_test)

if __name__ == '__main__':
    #����KNN
    #��ö���������
    # pd_multi_train,pd_multi_test = getMultiClassData()
    # multi_result = knn(pd_multi_train,pd_multi_test,K_LIST)
    # evaluation_multi_result = pd.DataFrame(evaluation_multiclass(multi_result,CLASS_NUM,K_LIST),columns=['macro_f1','micro_f1'])
    # evaluation_multi_result.to_csv(EVALUATION_MULTI_CSV)

    #��ö����������
    pd_binary_train, pd_binary_test = get_binary_class_data()
    binary_result = knn(pd_binary_train, pd_binary_test, K_LIST)
    evaluation_binary_result = pd.DataFrame(evaluation_binaryclass(binary_result, K_LIST), columns=['precision','recall','f1'])
    evaluation_binary_result.to_csv(EVALUATION_MULTI_BDC_CSV)
    print("evaluation_binary_result:",evaluation_binary_result)