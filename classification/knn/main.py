# coding=gbk
import pandas as pd
from classification.knn.knn import *
from classification.knn.evaluation import *
"""
    这里是KNN的调用函数
"""
#默认参数
EVALUATION_MULTI_BDC_CSV = "../../data/reuters/evaluation_multi_csv.csv"   #多分类结果集
EVALUATION_BINARY_DF_BDC_CSV = "../../data/reuters/evaluation_binary_csv.csv"   #二分类结果集
K_LIST = [50,60,70,80,90,100]
CLASS_NUM = 8   #多分类类的个数
CONFIRM_POS_CLASS = 0  #指定二分类正类序号

#获得多分类数据
def get_multi_class_data():
    train_bdc_csv = "../data/reuters/multiclass/reuter_train_multiclass_bdc.csv"  # 分开的训练集
    test_bdc_csv = "../data/reuters/multiclass/reuter_test_multiclass_bdc.csv"  # 分开的测试集
    train_df_bdc_csv = "../data/reuters/multiclass/reuter_train_multiclass_df_bdc.csv"  # 分开的训练集
    test_df_bdc_csv = "../data/reuters/multiclass/reuter_test_multiclass_df_bdc.csv"  # 分开的测试集

    # 数据读取
    pd_train = pd.read_csv(train_bdc_csv)
    pd_test = pd.read_csv(test_bdc_csv)

    # 数据量控制
    pd_test = pd_test.head(50)
    return (pd_train,pd_test)  #后面其实没有对数据作改动

#获得二分类数据，只需要改动class列就行
def get_binary_class_data():
    train_bdc_csv = "../data/reuters/binaryclass/reuter_train_binaryclass_bdc.csv"  # 分开的训练集
    test_bdc_csv = "../data/reuters/binaryclass/reuter_test_binaryclass_bdc.csv"  # 分开的测试集
    train_df_bdc_csv = "../data/reuters/binaryclass/reuter_train_binaryclass_df_bdc.csv"  # 分开的训练集
    test_df_bdc_csv = "../data/reuters/binaryclass/reuter_test_binaryclass_df_bdc.csv"  # 分开的测试集

    # 数据读取
    pd_train = pd.read_csv(train_bdc_csv)
    pd_test = pd.read_csv(test_bdc_csv)

    # 数据量控制
    pd_test = pd_test.head(50)
    return (pd_train,pd_test)

if __name__ == '__main__':
    #调用KNN
    #获得多分类的数据
    # pd_multi_train,pd_multi_test = getMultiClassData()
    # multi_result = knn(pd_multi_train,pd_multi_test,K_LIST)
    # evaluation_multi_result = pd.DataFrame(evaluation_multiclass(multi_result,CLASS_NUM,K_LIST),columns=['macro_f1','micro_f1'])
    # evaluation_multi_result.to_csv(EVALUATION_MULTI_CSV)

    #获得二分类的数据
    pd_binary_train, pd_binary_test = get_binary_class_data()
    binary_result = knn(pd_binary_train, pd_binary_test, K_LIST)
    evaluation_binary_result = pd.DataFrame(evaluation_binaryclass(binary_result, K_LIST), columns=['precision','recall','f1'])
    evaluation_binary_result.to_csv(EVALUATION_MULTI_BDC_CSV)
    print("evaluation_binary_result:",evaluation_binary_result)