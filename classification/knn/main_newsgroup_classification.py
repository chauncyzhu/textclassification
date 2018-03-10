# coding=gbk
import pandas as pd
from classification.knn.knn import *
from classification.knn.evaluation import *
import utils.newsgroup_path as path
"""
    这里是KNN的调用函数，关于newsgroup，其实和路透社代码相似但是为了方便起见还是另外写一份
"""
#默认参数
K_LIST = [50,60,70,80,90,100]
CLASS_NUM = 6   #多分类类的个数
CONFIRM_POS_CLASS = 0  #指定二分类正类序号

#获得多分类数据
def get_multi_class_data():
    # 数据读取
    #pd_train = pd.read_csv(path.TRAIN_MULTI_BDC_CSV)
    #pd_test = pd.read_csv(path.TEST_MULTI_BDC_CSV)

    #df bdc
    pd_train = pd.read_csv(path.TRAIN_MULTI_DF_BDC_CSV)
    pd_test = pd.read_csv(path.TEST_MULTI_DF_BDC_CSV)

    # 数据量控制
    pd_test = pd_test.head(20)
    return (pd_train,pd_test)  #后面其实没有对数据作改动

#获得二分类数据，只需要改动class列就行
def get_binary_class_data():
    # 数据读取
    #pd_train = pd.read_csv(path.TRAIN_BINARY_BDC_CSV)
    #pd_test = pd.read_csv(path.TEST_BINARY_BDC_CSV)

    # #df bdc
    pd_train = pd.read_csv(path.TRAIN_BINARY_DF_BDC_CSV)
    pd_test = pd.read_csv(path.TEST_BINARY_DF_BDC_CSV)

    # 数据量控制
    #pd_test = pd_test.head(10)
    return (pd_train,pd_test)

if __name__ == '__main__':
    # 获得多分类的数据
    # #feature_name = "bdc"
    # feature_name = "df_bdc"
    # pd_multi_train, pd_multi_test = get_multi_class_data()
    # multi_result = knn(pd_multi_train, pd_multi_test, K_LIST, feature_name)
    # evaluation_multi_result = pd.DataFrame(evaluation_multiclass(multi_result, CLASS_NUM, K_LIST),
    #                                        columns=['macro_f1', 'micro_f1'])
    # # 写入文件中
    # #evaluation_multi_result.to_csv(path.EVALUATION_MULTI_BDC_CSV)
    # evaluation_multi_result.to_csv(path.EVALUATION_MULTI_DF_BDC_CSV)
    # print("evaluation_multi_result:", evaluation_multi_result)

    #获得二分类的数据
    #feature_name = "bdc"
    feature_name = "df_bdc"
    pd_binary_train, pd_binary_test = get_binary_class_data()
    binary_result = knn(pd_binary_train, pd_binary_test, K_LIST,feature_name)
    evaluation_binary_result = pd.DataFrame(evaluation_binaryclass(binary_result, K_LIST), columns=['precision','recall','f1'])
    #写入文件
    #evaluation_binary_result.to_csv(path.EVALUATION_BIANRY_BDC_CSV)
    evaluation_binary_result.to_csv(path.EVALUATION_BINARY_DF_BDC_CSV)
    print("evaluation_binary_result:",evaluation_binary_result)