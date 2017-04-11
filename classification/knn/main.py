# coding=gbk
import pandas as pd
from classification.knn.knn import *
from classification.knn.evaluation import *
"""
    这里是KNN的调用函数
"""
#默认参数
TRAIN_CSV = "../../data/reuters/reuter_train.csv"  #分开的训练集
TEST_CSV = "../../data/reuters/reuter_test.csv"   #分开的测试集
EVALUATION_MULTI_CSV = "../../data/reuters/evaluation_multi_csv.csv"   #多分类结果集
EVALUATION_BINARY_CSV = "../../data/reuters/evaluation_binary_csv.csv"   #二分类结果集
K_LIST = [50,60,70,80,90,100]
CLASS_NUM = 8   #多分类类的个数
CONFIRM_POS_CLASS = 0  #指定二分类正类序号

#数据读取
PD_TRAIN = pd.read_csv(TRAIN_CSV)
PD_TEST = pd.read_csv(TEST_CSV)

#数据量控制
PD_TEST = PD_TEST.head(50)

#获得多分类数据
def getMultiClassData():
    return (PD_TRAIN,PD_TEST)  #后面其实没有对数据作改动

#获得二分类数据，只需要改动class列就行
def getBinaryClassData():
    def f(x):
        if eval(x)[CONFIRM_POS_CLASS] == 1:
            return '[1,0]'  #属于正类
        else:
            return '[0,1]'  #属于负类
    PD_TRAIN['class'] = PD_TRAIN['class'].apply(f)  #apply返回的是一个新的Dataframe
    PD_TEST['class'] = PD_TEST['class'].apply(f)
    print(PD_TRAIN)
    print(PD_TEST)
    return (PD_TRAIN,PD_TEST)


if __name__ == '__main__':
    #调用KNN
    #获得多分类的数据
    # pd_multi_train,pd_multi_test = getMultiClassData()
    # multi_result = knn(pd_multi_train,pd_multi_test,K_LIST)
    # evaluation_multi_result = pd.DataFrame(evaluation_multiclass(multi_result,CLASS_NUM,K_LIST),columns=['macro_f1','micro_f1'])
    # evaluation_multi_result.to_csv(EVALUATION_MULTI_CSV)

    #获得二分类的数据
    pd_binary_train, pd_binary_test = getBinaryClassData()
    binary_result = knn(pd_binary_train, pd_binary_test, K_LIST)
    evaluation_binary_result = pd.DataFrame(evaluation_binaryclass(binary_result, K_LIST), columns=['precision','recall','f1'])
    evaluation_binary_result.to_csv(EVALUATION_BINARY_CSV)
    print("evaluation_binary_result:",evaluation_binary_result)