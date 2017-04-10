# coding=gbk
import pandas as pd
from classification.knn.knn import *
"""
    这里是KNN的调用函数
"""
train_csv = "../../data/reuters/reuter_train.csv"  #分开的训练集
test_csv = "../../data/reuters/reuter_test.csv"   #分开的测试集
evaluation_csv = "../../data/reuters/evaluation_csv.csv"   #结果集
k_list = [1,2,3,4,5,6,7,8,9,10]
class_num = 8


#导入训练集和测试集数据
pd_train = pd.read_csv(train_csv)
pd_test = pd.read_csv(test_csv)

#控制训练集和测试集数量
pd_test = pd_test.head(50)

if __name__ == '__main__':
    #调用KNN
    result = KNN(pd_train,pd_test,k_list)
    evaluation_result = pd.DataFrame(evaluation(result,class_num,k_list),columns=['macro_f1','micro_f1'])
    evaluation_result.to_csv(evaluation_csv)
    print(evaluation_result)