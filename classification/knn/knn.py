# coding=gbk
import numpy as np
import pandas as pd
import types
import time
from threading import Thread
from multiprocessing import Process,Manager
import multiprocessing
import math
"""
    找最近的K个邻居
"""
POOL_NUM = 4  #创建4个线程
#用eval将str转为list
def __getOriginalValue(value):
    if type(value) == types.StringType:
        return list(eval(value))
    else:
        return list(value)

#将dataframe数据等分成number份
def __getSplitPDData(pd_data,number):
    #对测试集分割成POOL_NUM份
    pd_data_len = len(pd_data)
    number = float(number)  #转成浮点数
    split_len = int(math.floor(pd_data_len/number))
    multi_pd_data = [pd_data[m:m+split_len] for m in range(pd_data_len) if m % split_len == 0]
    return multi_pd_data


#计算两个向量的距离，使用欧氏距离
def __calDistance(vector_one,vector_two):
    #转成numpy类型
    vector_one = np.array(list(vector_one))
    vector_two = np.array(list(vector_two))
    #维度置为一样
    max_len = np.max([len(vector_one),len(vector_two)])
    #在后面加上0
    vector_one = np.append(vector_one,[0]*(max_len-len(vector_one)))
    vector_two = np.append(vector_two, [0] * (max_len - len(vector_two)))
    return np.sqrt(np.sum(np.square(vector_one - vector_two)))

#KNN的核心，通过循环测试集来找到最近的K个训练及所属的分类，并用result包含起来，result是共享变量
def getKNNCore(pd_test,pd_train,k_list,result):
    for index_test,row_test in pd_test.iterrows():
        #if index_test%200 == 0:
        print("has predict test doc num:",index_test)
        test_vector = __getOriginalValue(row_test['vector'])
        test_class = __getOriginalValue(row_test['class']).index(1)  #因为每个文档只属于一个类，因此可以直接index
        #计算测试集和训练集的距离，并根据distance进行排序
        pd_distance = pd.DataFrame(data=[[[0],[0]]]*len(pd_train),columns=['class','distance'])   #第一个为所属类别，第二个为距离，index即对应的训练集序号

        #下面循环代码可以使用多线程
        for index_train,row_train in pd_train.iterrows():
            #if index_test % 200 == 0 and index_train % 1000 == 0:
                #print('has handle train doc num:', index_train, 'total num:', len(pd_train))
            pd_distance['class'][index_train] = __getOriginalValue(row_train['class'])
            # print("row_train['vector']",row_train['vector'])
            # print(eval(row_train['vector']))
            train_vector = __getOriginalValue(row_train['vector'])
            pd_distance['distance'][index_train] = __calDistance(train_vector,test_vector)  #计算两个向量之间的距离
        #对distance_list进行处理
        pd_distance = pd_distance.sort(columns='distance')  #指定列进行排序，需要返回值

        #取出最近的K个邻居
        neighbor = []
        for k in k_list:
            temp_np = np.array(pd_distance.head(k)['class']) #对应的类别
            neighbor.append(np.argmax(np.sum(temp_np,0)))  #sum沿着列进行求和，argmax找到最大值所在的位置

        #将结果封装
        result.append([test_class,neighbor])

#传入训练集和测试集，都是dataframe类型，并使用多线程来处理测试集
def KNN(pd_train,pd_test,k_list):
    #对于每个测试集
    begin = time.time()
    result = Manager().list()  #进程各自持有一份数据，默认无法共享数据，因此使用manager的list实现数据共享，注意，这个是在下面的append、get出错之后采取的第二种方法
    #result = list()  #由于Manager().list()，而result内部与顺序无关，因此直接append
    pool = multiprocessing.Pool()  # 创建4个进程，进程池设置最好等于CPU核心数量
    multi_data = __getSplitPDData(pd_test,POOL_NUM)  #等分成POOL_NUM份
    for data in multi_data:
        pool.apply_async(getKNNCore,(data, pd_train, k_list, result))  # 注意，最好在真实的数据集上跑完整个，再在多进程中跑，因为多进程不会爆出完整的错误
    pool.close()  # 关闭进程池，表示不能再往进程池中添加进程，需要在join之前调用，close()会等待池中的worker进程执行结束再关闭pool,而terminate()则是直接关闭
    pool.join()  # 等待进程池中的所有进程执行完毕
    end = time.time()
    print("total time:",end-begin)
    for i in result:
        print("result:",result)
    return result

#对分类进过进行评估
#result_data格式为，前面为测试集正确分类，后面为K个预测分类
def evaluation(result_data,class_num,k_list):
    evaluation_result = []
    for k in range(len(k_list)):
        precision,recall = [0] * class_num,[0] * class_num
        true_pos,false_neg,false_pos,true_neg = [],[],[],[]
        for i in range(class_num):  #i为正
            tp,fn,fp,tn = 0,0,0,0
            for j in range(len(result_data)):
                #result_data[j][0]为真实标记，result_data[j][1][k]为第K个预测标记
                if result_data[j][0] == i and result_data[j][1][k] == i: # 真实==预测==正
                    tp += 1
                if result_data[j][0] == i and result_data[j][1][k] != i: # 真实为正，预测为负
                    fn += 1
                if result_data[j][0] != i and result_data[j][1][k] == i: # 真实为负，预测为正
                    fp += 1
                if result_data[j][0] != i and result_data[j][1][k] != i: # 真实为负，预测为负
                    tn += 1
                true_pos.append(tp)
                false_neg.append(fn)
                false_pos.append(fp)
                true_neg.append(tn)
            if tp+fp == 0:
                p = 0
            else:
                p = float(tp)/(tp+fp)
            if tp+fn == 0:
                r = 0
            else:
                r = float(tp)/(tp+fn)
            precision.append(p)
            recall.append(r)
        macro_p = sum(precision)/class_num
        macro_r = sum(recall)/class_num
        macro_f1 = 2*macro_p*macro_r/(macro_p+macro_r)  #宏平均
        micro_p = (float(sum(true_pos))/len(true_pos))/(float(sum(true_pos))/len(true_pos)+float(sum(false_pos))/len(false_pos))
        micro_r = (float(sum(true_pos))/len(true_pos))/(float(sum(true_pos))/len(true_pos)+float(sum(false_neg))/len(false_neg))
        micro_f1 = 2*micro_p*micro_r/(micro_p+micro_r)  #微平均
        evaluation_result.append([macro_f1,micro_f1])
    return evaluation_result


if __name__ == '__main__':
    pd_data = pd.DataFrame(range(55))
    number = 4
    multi_data = __getSplitPDData(pd_data, number)
    for data in multi_data:
        print(data)