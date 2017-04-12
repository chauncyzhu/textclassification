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
    �������K���ھ�
"""
POOL_NUM = 2  #����4���߳�
# ��eval��strתΪlist
def getOriginalValue(value):
    if type(value) == types.StringType:
        return list(eval(value))
    else:
        return list(value)

#��dataframe���ݵȷֳ�number��
def __getSplitPDData(pd_data,number):
    #�Բ��Լ��ָ��POOL_NUM��
    pd_data_len = len(pd_data)
    number = float(number)  #ת�ɸ�����
    split_len = int(math.floor(pd_data_len/number))
    multi_pd_data = [pd_data[m:m+split_len] for m in range(pd_data_len) if m % split_len == 0]
    return multi_pd_data


#�������������ľ��룬ʹ��ŷ�Ͼ���
def __calDistance(vector_one,vector_two):
    #ת��numpy����
    vector_one = np.array(list(vector_one))
    vector_two = np.array(list(vector_two))
    #ά����Ϊһ��
    max_len = np.max([len(vector_one),len(vector_two)])
    #�ں������0
    vector_one = np.append(vector_one,[0]*(max_len-len(vector_one)))
    vector_two = np.append(vector_two, [0] * (max_len - len(vector_two)))
    return np.sqrt(np.sum(np.square(vector_one - vector_two)))

#KNN�ĺ��ģ�ͨ��ѭ�����Լ����ҵ������K��ѵ���������ķ��࣬����result����������result�ǹ������
def knn_core(pd_test,pd_train,k_list,feature_name,result):
    for index_test,row_test in pd_test.iterrows():
        #if index_test%200 == 0:
        print("has predict test doc num:",index_test)
        test_vector = eval(row_test[feature_name])
        test_class = eval(row_test['class']).index(1)  #��Ϊÿ���ĵ�ֻ����һ���࣬��˿���ֱ��index
        #������Լ���ѵ�����ľ��룬������distance��������
        pd_distance = pd.DataFrame(data=[[[0],[0]]]*len(pd_train),columns=['class','distance'])   #��һ��Ϊ������𣬵ڶ���Ϊ���룬index����Ӧ��ѵ�������
        #����ѭ���������ʹ�ö��߳�
        for index_train,row_train in pd_train.iterrows():
            #if index_train % 1000 == 0:
                #print('has handle train doc num:', index_train, 'total num:', len(pd_train))
            pd_distance['class'][index_train] = eval(row_train['class'])
            #print("row_train['vector']",row_train['vector'])
            #print(eval(row_train['vector']))
            train_vector = eval(row_train[feature_name])
            pd_distance['distance'][index_train] = __calDistance(train_vector,test_vector)  #������������֮��ľ���
        #��distance_list���д���
        pd_distance = pd_distance.sort_values(by='distance')  #ָ���н���������Ҫ����ֵ

        #ȡ�������K���ھ�
        neighbor = []
        for k in k_list:
            temp_np = np.array(pd_distance.head(k)['class']) #��Ӧ�����
            neighbor.append(np.argmax(np.sum(temp_np,0)))  #sum�����н�����ͣ�argmax�ҵ����ֵ���ڵ�λ��
        #�������װ
        result.append([test_class,neighbor])

#����ѵ�����Ͳ��Լ�������dataframe���ͣ���ʹ�ö��߳���������Լ�
def knn(pd_train,pd_test,k_list,feature_name):
    #����ÿ�����Լ�
    begin = time.time()
    result = Manager().list()  #���̸��Գ���һ�����ݣ�Ĭ���޷��������ݣ����ʹ��manager��listʵ�����ݹ���ע�⣬������������append��get����֮���ȡ�ĵڶ��ַ���
    #result = list()  #����Manager().list()����result�ڲ���˳���޹أ����ֱ��append
    pool = multiprocessing.Pool()  # ����4�����̣����̳�������õ���CPU��������
    multi_data = __getSplitPDData(pd_test,POOL_NUM)  #�ȷֳ�POOL_NUM��
    for data in multi_data:
        pool.apply_async(knn_core,(data, pd_train, k_list, feature_name, result))  # ע�⣬�������ʵ�����ݼ����������������ڶ�������ܣ���Ϊ����̲��ᱬ�������Ĵ���
    pool.close()  # �رս��̳أ���ʾ�����������̳�����ӽ��̣���Ҫ��join֮ǰ���ã�close()��ȴ����е�worker����ִ�н����ٹر�pool,��terminate()����ֱ�ӹر�
    pool.join()  # �ȴ����̳��е����н���ִ�����
    end = time.time()
    print("total time:",end-begin)
    #��result���д�����֪��Ϊʲô����ListProxy object����list()תΪlist
    return list(result)