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
POOL_NUM = 4  #����4���߳�
#��eval��strתΪlist
def __getOriginalValue(value):
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
def getKNNCore(pd_test,pd_train,k_list,result):
    for index_test,row_test in pd_test.iterrows():
        #if index_test%200 == 0:
        print("has predict test doc num:",index_test)
        test_vector = __getOriginalValue(row_test['vector'])
        test_class = __getOriginalValue(row_test['class']).index(1)  #��Ϊÿ���ĵ�ֻ����һ���࣬��˿���ֱ��index
        #������Լ���ѵ�����ľ��룬������distance��������
        pd_distance = pd.DataFrame(data=[[[0],[0]]]*len(pd_train),columns=['class','distance'])   #��һ��Ϊ������𣬵ڶ���Ϊ���룬index����Ӧ��ѵ�������

        #����ѭ���������ʹ�ö��߳�
        for index_train,row_train in pd_train.iterrows():
            #if index_test % 200 == 0 and index_train % 1000 == 0:
                #print('has handle train doc num:', index_train, 'total num:', len(pd_train))
            pd_distance['class'][index_train] = __getOriginalValue(row_train['class'])
            # print("row_train['vector']",row_train['vector'])
            # print(eval(row_train['vector']))
            train_vector = __getOriginalValue(row_train['vector'])
            pd_distance['distance'][index_train] = __calDistance(train_vector,test_vector)  #������������֮��ľ���
        #��distance_list���д���
        pd_distance = pd_distance.sort(columns='distance')  #ָ���н���������Ҫ����ֵ

        #ȡ�������K���ھ�
        neighbor = []
        for k in k_list:
            temp_np = np.array(pd_distance.head(k)['class']) #��Ӧ�����
            neighbor.append(np.argmax(np.sum(temp_np,0)))  #sum�����н�����ͣ�argmax�ҵ����ֵ���ڵ�λ��

        #�������װ
        result.append([test_class,neighbor])

#����ѵ�����Ͳ��Լ�������dataframe���ͣ���ʹ�ö��߳���������Լ�
def KNN(pd_train,pd_test,k_list):
    #����ÿ�����Լ�
    begin = time.time()
    result = Manager().list()  #���̸��Գ���һ�����ݣ�Ĭ���޷��������ݣ����ʹ��manager��listʵ�����ݹ���ע�⣬������������append��get����֮���ȡ�ĵڶ��ַ���
    #result = list()  #����Manager().list()����result�ڲ���˳���޹أ����ֱ��append
    pool = multiprocessing.Pool()  # ����4�����̣����̳�������õ���CPU��������
    multi_data = __getSplitPDData(pd_test,POOL_NUM)  #�ȷֳ�POOL_NUM��
    for data in multi_data:
        pool.apply_async(getKNNCore,(data, pd_train, k_list, result))  # ע�⣬�������ʵ�����ݼ����������������ڶ�������ܣ���Ϊ����̲��ᱬ�������Ĵ���
    pool.close()  # �رս��̳أ���ʾ�����������̳�����ӽ��̣���Ҫ��join֮ǰ���ã�close()��ȴ����е�worker����ִ�н����ٹر�pool,��terminate()����ֱ�ӹر�
    pool.join()  # �ȴ����̳��е����н���ִ�����
    end = time.time()
    print("total time:",end-begin)
    for i in result:
        print("result:",result)
    return result

#�Է��������������
#result_data��ʽΪ��ǰ��Ϊ���Լ���ȷ���࣬����ΪK��Ԥ�����
def evaluation(result_data,class_num,k_list):
    evaluation_result = []
    for k in range(len(k_list)):
        precision,recall = [0] * class_num,[0] * class_num
        true_pos,false_neg,false_pos,true_neg = [],[],[],[]
        for i in range(class_num):  #iΪ��
            tp,fn,fp,tn = 0,0,0,0
            for j in range(len(result_data)):
                #result_data[j][0]Ϊ��ʵ��ǣ�result_data[j][1][k]Ϊ��K��Ԥ����
                if result_data[j][0] == i and result_data[j][1][k] == i: # ��ʵ==Ԥ��==��
                    tp += 1
                if result_data[j][0] == i and result_data[j][1][k] != i: # ��ʵΪ����Ԥ��Ϊ��
                    fn += 1
                if result_data[j][0] != i and result_data[j][1][k] == i: # ��ʵΪ����Ԥ��Ϊ��
                    fp += 1
                if result_data[j][0] != i and result_data[j][1][k] != i: # ��ʵΪ����Ԥ��Ϊ��
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
        macro_f1 = 2*macro_p*macro_r/(macro_p+macro_r)  #��ƽ��
        micro_p = (float(sum(true_pos))/len(true_pos))/(float(sum(true_pos))/len(true_pos)+float(sum(false_pos))/len(false_pos))
        micro_r = (float(sum(true_pos))/len(true_pos))/(float(sum(true_pos))/len(true_pos)+float(sum(false_neg))/len(false_neg))
        micro_f1 = 2*micro_p*micro_r/(micro_p+micro_r)  #΢ƽ��
        evaluation_result.append([macro_f1,micro_f1])
    return evaluation_result


if __name__ == '__main__':
    pd_data = pd.DataFrame(range(55))
    number = 4
    multi_data = __getSplitPDData(pd_data, number)
    for data in multi_data:
        print(data)