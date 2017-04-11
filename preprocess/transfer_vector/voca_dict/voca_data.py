# coding=gbk
import pandas as pd
import time
import copy
import numpy as np
"""
    �ҳ�����Ԥ�����dataframe��Ӧ���ֵ��б�
"""
#��ȡdata��Ӧ���ֵ䣬����������Ե�Ƶ����һ��ɸѡ
def getUniqueVocabulary(data):
    w = []  # �����д���������һ��
    for i in data['content']:
        w.extend(i)
    vocabulary = pd.DataFrame(pd.Series(w).value_counts(),columns=['tf'])  #�����������tfֵ
    vocabulary = vocabulary[vocabulary['tf'] > 10]   #ȥ��ĳЩ��Ƶ��
    vocabulary = vocabulary[vocabulary['tf'] < 4000]  # ȥ��ĳЩ��Ƶ��
    return vocabulary  #pd.DataFrame

#������Ҫͳ�����ֵ��dataframe�����Ӧ���ֵ�����
def getRelativeValue(pd_data,voca_data,class_num):
    #get:
    # number of all vocabulary in a class [��1�ܴ�������2�ܴ���,.....]
    # number of every vocabulary in every class [[��1�ô��������,��2�ô��������...]��[��1�ô��������,��2�ô��������...],...]
    # number of documents contains every vocabulary in every class [[��1���ָô�����ĵ���,��2���ָô�����ĵ���...]��[��1���ָô�����ĵ���,��2���ָô�����ĵ���...],...]
    # number of documents in every class [��1���ĵ�������2���ĵ���,.....]
    #but first, we need all vocabulary list
    #���еĴʻ�
    voca = voca_data.index
    temp = [0]*class_num
    word_appear_set = []
    class_word_appear_set = copy.deepcopy(temp)
    word_doc_set = []
    doc_class_set = copy.deepcopy(temp)
    for i in voca:
        word_appear_set.append(copy.deepcopy(temp))
        word_doc_set.append(copy.deepcopy(temp))
    print 'calculating'
    begin = time.time()
    #��ÿһ���ĵ�����ѭ��
    for index, row in pd_data.iterrows():
        if(index%100==0):
            print("now doc num:",index)
        content = row['content']
        class_list = row['class']
        for i in range(len(voca)):
            if voca[i] in content:
                for j in range(class_num):
                    if int(class_list[j])==1:
                        # ÿ������ĳ������ֵ�Ƶ��
                        word_appear_set[i][j] += content.count(voca[i])
                        word_doc_set[i][j] += 1
        for j in range(class_num):
            if int(class_list[j]) == 1:
                class_word_appear_set[j] += len(content)
                doc_class_set[j] += 1
    end = time.time()
    print("time is:",(end-begin))
    #word_appear_setΪÿ������ĳ�����г��ֵ�Ƶ��  class_word_appear_set�����ܴ���  word_doc_setÿ������ָôʵ��ĵ���    doc_class_set�������ĵ���
    #˳����Ǵʵ�˳��
    temp = []
    for i in range(len(voca)):
        temp.append([word_appear_set[i],class_word_appear_set,word_doc_set[i],doc_class_set])
    ans = pd.DataFrame(temp,index=voca,columns=['word_appear_set','class_word_appear_set','word_doc_set','doc_class_set'])
    print(ans)
    return ans