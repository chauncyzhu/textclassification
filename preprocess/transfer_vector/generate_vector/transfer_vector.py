# coding=gbk
import numpy as np
"""
    ������Ȩ��Ӧ�õ���������
"""
#�����������С���ĵ���sizeһ��
def changeToDocVector(pd_data,voca_dict,feature_name,target_file=None):
    #���ִ�����ת��Ϊ����
    def f(x,name):
        return list(voca_dict[name][x])  # ̫�����ˣ�DataFrame�����б����룬���Զ������б��ÿһ��Ԫ��
    pd_data['vector'] = pd_data['content'].apply((lambda x:f(x,feature_name)))  # ����ÿһ��words���ԣ���������������������Ȼ����ÿһ���ʣ�pn['words']�൱��ȡ����һ��series
    #��pnд���ļ���
    if target_file:
        pd_data.to_csv(target_file)

#��pd_data��content����תΪ��������������feature_name��һ���У�targetfile����ָ��Ҫ�������Ǹ��ļ�����ȥ
#�����vector�Ǻ����������Ĵ�Сһ��
def changeToFeatureVector(pd_data,voca_dict,feature_name,target_file=None):
    # ѡ��voca_dict��ǰ1/32
    #voca_dict = voca_dict.sort_values(by=feature_name,ascending=False).head(int(len(voca_dict)/32))  #���򣬲�ȡ��ǰ1/32
    print("voca_dict:",voca_dict)
    # ���ִ�����ת��Ϊ����
    def f(x, name):
        vector = []  # vector����ʵ��
        for word in voca_dict.index:
            if word in x:  #����������ĵ���
                vector.append(float(voca_dict[name][word]))
            else:
                vector.append(0)  #�����NaN
        return vector
    pd_data[feature_name] = pd_data['content'].apply((lambda x: f(x, feature_name)))  # ����ÿһ��words���ԣ���������������������Ȼ����ÿһ���ʣ�pn['words']�൱��ȡ����һ��series
    #��NaNת��0
    #pd_data['vector'] = pd_data['vector'].apply(lambda x:list(np.nan_to_num(x)))
    # ��pnд���ļ���
    if target_file:
        pd_data.to_csv(target_file)

