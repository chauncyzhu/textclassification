# coding=gbk
"""
    ������Ȩ��Ӧ�õ���������
"""
#��pd_data��content����תΪ��������������"vector"��һ���У�targetfile����ָ��Ҫ�������Ǹ��ļ�����ȥ
def changeToFeatureVector(pd_data,voca_dict,feature_name,target_file=None):
    #���ִ�����ת��Ϊ����
    def f(x,name):
        return list(voca_dict[name][x])  # ̫�����ˣ�DataFrame�����б����룬���Զ������б��ÿһ��Ԫ��
    pd_data['vector'] = pd_data['content'].apply((lambda x:f(x,feature_name)))  # ����ÿһ��words���ԣ���������������������Ȼ����ÿһ���ʣ�pn['words']�൱��ȡ����һ��series
    #��pnд���ļ���
    if target_file:
        pd_data.to_csv(target_file)