# coding=gbk
import pandas as pd
import utils.newsgroup_path as path
"""
    ��20newsgroup textתΪpandas dataframe��py2.7
"""
#��ȡtext�ļ�
def readfile(filename):
    result = []
    lines = open(filename).readlines()
    for line in lines:
        line = line.strip()
        line = line.split("\t")
        result.append(line)
    return result

def getNameList(name):
    name_dict = {'comp.os.ms-windows.misc': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'comp.sys.mac.hardware': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'sci.space': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 'rec.motorcycles': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'sci.med': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'sci.crypt': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'talk.politics.misc': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 'misc.forsale': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'rec.autos': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'comp.graphics': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'talk.politics.guns': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'soc.religion.christian': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'comp.sys.ibm.pc.hardware': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'comp.windows.x': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'alt.atheism': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'rec.sport.baseball': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'sci.electronics': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'rec.sport.hockey': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'talk.religion.misc': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 'talk.politics.mideast': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]}
    return name_dict[name]

#class_numָ��Ҫȡ����������ı���ÿ�����г�ȡ20%��Ϊ���Լ�
def getTrainAndTest(sourcefile):
    total_data = pd.DataFrame(readfile(sourcefile),columns=['class','content'])  #��ȡ�������ݲ�תΪpandas dataframe

    total_data_group = total_data.groupby("class")  # ����class����group

    #ע�⣬����newsgroup�������ԣ�ÿ������ĵ�����������ͬ������ڶ������ʱ�����ֻȡ������������
    count = 0  #ע�����ֻΪ����������ݶ�����
    pd_train,pd_test = pd.DataFrame(),pd.DataFrame()   #ѵ����  ���Լ�
    for name, group in total_data_group:
        if count > 1:  #ѡȡǰ����������ݣ���δ���ֻ��Զ�����
            break
        count += 1

        group['class'] = group['class'].apply(getNameList)
        pd_train = pd_train.append(group.head(int(len(group)*0.8)),ignore_index=True)  #ѵ����ռ80%
        pd_test = pd_test.append(group.tail(len(group) - int(len(group)*0.8)), ignore_index=True)  # ���Լ�ռ20%
    print(pd_train)
    print(pd_test)
    return (pd_train,pd_test)


def getPDData(sourcefile):
    """
    �����ļ����ݣ����ｫѵ�����Ͳ��Լ��ֿ���
    :param sourcefile: 
    :return: ֱ�ӷ�������
    """
    pd_data = pd.DataFrame(readfile(sourcefile),columns=['class','content'])  #��ȡ�������ݲ�תΪpandas dataframe
    pd_data['class'] = pd_data['class'].apply(getNameList)
    return pd_data

if __name__ == '__main__':
    filename = path.SOURCEFILE
    getTrainAndTest(filename)



