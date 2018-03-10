# coding=gbk
import pandas as pd
import utils.newsgroup_path as path
"""
    将20newsgroup text转为pandas dataframe，py2.7
"""
#读取text文件
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

#class_num指定要取出多少类的文本，每个类中抽取20%作为测试集
def getTrainAndTest(sourcefile):
    total_data = pd.DataFrame(readfile(sourcefile),columns=['class','content'])  #读取所有数据并转为pandas dataframe

    total_data_group = total_data.groupby("class")  # 根据class进行group

    #注意，由于newsgroup的特殊性，每个类的文档个数几乎相同，因此在二分类的时候最好只取出其中两个类
    count = 0  #注意这个只为二分类的数据而设置
    pd_train,pd_test = pd.DataFrame(),pd.DataFrame()   #训练集  测试集
    for name, group in total_data_group:
        if count > 1:  #选取前两个类别数据，这段代码只针对二分类
            break
        count += 1

        group['class'] = group['class'].apply(getNameList)
        pd_train = pd_train.append(group.head(int(len(group)*0.8)),ignore_index=True)  #训练集占80%
        pd_test = pd_test.append(group.tail(len(group) - int(len(group)*0.8)), ignore_index=True)  # 测试集占20%
    print(pd_train)
    print(pd_test)
    return (pd_train,pd_test)


def getPDData(sourcefile):
    """
    导入文件数据，这里将训练集和测试集分开了
    :param sourcefile: 
    :return: 直接返回数据
    """
    pd_data = pd.DataFrame(readfile(sourcefile),columns=['class','content'])  #读取所有数据并转为pandas dataframe
    pd_data['class'] = pd_data['class'].apply(getNameList)
    return pd_data

if __name__ == '__main__':
    filename = path.SOURCEFILE
    getTrainAndTest(filename)



