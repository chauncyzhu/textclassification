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
        result.append([line[0],line[1]])
    return result

def getNameList(name):
    name_dict = {"comp.windows.x":[1,0,0,0,0,0],"rec.sport.hockey":[0,1,0,0,0,0],"sci.crypt":[0,0,1,0,0,0],
                 "sci.med":[0,0,0,1,0,0],"soc.religion.christian":[0,0,0,0,1,0],"talk.politics.mideast":[0,0,0,0,0,1]}
    return name_dict[name]

#class_num指定要取出多少类的文本，每个类中抽取20%作为测试集
def getTrainAndTest(sourcefile,class_num):
    total_data = pd.DataFrame(readfile(sourcefile),columns=['class','content'])  #读取所有数据并转为pandas dataframe

    total_data_group = total_data.groupby("class")  # 根据class进行group

    pd_train,pd_test = pd.DataFrame(),pd.DataFrame()   #训练集  测试集
    for name, group in total_data_group:
        group['class'] = group['class'].apply(getNameList)
        pd_train = pd_train.append(group.head(int(len(group)*0.8)),ignore_index=True)  #训练集占80%
        pd_test = pd_test.append(group.tail(len(group) - int(len(group)*0.8)), ignore_index=True)  # 测试集占20%
    print(pd_train)
    print(pd_test)
    return (pd_train,pd_test)


if __name__ == '__main__':
    filename = path.SOURCEFILE
    class_num = 8
    getTrainAndTest(filename, class_num)



