# coding=gbk
import pandas as pd

"""
    读取文件、提取词干、去除提用词等，py2.7
"""
def readfile(filename):#This function is used for getting useful informations in Reuters-21578
    try:
        import xml.etree.cElementTree as ET
    except ImportError:
        import xml.etree.ElementTree as ET
    import sys
    try:
        reuter = ET.parse(filename)  # 打开xml文档
        root = reuter.getroot()  # 获得root节点
    except Exception, e:
        print("error:",e)
        print "Error:cannot parse file:reuters.xml."
        sys.exit(1)
    allfiles = []
    for child in root:
        file = []  #file [train or test,topics,title,body]
        topics = child.get('TOPICS')
        if(topics == 'YES'):
            a = child.get("LEWISSPLIT")
            if(a=='TRAIN'):
                a = 0
                file.append(a)
            elif(a=='TEST'):
                a = 1
#                print 'testset'
                file.append(a)
            else:
                continue
            topic_item = child.find('TOPICS')
            topics = topic_item.itertext()
            topic_group = []
            for eachtopic in topics:
                topic_group.append(eachtopic)
            #如果文档超过了一个分类或者没有分类，则不添加进去
            if len(topic_group)>1 or len(topic_group)<1:
                continue
            file.append(topic_group)
            text = child.find('TEXT')
            file.append(text.findtext('TITLE'))
            file.append(text.findtext('BODY'))
            allfiles.append(file)
    return allfiles

#class_num指定要取出多少类的文本
def getTrainAndTest(sourcefile,class_num):
    allfiles = readfile(sourcefile)
    #后面的要针对多分类类重新写
    #对于多分类问题，首先去掉同属于多个分类的文档，然后再选出文档数量最多的8个分类
    allclass = []
    class_file_num = []
    for item in allfiles:
        for type in item[1]:
            if type not in allclass:
                allclass.append(type)
                class_file_num.append(1)
            else:
                #如果已经添加进去了，则在这个类型中加1,实际上是统计每种类型的文档数量是多少
                class_file_num[allclass.index(type)] += 1

    temp_sort = []
    for i in range(len(allclass)):
        temp_sort.append([allclass[i],class_file_num[i]])
    newset = sorted(temp_sort, key=lambda file: file[1],reverse=True)
    print("each class number:")
    print(map(lambda x:x[0]+"--"+str(x[1]),newset))
    ans = []
    for i in range(class_num):
        ans.append(newset[i][0])
    all_file_set = []
    for item in allfiles:
        for type in item[1]:
            if type in ans:
                type_vector = []
                for kind in ans:
                    if kind in type:
                        type_vector.append(1)
                    else:
                        type_vector.append(0)
                if((item[2]!=None)&(item[3]!=None)):
                    all_file_set.append([item[0],type_vector,item[2]+' '+item[3]])
                elif(item[3]!=None):
                        all_file_set.append([item[0], type_vector, item[3]])
                break
    train = []
    test = []
    print(all_file_set[0])
    for item in all_file_set:
        if item[0] == 0:
            train.append([item[1],item[2]])
        elif item[0] == 1:
            test.append([item[1], item[2]])

    #将list转为pandas
    pd_train = pd.DataFrame(train)
    pd_train.columns = ["class","content"]
    pd_train['mark'] = 0  #表示训练集还是测试集
    pd_test = pd.DataFrame(test)
    pd_test.columns = ["class", "content"]
    pd_test['mark'] = 1  # 表示训练集还是测试集
    print(pd_train)
    print(pd_test)
    #返回训练集和测试集
    return (pd_train,pd_test)

    # train_num = [0]*class_num
    # test_num = [0]*class_num
    # for item in all_file_set:
    #     if item[0] == 0:
    #         for i in range(len(item[1])):
    #             if item[1][i] == 1:
    #                 train_num[i] += 1
    #     elif item[0] == 1:
    #         for i in range(len(item[1])):
    #             if item[1][i] == 1:
    #                 test_num[i] += 1
    # print("train_num:",train_num)
    # print("test_num:",test_num)
    # print("train data",train)
    # print("test data",test)

if __name__ == '__main__':
    sourcefile = "../../data/reuters/reuter_all.xml"
    class_num = 8
    getTrainAndTest(sourcefile, class_num)