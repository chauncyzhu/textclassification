# coding=gbk
import pandas as pd

"""
    ��ȡ�ļ�����ȡ�ʸɡ�ȥ�����ôʵȣ�py2.7
"""
def readfile(filename):#This function is used for getting useful informations in Reuters-21578
    try:
        import xml.etree.cElementTree as ET
    except ImportError:
        import xml.etree.ElementTree as ET
    import sys
    try:
        reuter = ET.parse(filename)  # ��xml�ĵ�
        root = reuter.getroot()  # ���root�ڵ�
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
            #����ĵ�������һ���������û�з��࣬����ӽ�ȥ
            if len(topic_group)>1 or len(topic_group)<1:
                continue
            file.append(topic_group)
            text = child.find('TEXT')
            file.append(text.findtext('TITLE'))
            file.append(text.findtext('BODY'))
            allfiles.append(file)
    return allfiles

#class_numָ��Ҫȡ����������ı�
def getTrainAndTest(sourcefile,class_num):
    allfiles = readfile(sourcefile)
    #�����Ҫ��Զ����������д
    #���ڶ�������⣬����ȥ��ͬ���ڶ��������ĵ���Ȼ����ѡ���ĵ���������8������
    allclass = []
    class_file_num = []
    for item in allfiles:
        for type in item[1]:
            if type not in allclass:
                allclass.append(type)
                class_file_num.append(1)
            else:
                #����Ѿ���ӽ�ȥ�ˣ�������������м�1,ʵ������ͳ��ÿ�����͵��ĵ������Ƕ���
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

    #��listתΪpandas
    pd_train = pd.DataFrame(train)
    pd_train.columns = ["class","content"]
    pd_train['mark'] = 0  #��ʾѵ�������ǲ��Լ�
    pd_test = pd.DataFrame(test)
    pd_test.columns = ["class", "content"]
    pd_test['mark'] = 1  # ��ʾѵ�������ǲ��Լ�
    print(pd_train)
    print(pd_test)
    #����ѵ�����Ͳ��Լ�
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