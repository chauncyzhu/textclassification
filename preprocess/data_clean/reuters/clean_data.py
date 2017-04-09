# coding=gbk
import cPickle as Pickle
import nltk
import string
import re
"""
    将每条数据进行清理
"""
#先进行分词，传入一个line
def getTokenWords(line):
    return nltk.word_tokenize(line)

#去停用词
def removeStopwords(line):
    #stopwords = nltk.corpus.stopwords.words('english')
    temp_line = []
    exceptwords = ['--','/','//']
    for word in line:
        """
            这里对单词进行处理，去掉标点符号，注意暂时不去掉-连接的词
            tmpword = re.sub('[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+',"",word)
            使用正则表达式去掉除去字母、-、/之外的符号，保留-、/是因为会出现  japan/india-pakistan-gulf/japan  之类情况，但是否最好把这个分开比较好
            是否多去除了，对于u.s.这样的
        """
        word = word.lower()
        # 对于像3rd之类的去掉
        if ((word.isdigit()) | (word[0].isdigit()) | (word[-1].isdigit())):
            continue
        tmpword = re.sub('[^a-z\-\/]', "", word)
        if len(tmpword) == 0 or tmpword in exceptwords:
            continue
        else:
            temp_line.append(tmpword)
    return temp_line

def getStem(line):
    temp_line = []
    s = nltk.stem.SnowballStemmer('english')
    try:
        for word in line:
            temp_line.append(s.stem(word))
    except:
        print 'wrong'
    return temp_line


def clean_data(pd_data):
    def f(line):
        # 先进行分词
        line = getTokenWords(line)
        # 去除停用词
        line = removeStopwords(line)
        # 提取词干
        line = getStem(line)
        return line
    pd_data['content'] = pd_data['content'].apply(f)


