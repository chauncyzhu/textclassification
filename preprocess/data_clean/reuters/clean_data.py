# coding=gbk
import cPickle as Pickle
import nltk
import string
import re
"""
    ��ÿ�����ݽ�������
"""
#�Ƚ��зִʣ�����һ��line
def getTokenWords(line):
    return nltk.word_tokenize(line)

#ȥͣ�ô�
def removeStopwords(line):
    #stopwords = nltk.corpus.stopwords.words('english')
    temp_line = []
    exceptwords = ['--','/','//']
    for word in line:
        """
            ����Ե��ʽ��д���ȥ�������ţ�ע����ʱ��ȥ��-���ӵĴ�
            tmpword = re.sub('[\s+\.\!\/_,$%^*(+\"\']+|[+��������������~@#��%����&*����]+',"",word)
            ʹ��������ʽȥ����ȥ��ĸ��-��/֮��ķ��ţ�����-��/����Ϊ�����  japan/india-pakistan-gulf/japan  ֮����������Ƿ���ð�����ֿ��ȽϺ�
            �Ƿ��ȥ���ˣ�����u.s.������
        """
        word = word.lower()
        # ������3rd֮���ȥ��
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
        # �Ƚ��зִ�
        line = getTokenWords(line)
        # ȥ��ͣ�ô�
        line = removeStopwords(line)
        # ��ȡ�ʸ�
        line = getStem(line)
        return line
    pd_data['content'] = pd_data['content'].apply(f)


