# coding=gbk
"""
    评估函数，二分类和多分类不一样
"""
#二分类评估函数，默认class第一个是1为正类，第二个是1为负类
def evaluation_binaryclass(result_data,k_list):
    print("classification result:",result_data)
    evaluation_result = []
    for k in range(len(k_list)):
        tp, fn, fp, tn = 0, 0, 0, 0
        for j in range(len(result_data)):
            # result_data[j][0]为真实标记，result_data[j][1][k]为第K个预测标记，均只有0，1两种，0为正（第0个数为1），1为负（第1个数为1）
            if result_data[j][0] == 0 and result_data[j][1][k] == 0:  # 真实==预测==正
                tp += 1
            if result_data[j][0] == 0 and result_data[j][1][k] == 1:  # 真实为正，预测为负
                fn += 1
            if result_data[j][0] == 1 and result_data[j][1][k] == 0:  # 真实为负，预测为正
                fp += 1
            if result_data[j][0] == 1 and result_data[j][1][k] == 1:  # 真实为负，预测为负
                tn += 1
        if tp + fp == 0:
            precision = float(0)
        else:
            precision = float(tp) / (tp + fp)
        if tp + fn == 0:
            recall = float(0)
        else:
            recall = float(tp) / (tp + fn)
        if precision + recall == 0:
            f1 = float(0)
        else:
            f1 = 2*precision*recall/(precision+recall)
        evaluation_result.append([precision,recall,f1])
    return evaluation_result

#对分类进过进行评估
#result_data格式为，前面为测试集正确分类，后面为K个预测分类
def evaluation_multiclass(result_data,class_num,k_list):
    print("classification result:",result_data)
    evaluation_result = []
    for k in range(len(k_list)):
        precision,recall = [0] * class_num,[0] * class_num
        true_pos,false_neg,false_pos,true_neg = [],[],[],[]
        for i in range(class_num):  #i为正
            tp,fn,fp,tn = 0,0,0,0
            for j in range(len(result_data)):
                #result_data[j][0]为真实标记，result_data[j][1][k]为第K个预测标记
                if result_data[j][0] == i and result_data[j][1][k] == i: # 真实==预测==正
                    tp += 1
                if result_data[j][0] == i and result_data[j][1][k] != i: # 真实为正，预测为负
                    fn += 1
                if result_data[j][0] != i and result_data[j][1][k] == i: # 真实为负，预测为正
                    fp += 1
                if result_data[j][0] != i and result_data[j][1][k] != i: # 真实为负，预测为负
                    tn += 1
            true_pos.append(tp)  #修正错误
            false_neg.append(fn)
            false_pos.append(fp)
            true_neg.append(tn)
            if tp+fp == 0:
                p = 0
            else:
                p = float(tp)/(tp+fp)
            if tp+fn == 0:
                r = 0
            else:
                r = float(tp)/(tp+fn)
            precision.append(p)
            recall.append(r)
        macro_p = sum(precision)/class_num
        macro_r = sum(recall)/class_num
        if macro_p+macro_r == 0:
            macro_f1 = 0
        else:
            macro_f1 = 2*macro_p*macro_r/(macro_p+macro_r)  #宏平均
        if float(sum(true_pos))/len(true_pos)+float(sum(false_pos))/len(false_pos) == 0:
            micro_p = 0
        else:
            micro_p = (float(sum(true_pos))/len(true_pos))/(float(sum(true_pos))/len(true_pos)+float(sum(false_pos))/len(false_pos))
        if float(sum(true_pos))/len(true_pos)+float(sum(false_neg))/len(false_neg) == 0:
            micro_r = 0
        else:
            micro_r = (float(sum(true_pos))/len(true_pos))/(float(sum(true_pos))/len(true_pos)+float(sum(false_neg))/len(false_neg))
        if micro_p+micro_r == 0:
            micro_f1 = 0
        else:
            micro_f1 = 2*micro_p*micro_r/(micro_p+micro_r)  #微平均
        evaluation_result.append([macro_f1,micro_f1])
    #evaluation_result.append(result_data)
    return evaluation_result