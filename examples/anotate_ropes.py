# -*- coding：utf-8 -*-
import json
import tqdm
import regex as re

from transformers.data.metrics.squad_metrics import (
    compute_exact,
    compute_f1
)
import xlwt
import xlrd

def find_substring_index(pattern_string, string):
    if not pattern_string:
        return None
    regex = re.escape(pattern_string)
    if pattern_string[0].isalpha() or pattern_string[0].isdigit():
        regex = "\\b" + regex
    if pattern_string[-1].isalpha() or pattern_string[-1].isdigit():
        regex = regex + "\\b"
    res = [match.start() for match in re.finditer(regex, string)]
    if len(res) == 0:
        regex_uncased = "(?i)" + regex
        res = [match.start() for match in re.finditer(regex_uncased, string.lower())]
    return [[i,i+len(pattern_string)] for i in res]

def find_all_substring_index(pattern_string, string):
    if not pattern_string:
        return None
    regex = re.escape(pattern_string)
    if pattern_string[0].isalpha() or pattern_string[0].isdigit():
        regex = "\\b" + regex
    if pattern_string[-1].isalpha() or pattern_string[-1].isdigit():
        regex = regex + "\\b"
    res = [match.start() for match in re.finditer(regex, string)]
    if len(res) == 0:
        regex_uncased = "(?i)" + regex
        res = [match.start() for match in re.finditer(regex_uncased, string.lower())]
    return res if res else []

def find_all_indexes(input_str, search_str):
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = input_str.find(search_str, index)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1



def read_data(path):
    with open(path, "r", encoding="utf-8")as reader:
        input_data = json.load(reader)
    data = []
    for entry in input_data["data"]:
        title = entry['title']
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["background"]
            situation_text = paragraph['situation']
            label={}
            for qa in paragraph['qas']:
                if qa['label'] != {}:
                    continue
                qas_id = qa['id']
                question_text = qa['question']
                answer = qa["answers"][0]
                answer_text = answer["text"]
                shows = {
                    "qas_id":qas_id,
                    "Background": paragraph_text,
                    "Situation": situation_text,
                    "Question": question_text,
                    "Answer": answer_text
                }
                data.append(shows)
    return data


def set_style(name,height,bold=False):
	style = xlwt.XFStyle()
	font = xlwt.Font()
	font.name = name
	font.bold = bold
	font.color_index = 4
	font.height = height
	style.font = font
	return style


def make_excel_and_write_examples(data):
    f = xlwt.Workbook(encoding = 'utf-8')
    sheet1 = f.add_sheet('train', cell_overwrite_ok=True)
    row0 = ["qas_id", "Background", "Situation", "Question","Answer",
            "Object1","O1_support","Object2","O2_support",
            "s_SP_o1","s_SP_o2", "b_SP","b_TP",
            "Relevance",'Polarity',"TP_Relevance",
            "SP_O1_support","SP_O2_support","SP_back_support","TP_back_support"]
    for i in range(0, len(row0)):
        sheet1.write(0, i, row0[i], set_style('Times New Roman', 220, True))

    for i in range(0,len(data)):
        for j in range(0, 5):
            sheet1.write(i + 1, j, data[i][row0[j]])

    f.save("path\\to\\save\\train.xls")


def remove_blank(src_string):
    if src_string[0] ==' ':
        src_string = src_string[1:]

    if src_string[-1] == ' ':
        src_string = src_string[:-1]

    return src_string

def find_ind_with_support(src_string,support_string,passage):
    if not src_string:
        return None
    else:
        src_string = remove_blank(src_string)
        if not support_string or src_string not in support_string:
            try:
                return find_substring_index(src_string,passage)[0]
            except:
                return [passage.find(src_string),passage.find(src_string)+len(src_string)]
        else:
            try:
                support_idx = find_substring_index(support_string,passage)[0]
            except:
                support_idx =[passage.find(support_string),passage.find(support_string)+len(support_string)]
            try:
                src_idx =find_substring_index(src_string,passage)
            except:
                src_idx = [[passage.find(src_string),passage.find(src_string)+len(src_string)]]
            for i in src_idx:
                if i[0]>=support_idx[0] and i[1]<=support_idx[1]:
                    return i
                    break


def read_excel_and_make_label(path,rown=0,maxn = None):
    wb = xlrd.open_workbook(filename=path)
    sheet = wb.sheet_by_name("train")
    instances = {}
    if maxn is not None:
        maxn = int(maxn)
    else:
        maxn = sheet.nrows
    for i in range(rown,maxn):
        qas_id = str(int(float(sheet.cell(i,0).value)))
        Background = sheet.cell(i, 1).value
        Situation = sheet.cell(i, 2).value
        Question = sheet.cell(i, 3).value

        try:
            Object1= sheet.cell(i,5).value
            if not Object1:
                continue
            O1_support = sheet.cell(i,6).value
            print(O1_support)
            print(Object1 in O1_support)
            O1_ind = find_ind_with_support(Object1,O1_support,Situation)
            print(O1_ind)
            Object2 = sheet.cell(i, 7).value
            O2_support = sheet.cell(i, 8).value
            O2_ind = find_ind_with_support(Object2, O2_support, Situation)
        except:
            print(qas_id)
            print("object!")

        try:
            s_SP_o1 = sheet.cell(i, 9).value
            SP_O1_support = sheet.cell(i, 16).value
            SP_O1_ind = find_ind_with_support(s_SP_o1, SP_O1_support, Situation)
            s_SP_o2 = sheet.cell(i, 10).value
            SP_O2_support = sheet.cell(i, 17).value
            SP_O2_ind = find_ind_with_support(s_SP_o2, SP_O2_support, Situation)
        except:
            print(qas_id)
            print("SP!")

        try:
            b_SP = sheet.cell(i, 11).value
            SP_back_support = sheet.cell(i, 18).value
            SP_Back_ind = find_ind_with_support(b_SP, SP_back_support, Background)
            b_TP = sheet.cell(i, 12).value
            TP_back_support = sheet.cell(i, 19).value
            TP_Back_ind = find_ind_with_support(b_TP, TP_back_support, Background)
        except:
            print(qas_id)
            print("back!!")

        try:
            Relevance = sheet.cell(i, 13).value
            Rel_label = 1 if int(Relevance) ==1 else 0
            Polarity = sheet.cell(i, 14).value
            Pol_label = 1 if int(Polarity) == 1 else 0
            TP_Relevance = sheet.cell(i, 15).value
            TP_Rel_label = 1 if int(TP_Relevance) == 1 else 0
        except:
            print(qas_id)
            print("number!!!")

        if (int(Relevance) == 1 and int(Polarity) == 1) or (int(Relevance) == 0 and int(Polarity) == 0):
            try:
                assert int(TP_Relevance) ==1
            except:
                print(qas_id)
        else:
            try:
                assert int(TP_Relevance) ==0
            except:
                print(qas_id)

        out_label = {
            "Object1":Object1,
            "Object1_ind":O1_ind,
            "Object2": Object2,
            "Object2_ind": O2_ind,
            "SP_O1":s_SP_o1,
            "SP_O1_ind":SP_O1_ind,
            "SP_O2": s_SP_o2,
            "SP_O2_ind": SP_O2_ind,
            "Back_SP":b_SP,
            "Back_SP_ind": SP_Back_ind,
            "Back_TP":b_TP,
            "Back_TP_ind":TP_Back_ind,
            "Relevance":Rel_label,
            "Polarity":Pol_label,
            "TP_Relevacne":TP_Rel_label
        }
        instances[qas_id] = out_label
        # print(out_label)
    return instances



def remake_train_dev(path,dict_train,dict_dev):
    with open(path, "r", encoding="utf-8")as reader:
        input_data = json.load(reader)

    train = {
        "version": "1.0",
        "data": [
            {
                "title": "ropes",
                "paragraphs": []
            }]
    }

    dev = {
        "version": "1.0",
        "data": [
            {
                "title": "ropes",
                "paragraphs": []
            }]
    }

    for entry in input_data["data"]:
        for paragraph in entry["paragraphs"]:
            is_train = False
            is_Dev = False
            for qa in paragraph['qas']:
                qas_id = qa['id']
                if qas_id in dict_train.keys():
                    label = dict_train[qas_id]
                    qa['label'] = label
                    is_train = True
                if qas_id in dict_dev.keys():
                    label = dict_dev[qas_id]
                    qa['label'] = label
                    is_Dev = True
            if is_train:
                train['data'][0]['paragraphs'].append(paragraph)
            if is_Dev:
                dev['data'][0]['paragraphs'].append(paragraph)
    writer_trian = open("R:\\lablelropes\\moredata\\labelled_train_more.json",'w+',encoding='utf-8')
    writer_trian.write(json.dumps(train,indent=4))
    writer_dev = open("R:\\lablelropes\\moredata\\labelled_dev_more.json",
                        'w+', encoding='utf-8')
    writer_dev.write(json.dumps(dev, indent=4))
    writer_trian.close()
    writer_dev.close()
    return train, dev

def make_new_with_synthetic_text(synthetic_path,train_path,outpath):
    with open(synthetic_path,'r',encoding='utf-8') as reader:
        synthetic_data = json.load(reader)
    synthetic_text = {}
    for key,value in synthetic_data.items():
        synthetic_text[key] = value['constructed text']

    writer = open(outpath,'w+',encoding='utf-8')

    with open(train_path, "r", encoding="utf-8")as reader:
        input_data = json.load(reader)
    for entry in input_data["data"]:
        title = entry['title']
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["background"]
            situation_text = paragraph['situation']
            label={}
            for qa in paragraph['qas']:
                qas_id = qa['id']
                synthetic_t = synthetic_text[qas_id]
                qa['synthetic_text'] = synthetic_t

    writer.write(json.dumps(input_data,indent=4))
    writer.close()

def make_gold_synthetic_text(labelled_path,file_path,output_path):
    with open(labelled_path,'r',encoding='utf-8') as reader:
        lablled_data = json.load(reader)
    all_synthetic_text = {}
    for entry in lablled_data["data"]:
        for paragraph in entry["paragraphs"]:
            for qa in paragraph['qas']:
                qas_id = qa['id']
                try:
                    label = qa['label']
                except:
                    print(qas_id)
                    continue
                pred_tp_rel = label["TP_Relevacne"]
                o1_final_text = remove_blank(label['Object1'])
                o2_final_text = remove_blank(label["Object2"])
                TP_final_text = remove_blank(label['Back_TP'])

                if pred_tp_rel == 0:
                    predict_text  = o1_final_text+ ' has larger '+ TP_final_text + ' than '+o2_final_text
                if pred_tp_rel == 1:
                    predict_text  = o1_final_text+ ' has smaller '+ TP_final_text + ' than '+o2_final_text
                all_synthetic_text[qas_id] = predict_text

    with open(file_path,'r',encoding='utf-8') as reader:
        input_data = json.load(reader)
    for entry in input_data["data"]:
        for paragraph in entry["paragraphs"]:
            for qa in paragraph['qas']:
                qas_id = qa['id']
                try:
                    synthetic_t = all_synthetic_text[qas_id]
                    qa['synthetic_text'] = synthetic_t
                except:
                    synthetic_t =qa['synthetic_text']

    writer = open(output_path,'w+',encoding='utf-8')
    writer.write(json.dumps(input_data,indent=4))
    writer.close()

def statistics_synthetic(t_path):
    with open(t_path, "r", encoding="utf-8") as reader:
        train_data = json.load(reader)["data"]
    t_ainq=[]
    t_ains=[]
    t_ainqs = []
    t_ainsyn =[]
    n_inq = 0
    n_ins =0
    n_inqs =0
    n_insyn =0
    t_all=[]
    for entry in train_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["background"]
            situation_text = paragraph['situation']
            for qa in paragraph['qas']:
                qas_id = qa['id']
                question_text = qa['question']
                synthetic_text = qa['synthetic_text']
                context_text = synthetic_text

                answer = qa["answers"][0]
                answer_text = answer["text"]
                answer_offsets_q = find_all_substring_index(answer_text, question_text)
                answer_offsets_s = find_all_substring_index(answer_text, situation_text)
                answer_offsets_syn = find_all_substring_index(answer_text, context_text)
                answer_offsets_qs = find_all_substring_index(answer_text, question_text+' '+situation_text)
                if answer_offsets_q:
                    n_inq+=1
                    t_ainq.append(len(answer_offsets_q))
                if answer_offsets_s:
                    n_ins += 1
                    t_ains.append(len(answer_offsets_s))
                if answer_offsets_syn:
                    n_insyn += 1
                    t_ainsyn.append(len(answer_offsets_syn))
                if answer_offsets_qs:
                    n_inqs += 1
                    t_ainqs.append(len(answer_offsets_qs))
                t_all.append(len(answer_offsets_syn)+len(answer_offsets_qs))

    return t_ainq,t_ains,t_ainqs,t_ainsyn,n_inq,n_ins,n_inqs,n_insyn,t_all
def comput_scores(predict_path,multi_answer_path,output):
    with open(multi_answer_path,'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    candidate_answer= {}
    for line in lines:
        line = json.loads(line)
        candidate_answer[line['id']] = line['answer_text']

    writer = open(output,'w+',encoding='utf-8')
    ignored_list =[
        'more','less','higher','lower','increase','decrease','high','low','harder','easier','increasing','decreasing','up','down','larger','smaller'
    ]
    final_score = {}
    cnt= 0
    both_correct = 0
    half_correct = 0
    half_good =0
    same = 0
    both_good = 0
    both_wrong =0
    both_bad= 0
    with open(predict_path, "r", encoding="utf-8") as reader:
        input_data = json.load(reader)
    for entry in input_data["data"]:
        for paragraph in entry["paragraphs"]:
            background = paragraph['background']
            situation = paragraph['situation']
            for qa in paragraph['qas']:
                qas_id = str(qa['id'])
                predict = qa['predicts']
                candidate = candidate_answer[qas_id]

                first_can = candidate[0]
                second_can =candidate[1]
                o1_can1 =compute_f1(first_can,predict['object1'])
                o1_can2=compute_f1(second_can,predict['object1'])
                o2_can1 =compute_f1(first_can,predict['object2'])
                o2_can2=compute_f1(second_can,predict['object2'])

                if qa["answers"][0]['text'] in ignored_list:
                    cnt+=1
                    continue

                # both are correct
                if (o1_can1 ==1 and o2_can2 == 1) or (o1_can2 ==1 and o2_can1 ==1):
                    both_correct+=1

                if (o1_can1 >= 0.5 and o2_can2 >=0.5) or (o1_can2 >=0.5 and o2_can1 >=0.5):
                    both_good += 1

                o1_f1 = max(max(compute_f1(answer,predict['object1']) for answer in candidate),compute_f1(qa["answers"][0]['text'],predict['object1']))
                o2_f1 = max(max(compute_f1(answer,predict['object2']) for answer in candidate),compute_f1(qa["answers"][0]['text'],predict['object2']))

                if (o1_f1 ==1 or o2_f1 ==1) and predict['object1']!= predict['object2'] :
                    continue
                if o1_f1 >= 0.5 and o2_f1 >= 0.5:
                    continue
                final_score[qas_id] = [o1_f1,o2_f1]
                qa['predicts']['f1'] = [o1_f1,o2_f1]
                out = {
                    "background":background,
                    "situation":situation,
                    "qa":qa,
                }
                writer.write(json.dumps(out,indent=2))
                writer.write("\n")
    writer.close()
    print(cnt)
    print(same)
    print(both_correct)
    print(half_correct)
    print(both_good)
    print(half_good)

    return final_score
def statistsic_step18(predict_path, multi_answer_path, output):
    with open(multi_answer_path, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    candidate_answer = {}
    for line in lines:
        line = json.loads(line)
        candidate_answer[line['id']] = line['answer_text']

    writer = open(output, 'w+', encoding='utf-8')
    ignored_list = [
        'more', 'less', 'higher', 'lower', 'increase', 'decrease', 'high', 'low', 'harder', 'easier', 'increasing',
        'decreasing', 'up', 'down', 'larger', 'smaller'
    ]
    cnt=0
    final_score = {}
    compare = 0
    both_correct = 0
    half_correct = 0
    half_good = 0
    same = 0
    both_good = 0
    both_wrong = 0
    both_bad = 0
    with open(predict_path, "r", encoding="utf-8") as reader:
        input_data = json.load(reader)
    for entry in input_data["data"]:
        for paragraph in entry["paragraphs"]:
            background = paragraph['background']
            situation = paragraph['situation']
            for qa in paragraph['qas']:
                qas_id = str(qa['id'])
                predict = qa['predicts']
                candidate = candidate_answer[qas_id]

                first_can = candidate[0]
                second_can = candidate[1]
                o1_can1 = compute_f1(first_can, predict['object1'])
                o1_can2 = compute_f1(second_can, predict['object1'])
                o2_can1 = compute_f1(first_can, predict['object2'])
                o2_can2 = compute_f1(second_can, predict['object2'])

                cnt+=1
                if qa["answers"][0]['text'] in ignored_list:
                    compare += 1
                    continue

                # both are correct
                if (o1_can1 == 1 and o2_can2 == 1) or (o1_can2 == 1 and o2_can1 == 1):
                    both_correct += 1
                elif (o1_can1 == 1 and o2_can2 == 0) or (o1_can2 == 1 and o2_can1 == 0) or (
                        o1_can1 == 0 and o2_can2 == 1) or (o1_can2 == 0 and o2_can1 == 1):
                    half_correct += 1

                # if (o1_can1 <= 0.7 and o2_can2 <= 0.7 and o1_can1 >= 0.5 and o2_can2 >= 0.5) or (o1_can2 >= 0.5 and o2_can1 >= 0.5 and o1_can2 <= 0.7 and o2_can1 <= 0.7):
                #     # both_good += 1


                if (o1_can1 >= 0.5 and o2_can2 >= 0.5) or (o1_can2 >= 0.5 and o2_can1 >= 0.5):
                    both_good += 1
                elif (o1_can1 >= 0.5 and o2_can2 < 0.5) or (o1_can2 >= 0.5 and o2_can1 < 0.5) or (o2_can2 >= 0.5 and o1_can1 < 0.5) or (o2_can1 >= 0.5 and o1_can2 < 0.5):
                    half_good += 1


                if (o1_can1 == 0 and o2_can2 == 0) and (o1_can2 == 0 and o2_can1 == 0):
                    both_wrong += 1

                if (o1_can1 < 0.5 and o2_can2 < 0.5) and (o1_can2 < 0.5 and o2_can1 < 0.5):
                    both_bad += 1
                    qa['predicts']['f1'] = [o1_can1, o2_can1, o1_can2, o2_can2]
                    out = {
                        "background": background,
                        "situation": situation,
                        "qa": qa,
                    }
                    writer.write(json.dumps(out, indent=2))
                    writer.write("\n")

    writer.close()

    print(cnt,both_correct,both_good,half_correct,half_good,both_wrong,both_bad)
    print(compare)

import os
def make_ref_can(data,pred,out,id):
    # writer_ref = open(os.path.join(out,"refs_a_1.txt"),'w+',encoding='utf-8')
    # writer_ref = open(os.path.join(out, "refs_a_"+str(id)+".txt"), 'w+', encoding='utf-8')
    # writer_hyps = open(os.path.join(out,"hyps_a_1_base.txt"),'w+',encoding='utf-8')
    filtered = [1912355095, 580926078, 2893035919, 3005625773, 4037359159, 1090395805, 1111432861, 1772822810,
                1779114266, 3003444033, 1687481522, 373839753, 3209385804, 3128907286, 1746228971, 4088330372,
                4227418477, 4047850835,
                302402461, 603409309, 2849858743, 2050057087, 336287454, 1787516699, 184586157, 2999947384, 1202686909,
                1223723965, 2335934912, 2677049981, 1182285725, 430326667, 4183507762, 1091390005, 1076054581,
                3847534556,
                4074158044, 955038082, 1423883267,
                1626887498, 184177891, 1586058524, 156980418, 1607423284, 1608996147, 3071497657,
                3865282641, 1588430868, 3876948090, 3344467660, 3346695902, 3600844723, 3227879116, 3484256179,
                2934519278, 3876075109, 2041918832, 1115960569, 1108882681, 2013214064, 3895408229, 2975020526,
                3885579536, 356328641, 985605428, 809437101, 494407430, 3999682891, 1562329124, 1505378340, 2022868633,
                2022147737, 1729594789, 1730315685, 1082622173, 1205043665, 1081901277, 1204322769]
    filtered_id = [str(i) for i in filtered]

    writer_hyps = open(os.path.join(out,"hyps_a_"+str(id)+"_our.txt"),'w+',encoding='utf-8')
    with open(data, "r", encoding="utf-8")as reader:
        input_data = json.load(reader)
    for entry in input_data["data"]:
        for paragraph in entry["paragraphs"]:
            # paragraph_text = paragraph["background"]
            situation_text = paragraph['situation'].strip()
            for qa in paragraph['qas']:
                qas_id = qa['id']
                if qas_id in filtered_id:
                    continue
                question_text = qa['question']
                answer_text = qa['answers'][0]['text']
                # ref1 = ' '.join((question_text,situation_text,answer_text))
                # can1 = ' '.join((question_text,situation_text,pred[qas_id].strip()))
                ref2 = answer_text
                can2 = pred[qas_id]
                # writer_ref.write(ref2)
                # writer_ref.write('\n')
                writer_hyps.write(can2)
                writer_hyps.write('\n')
    # writer_ref.close()
    writer_hyps.close()

def check_f1(data,pred):
    f1= []
    filtered = [1912355095, 580926078, 2893035919, 3005625773, 4037359159, 1090395805, 1111432861, 1772822810,
                1779114266, 3003444033, 1687481522, 373839753, 3209385804, 3128907286, 1746228971, 4088330372,
                4227418477, 4047850835,
                302402461, 603409309, 2849858743, 2050057087, 336287454, 1787516699, 184586157, 2999947384, 1202686909,
                1223723965, 2335934912, 2677049981, 1182285725, 430326667, 4183507762, 1091390005, 1076054581,
                3847534556,
                4074158044, 955038082, 1423883267,
                1626887498, 184177891, 1586058524, 156980418, 1607423284, 1608996147, 3071497657,
                3865282641, 1588430868, 3876948090, 3344467660, 3346695902, 3600844723, 3227879116, 3484256179,
                2934519278, 3876075109, 2041918832, 1115960569, 1108882681, 2013214064, 3895408229, 2975020526,
                3885579536, 356328641, 985605428, 809437101, 494407430, 3999682891, 1562329124, 1505378340, 2022868633,
                2022147737, 1729594789, 1730315685, 1082622173, 1205043665, 1081901277, 1204322769]
    filtered_id = [str(i) for i in filtered]
    em =[]
    with open(data, "r", encoding="utf-8")as reader:
        input_data = json.load(reader)
    for entry in input_data["data"]:
        for paragraph in entry["paragraphs"]:
            # paragraph_text = paragraph["background"]
            situation_text = paragraph['situation']
            for qa in paragraph['qas']:
                qas_id = qa['id']
                question_text = qa['question']
                answer_text = qa['answers'][0]['text']
                if qas_id in filtered_id:
                    continue
                else:
                    f1.append(compute_f1(answer_text,pred[qas_id]))
                    em.append(compute_exact(answer_text, pred[qas_id]))
    return f1,em


def fuzzy_f1(data):
    def comput_f(a):
        return sum(a)/len(a)
    group1 =[]
    group2 =[]
    effect_B = []
    cause_B = []
    cause_G1 = []
    casue_G2 =[]
    f_group1 = []
    f_group2 = []
    f_effect_B = []
    f_cause_B = []
    f_cause_G1 = []
    f_casue_G2 = []
    with open(data, "r", encoding="utf-8")as reader:
        input_data = json.load(reader)
    for entry in input_data["data"]:
        for paragraph in entry["paragraphs"]:
            for qa in paragraph['qas']:
                predict = qa['predicts']
                g1 = compute_f1(predict['object1']['label'],predict['object1']['predict'])
                g2 = compute_f1(predict['object2']['label'],predict['object2']['predict'])
                effect = compute_f1(predict['TP in back']['label'],predict['TP in back']['predict'])
                cb = compute_f1(predict['SP in back']['label'],predict['SP in back']['predict'])
                cg1 =compute_f1(predict['SP_object1']['label'],predict['SP_object1']['predict'])
                cg2 = compute_f1(predict['SP_object2']['label'],predict['SP_object2']['predict'])

                g12 = compute_f1(predict['object1']['label'], predict['object2']['predict'])
                g21 = compute_f1(predict['object2']['label'], predict['object1']['predict'])
                cg12 = compute_f1(predict['SP_object1']['label'], predict['SP_object2']['predict'])
                cg21 = compute_f1(predict['SP_object2']['label'], predict['SP_object1']['predict'])
                group1.append(g1) if g1 > g12 else group1.append(g12)
                group2.append(g2) if g2 > g21 else group2.append(g21)
                effect_B.append(effect)
                cause_B.append(cb)
                cause_G1.append(cg1)
                casue_G2.append(cg2)
                f_group1.append(1) if g1 > 0 or g12 >0 else f_group1.append(0)
                f_group2.append(1) if g2 > 0 or g21 >0 else f_group2.append(0)
                f_effect_B.append(1) if effect > 0 else f_effect_B.append(0)
                f_cause_B.append(1) if cb > 0 else f_cause_B.append(0)
                f_cause_G1.append(1) if cg1 > 0 or cg12>0 else f_cause_G1.append(0)
                f_casue_G2.append(1) if cg2 > 0 or cg21>0 else f_casue_G2.append(0)


    return comput_f(group1),comput_f(group2),comput_f(effect_B),comput_f(cause_B),comput_f(cause_G1),comput_f(casue_G2),comput_f(f_group1),comput_f(f_group2),comput_f(f_effect_B),comput_f(f_cause_B),comput_f(f_cause_G1),comput_f(f_casue_G2)

def comput_modified_f1(data,pred):
    f1= []
    em=[]
    with open(data, "r", encoding="utf-8")as reader:
        input_data = json.load(reader)
    for entry in input_data["data"]:
        for paragraph in entry["paragraphs"]:
            # paragraph_text = paragraph["background"]
            situation_text = paragraph['situation']
            for qa in paragraph['qas']:
                qas_id = qa['id']
                question_text = qa['question']
                answer_text = qa['answers'][0]['text']
                pred_ans = pred[qas_id]
                answer_text = answer_text[5:] if answer_text.startswith("Team") else answer_text
                answer_text = answer_text[4:] if answer_text.startswith("Day") else answer_text
                answer_text = answer_text[:-2].strip() if answer_text.endswith("PM") else answer_text
                answer_text = answer_text[:-2].strip() if answer_text.endswith("AM") else answer_text

                pred_ans = pred_ans[5:] if pred_ans.startswith("Team") else pred_ans
                pred_ans = pred_ans[4:] if pred_ans.startswith("Day") else pred_ans
                pred_ans = pred_ans[:-2].strip() if pred_ans.endswith("PM") else pred_ans
                pred_ans = pred_ans[:-2].strip() if pred_ans.endswith("AM") else pred_ans
                f1.append(compute_f1(answer_text, pred_ans))
                em.append(compute_exact(answer_text, pred_ans))
    return sum(f1)/len(f1), sum(em)/len(em)

def statistics_for_GCE(data):
    import nltk
    group1 =[]
    group2 =[]
    effect_B = []
    cause_B = []
    cause_G1 = []
    casue_G2 =[]
    with open(data, "r", encoding="utf-8")as reader:
        input_data = json.load(reader)
    for entry in input_data["data"]:
        for paragraph in entry["paragraphs"]:
            for qa in paragraph['qas']:
                predict = qa['predicts']
                g1 = predict['object1']['predict']
                g2 = predict['object2']['predict']
                effect = predict['TP in back']['predict']
                cb = predict['SP in back']['predict']
                cg1 =predict['SP_object1']['predict']
                cg2 = predict['SP_object2']['predict']
                group1.append(len(nltk.word_tokenize(g1)))
                group2.append(len(nltk.word_tokenize(g2)))
                effect_B.append(len(nltk.word_tokenize(effect)))
                cause_B.append(len(nltk.word_tokenize(cb)))
                cause_G1.append(len(nltk.word_tokenize(cg1)))
                casue_G2.append(len(nltk.word_tokenize(cg2)))
    return group1,group2,effect_B,cause_B,cause_G1,casue_G2

def statistics_for_GCE_label(data):
    import nltk
    group1 =[]
    group2 =[]
    effect_B = []
    cause_B = []
    cause_G1 = []
    casue_G2 =[]
    with open(data, "r", encoding="utf-8")as reader:
        input_data = json.load(reader)
    for entry in input_data["data"]:
        for paragraph in entry["paragraphs"]:
            for qa in paragraph['qas']:
                predict = qa['label']
                g1 = predict['Object1']
                g2 = predict['Object2']
                effect = predict['Back_TP']
                cb = predict['Back_SP']
                cg1 =predict['SP_O1']
                cg2 = predict['SP_O2']
                group1.append(len(nltk.word_tokenize(g1)))
                group2.append(len(nltk.word_tokenize(g2)))
                effect_B.append(len(nltk.word_tokenize(effect)))
                cause_B.append(len(nltk.word_tokenize(cb)))
                cause_G1.append(len(nltk.word_tokenize(cg1)))
                casue_G2.append(len(nltk.word_tokenize(cg2)))
    return group1,group2,effect_B,cause_B,cause_G1,casue_G2

