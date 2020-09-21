# -*- coding：utf-8 -*-
import json
from transformers.data.metrics.squad_metrics import (
    compute_f1
)
import os
import random
def read_predicted_file(predict_file,pred):
    Object_question_words ={"which",'in which','whose','who','what','for which','on which'}
    comparative_words =[
        'more', 'less', 'higher', 'lower', 'increase', 'decrease', 'high', 'low', 'harder', 'easier', 'increasing',
        'decreasing', 'up', 'down', 'larger', 'smaller','better','worse','faster','slower','weaker','stronger','closer','farther','louder','quieter','correctly','incorrectly','not','yes','no','not'
    ]
    positive_words = ['more', 'higher', 'increase', 'high', 'harder', 'increasing', 'up', 'larger', 'better', 'faster', 'stronger', 'closer', 'louder', 'correctly']
    negative_words = ['less', 'lower', 'decrease', 'low', 'easier', 'decreasing', 'down', 'smaller', 'worse', 'slower', 'weaker', 'farther', 'quieter', 'incorrectly','fewer','not','avoid']
    all_object_scores = []
    record={}
    pred_f1 =[]
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
    with open(predict_file, "r", encoding="utf-8")as reader:
        input_data = json.load(reader)
    for entry in input_data["data"]:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["background"]
            situation_text = paragraph['situation']
            for qa in paragraph['qas']:
                qas_id = qa['id']
                question_text =qa['question']
                answer_text = qa['answers'][0]['text']
                predicts = qa['predicts']
                if any(i==answer_text for i in comparative_words):
                    continue
                if qas_id in filtered_id:
                    continue
                if any(question_text.lower().startswith(i) for i in Object_question_words):
                    if max(compute_f1(answer_text, predicts['object1']),
                           compute_f1(answer_text, predicts['object2'])) == 0.0:
                        continue

                    predicts_answer = make_choice_pos(predicts)
                    for word in positive_words:
                        if word in question_text.lower():
                            predicts_answer = make_choice_pos(predicts)
                            break
                    for word in negative_words:
                        if word in question_text.lower():
                            if word not in predicts['TP in back'].lower():
                                predicts_answer = make_choice_neg(predicts)
                                # print(qas_id,predicts_answer)
                                break

                    f1 = compute_f1(remove_punc(answer_text),remove_punc(predicts_answer))
                    pred_f1.append(compute_f1(remove_punc(answer_text), remove_punc(pred[qas_id])))
                    record[qas_id] = [f1,answer_text,predicts_answer]
                    all_object_scores.append(f1)
                else:
                    continue
    return all_object_scores,record,pred_f1
def read_predicted_file_comparative(predict_file,pred):
    question_word = ['would','will']
    comparative_words = [
        'more', 'less', 'higher', 'lower', 'increase', 'decrease', 'harder', 'easier', 'increasing',
        'decreasing', 'larger', 'smaller', 'better', 'worse', 'faster', 'slower', 'weaker', 'stronger',
        'closer', 'farther', 'louder', 'quieter', 'correctly', 'incorrectly', 'not', 'yes', 'no', 'not'
    ]
    pairs = {
        'more':'less',
        'higher': 'lower',
        'increase':'decrease',
        'harder':'easier',
        'increasing':'decreasing',
        'larger':'smaller',
        'better':'worse',
        'faster':'slower',
        'stronger':'weaker',
        'closer':'farther',
        'louder':'quieter',
        'correctly': 'incorrectly',
    }
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
    cnt ={}
    prediction ={}
    f1 =[]
    pred_f1=[]
    debug={}
    with open(predict_file, "r", encoding="utf-8")as reader:
        input_data = json.load(reader)
    for entry in input_data["data"]:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["background"]
            situation_text = paragraph['situation']
            for qa in paragraph['qas']:
                candidate=[]
                qas_id = qa['id']
                question_text = qa['question']
                answer_text = qa['answers'][0]['text']
                predicts = qa['predicts']
                if not any(i == answer_text for i in comparative_words):
                    continue
                if qas_id in filtered_id:
                    continue
                # if any(question_text.lower().startswith(i) for i in question_word):
                #     continue
                for key, val in pairs.items():
                    if key in question_text.lower() and val in question_text.lower():
                        candidate= [key,val]
                        break
                if candidate!=[]:
                    o1_ind = question_text.lower().find(remove_punc(predicts["object1"].lower()))
                    o2_ind = question_text.lower().find(remove_punc(predicts["object2"].lower()))
                    than_ind = question_text.lower().find("than")
                    indx = [o1_ind,o2_ind,than_ind]
                    if (o1_ind== -1 and o2_ind ==-1 ) or o1_ind==o2_ind:
                        # print(qas_id,question_text)
                        continue
                    elif o1_ind!= -1 and o2_ind ==-1:
                        if than_ind != -1:
                            if o1_ind<than_ind:
                                o2_ind =1000

                    elif o1_ind == -1 and o2_ind != -1:
                        if than_ind != -1:
                            if o2_ind < than_ind:
                                o1_ind = 1000

                    if o1_ind < o2_ind:
                        if int(predicts['TP_relevance']) ==0:
                            prediction[qas_id] = [compute_f1(answer_text,candidate[0]),candidate[0],candidate,indx]
                            f1.append(compute_f1(answer_text,candidate[0]))
                        else:
                            prediction[qas_id] = [compute_f1(answer_text,candidate[1]),candidate[1],candidate,indx]
                            f1.append(compute_f1(answer_text, candidate[1]))
                    else:
                        if int(predicts['TP_relevance']) ==0:
                            prediction[qas_id] = [compute_f1(answer_text,candidate[1]),candidate[1],candidate,indx]
                            f1.append(compute_f1(answer_text, candidate[1]))
                        else:
                            prediction[qas_id] = [compute_f1(answer_text,candidate[0]),candidate[0],candidate,indx]
                            f1.append(compute_f1(answer_text, candidate[0]))
                    pred_f1.append(compute_f1(answer_text, pred[qas_id]))
                    cnt[qas_id] = [question_text,candidate]

    return pred_f1,f1,prediction
def make_choice_pos(predicts):
    if int(predicts['TP_relevance']) ==0:
        return predicts["object1"]
    else:
        return predicts["object2"]

def make_choice_neg(predicts):
    if int(predicts['TP_relevance']) ==1:
        return predicts["object1"]
    else:
        return predicts["object2"]

def remove_punc(text):
    ends =['.',',',"'s"]
    if text.endswith('.') or text.endswith(','):
        text = text[:-1]
    elif text.endswith("'s"):
        text = text[:-2]
    return text

def shuffle_and_make_CV(label_path,train_path,dev_path,out_path,fold =5,thresh=2500):
    t1 = []
    t2 = []
    t3 = []
    t4 = []
    t5 = []
    with open(label_path, "r", encoding="utf-8")as reader:
        input_data = json.load(reader)
    ignored_id=[]
    cnt = 0
    for entry in input_data["data"]:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["background"]
            situation_text = paragraph['situation']
            for qa in paragraph['qas']:
                qas_id = qa['id']
                ignored_id.append(qas_id)

    all_examples_except_label={}
    label_data =[]
    with open(train_path, "r", encoding="utf-8")as reader:
        input_data = json.load(reader)
    for entry in input_data["data"]:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["background"]
            is_label = False
            for qa in paragraph['qas']:
                qas_id = qa['id']
                if qas_id in ignored_id:
                    is_label = True
                else:
                    cnt +=1
            if not is_label:
                try:
                    all_examples_except_label[paragraph_text].append(paragraph)
                except:
                    all_examples_except_label[paragraph_text]=[paragraph]
            else:
                label_data.append(paragraph)
    with open(dev_path, "r", encoding="utf-8")as reader:
        input_data = json.load(reader)
    for entry in input_data["data"]:
        for paragraph in entry["paragraphs"]:
            is_label = False
            for qa in paragraph['qas']:
                qas_id = qa['id']
                if qas_id in ignored_id:
                    is_label = True
                else:
                    cnt += 1
            if not is_label:
                try:
                    all_examples_except_label[paragraph_text].append(paragraph)
                except:
                    all_examples_except_label[paragraph_text]=[paragraph]


    l = list(all_examples_except_label.items())
    random.shuffle(l)
    all_examples_except_label = dict(l)

    current=[]
    no_q = []
    current_len=0
    splitted= []
    total = 0
    for val in all_examples_except_label.values():
        for sit in val:
            current.append(sit)
            current_len +=len(sit['qas'])
            total+=len(sit['qas'])
        if current_len>=thresh and len(splitted)<fold-1:
            splitted.append(current)
            no_q.append(current_len)
            current_len =0
            current = []
        if len(splitted) == fold - 1:
            current.append(sit)
            current_len+=len(sit['qas'])
    no_q.append(current_len)
    splitted.append(current)
    print(total,cnt,no_q)
    next = input("ok?")
    if next !="y":
        pass
    else:
        for ind,split in enumerate(splitted):
            out_path_split =os.path.join(out_path,'split_'+str(ind)+'.json')
            writer = open(out_path_split,'w+',encoding='utf-8')
            out = {
            "version": "1.0",
            "data": [
                {
                    "title": "ropes",
                    "paragraphs": split,
                }]
        }
            writer.write(json.dumps(out,indent=4))
            writer.close()

        setting1 = label_data+splitted[0]+splitted[1]+splitted[2]
        setting2 = label_data+splitted[0]+splitted[1]+splitted[4]
        setting3 = label_data+splitted[0]+splitted[3]+splitted[4]
        setting4 = label_data+splitted[2]+splitted[3]+splitted[4]
        setting5 = label_data+splitted[1]+splitted[2]+splitted[3]
        settings = {
            '123':setting1,
            '125':setting2,
            '145':setting3,
            '345':setting4,
            '234':setting5
        }
        for key,val in settings.items():
            out_path_split = os.path.join(out_path, 'combine_' + str(key) + '.json')
            writer = open(out_path_split,'w+',encoding='utf-8')
            out = {
                "version": "1.0",
                "data": [
                    {
                        "title": "ropes",
                        "paragraphs": val,
                    }]
            }
            writer.write(json.dumps(out,indent=4))
            writer.close()

    return thresh,all_examples_except_label,label_data

def assign_new_value_for_shuffled(shuffled_path,train_path,dev_path,out_path):
    all_examples = {}
    with open(train_path, "r", encoding="utf-8")as reader:
        input_data = json.load(reader)
    for entry in input_data["data"]:
        for paragraph in entry["paragraphs"]:
            for qa in paragraph['qas']:
                qas_id = qa['id']
                all_examples[qas_id] = qa
    with open(dev_path, "r", encoding="utf-8")as reader:
        input_data = json.load(reader)
    for entry in input_data["data"]:
        for paragraph in entry["paragraphs"]:
            for qa in paragraph['qas']:
                qas_id = qa['id']
                all_examples[qas_id] = qa

    settings = ['123','125','145', '345', '234']
    for idx,combine in enumerate(settings):
        out_path_split = os.path.join(out_path, 'split_' + str(idx) + '.json')
        writer = open(out_path_split, 'w+', encoding='utf-8')
        with open(os.path.join(shuffled_path,'split_' + str(idx) + '.json'), "r", encoding="utf-8")as reader:
            input_data = json.load(reader)
        for entry in input_data["data"]:
            for paragraph in entry["paragraphs"]:
                for qa in paragraph['qas']:
                    qas_id = qa['id']
                    qa["synthetic_text"] = all_examples[qas_id]["synthetic_text"]
                    qa["predicts"] = all_examples[qas_id]["predicts"]
        writer.write(json.dumps(input_data,indent=4))
        writer.close()

        combine_path = os.path.join(out_path, 'combine_' + str(combine) + '.json')
        writer = open(combine_path, 'w+', encoding='utf-8')
        with open(os.path.join(shuffled_path,'combine_' + str(combine) + '.json'), "r", encoding="utf-8")as reader:
            input_data = json.load(reader)
        for entry in input_data["data"]:
            for paragraph in entry["paragraphs"]:
                for qa in paragraph['qas']:
                    qas_id = qa['id']
                    qa["synthetic_text"] = all_examples[qas_id]["synthetic_text"]
                    qa["predicts"] = all_examples[qas_id]["predicts"]
        writer.write(json.dumps(input_data,indent=4))
        writer.close()

    return None

def read_predicted_file_nn(predict_file,pred):
    Object_question_words ={"which",'in which','whose','who','what','for which','on which'}
    comparative_words =[
        'more', 'less', 'higher', 'lower', 'increase', 'decrease', 'high', 'low', 'harder', 'easier', 'increasing',
        'decreasing', 'up', 'down', 'larger', 'smaller','better','worse','faster','slower','weaker','stronger','closer','farther','louder','quieter','correctly','incorrectly','not','yes','no','not'
    ]
    positive_words = ['more', 'higher', 'increase', 'high', 'harder', 'increasing', 'up', 'larger', 'better', 'faster', 'stronger', 'closer', 'louder', 'correctly']
    negative_words = ['less', 'lower', 'decrease', 'low', 'easier', 'decreasing', 'down', 'smaller', 'worse', 'slower', 'weaker', 'farther', 'quieter', 'incorrectly','fewer','not','avoid']
    all_object_scores = []
    record={}
    pred_f1 =[]
    with open(predict_file, "r", encoding="utf-8")as reader:
        input_data = json.load(reader)
    for entry in input_data["data"]:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["background"]
            situation_text = paragraph['situation']
            for qa in paragraph['qas']:
                qas_id = qa['id']
                question_text =qa['question']
                answer_text = qa['answers'][0]['text']
                predicts = qa['predicts']
                if any(i==answer_text for i in comparative_words):
                    continue
                if any(question_text.lower().startswith(i) for i in Object_question_words):
                    # if max(compute_f1(answer_text, predicts['object1']),
                    #        compute_f1(answer_text, predicts['object2'])) == 0.0:
                    #     continue

                    predicts_answer = pred[qas_id]
                    # f1 = compute_f1(remove_punc(answer_text),remove_punc(predicts_answer))
                    pred_f1.append(compute_f1(remove_punc(answer_text), remove_punc(pred[qas_id])))
                    # record[qas_id] = [f1,answer_text,predicts_answer]
                    # all_object_scores.append(f1)
                else:
                    continue
    return all_object_scores,record,pred_f1
def read_predicted_file_comparative_nn(predict_file,pred):
    question_word = ['would','will']
    comparative_words = [
        'more', 'less', 'higher', 'lower', 'increase', 'decrease', 'harder', 'easier', 'increasing',
        'decreasing', 'larger', 'smaller', 'better', 'worse', 'faster', 'slower', 'weaker', 'stronger',
        'closer', 'farther', 'louder', 'quieter', 'correctly', 'incorrectly', 'not', 'yes', 'no', 'not'
    ]
    pairs = {
        'more':'less',
        'higher': 'lower',
        'increase':'decrease',
        'harder':'easier',
        'increasing':'decreasing',
        'larger':'smaller',
        'better':'worse',
        'faster':'slower',
        'stronger':'weaker',
        'closer':'farther',
        'louder':'quieter',
        'correctly': 'incorrectly',
    }

    cnt ={}
    prediction ={}
    f1 =[]
    pred_f1=[]
    debug={}
    with open(predict_file, "r", encoding="utf-8")as reader:
        input_data = json.load(reader)
    for entry in input_data["data"]:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["background"]
            situation_text = paragraph['situation']
            for qa in paragraph['qas']:
                candidate=[]
                qas_id = qa['id']
                question_text = qa['question']
                answer_text = qa['answers'][0]['text']
                predicts = qa['predicts']
                if not any(i == answer_text for i in comparative_words):
                    continue
                # if any(question_text.lower().startswith(i) for i in question_word):
                #     continue
                for key, val in pairs.items():
                    if key in question_text.lower() and val in question_text.lower():
                        candidate= [key,val]
                        break
                if candidate!=[]:
                    pred_f1.append(compute_f1(answer_text, pred[qas_id]))

    return pred_f1

def get_id(data1,data2):
    all_examples = []
    ids =[]
    with open(data1, "r", encoding="utf-8")as reader:
        input_data = json.load(reader)
    for entry in input_data["data"]:
        for paragraph in entry["paragraphs"]:
            for qa in paragraph['qas']:
                qas_id = qa['id']
                all_examples.append(qas_id)
    with open(data2, "r", encoding="utf-8")as reader:
        input_data = json.load(reader)
    for entry in input_data["data"]:
        for paragraph in entry["paragraphs"]:
            for qa in paragraph['qas']:
                qas_id = qa['id']
                if qas_id not in all_examples:
                    ids.append(qas_id)


    return ids

