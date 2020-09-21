# -*- coding：utf-8 -*-
import json
from transformers.data.metrics.squad_metrics import (
    compute_f1,compute_exact
)
import os
import random

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

def make_find_answer_by_rule(no_label_synthetic,output):
    Object_question_words ={"which",'in which','whose','who','what','for which','on which'," which",'when','at which','where','during which','when'}
    positive_words = ['more', 'higher', 'increase', 'high', 'harder', 'increasing', 'up', 'larger', 'better', 'faster', 'stronger', 'closer', 'louder', 'correctly']
    negative_words = ['less', 'lower', 'decrease', 'low', 'easier', 'decreasing', 'down', 'smaller', 'worse', 'slower', 'weaker', 'farther', 'quieter', 'incorrectly','fewer','not','avoid']
    comparative_words = [
        'more', 'less', 'higher', 'lower', 'increase', 'decrease', 'harder', 'easier', 'increasing',
        'decreasing', 'larger', 'smaller', 'better', 'worse', 'faster', 'slower', 'weaker', 'stronger',
        'closer', 'farther', 'louder', 'quieter', 'correctly', 'incorrectly', 'not', 'yes', 'no', 'not'
    ]
    pairs = {
        'more': 'less',
        'higher': 'lower',
        'increase': 'decrease',
        'harder': 'easier',
        'increasing': 'decreasing',
        'larger': 'smaller',
        'better': 'worse',
        'faster': 'slower',
        'stronger': 'weaker',
        'closer': 'farther',
        'louder': 'quieter',
        'correctly': 'incorrectly',
        'increased':"reduced",
        "warmer":"colder",
        'high':'low',
        'turn on':"turn off",
        'rise':'fall',
        'up':'down',
        'longer':'shorter',
        'deeper':"shallower",
        'positively':'negatively',
    }
    final_answer = {}
    f1 =[]
    unsolved= []
    exact = []
    with open(no_label_synthetic, "r", encoding="utf-8")as reader:
        input_data = json.load(reader)
    for entry in input_data["data"]:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["background"]
            situation_text = paragraph['situation']
            for qa in paragraph['qas']:
                qas_id = qa['id']
                question_text = qa['question']
                try:
                    answer_text = qa['answers'][0]['text']
                except:
                    answer_text="null"
                predicts = qa['predicts']

                # Object Type:
                if any(question_text.lower().startswith(i) for i in Object_question_words):
                    predicts_answer = make_choice_pos(predicts)
                    for word in positive_words:
                        if word in question_text.lower():
                            predicts_answer = make_choice_pos(predicts)
                            break
                    for word in negative_words:
                        if word in question_text.lower():
                            if word not in predicts['TP in back'].lower():
                                predicts_answer = make_choice_neg(predicts)
                                break
                    f1.append(compute_f1(remove_punc(answer_text), remove_punc(predicts_answer)))

                else:
                    candidate = []
                    for key, val in pairs.items():
                        if key in question_text.lower() and val in question_text.lower():
                            candidate = [key, val]
                            break
                    # comparative
                    if candidate:
                        o1_ind = question_text.lower().find(remove_punc(predicts["object1"].lower()))
                        o2_ind = question_text.lower().find(remove_punc(predicts["object2"].lower()))
                        than_ind = question_text.lower().find("than")
                        indx = [o1_ind, o2_ind, than_ind]

                        # if (o1_ind == -1 and o2_ind == -1) or o1_ind == o2_ind:
                        #     # print(qas_id,question_text)
                        #     continue
                        if o1_ind != -1 and o2_ind == -1:
                            if than_ind != -1:
                                if o1_ind < than_ind:
                                    o2_ind = 1000
                        elif o1_ind == -1 and o2_ind != -1:
                            if than_ind != -1:
                                if o2_ind < than_ind:
                                    o1_ind = 1000

                        if o1_ind < o2_ind:
                            if int(predicts['TP_relevance']) == 0:
                                predicts_answer = candidate[0]
                                f1.append(compute_f1(answer_text, candidate[0]))
                            else:
                                predicts_answer = candidate[1]
                                f1.append(compute_f1(answer_text, candidate[1]))
                        else:
                            if int(predicts['TP_relevance']) == 0:
                                predicts_answer = candidate[1]
                                f1.append(compute_f1(answer_text, candidate[1]))
                            else:
                                predicts_answer = candidate[0]
                                f1.append(compute_f1(answer_text, candidate[0]))

                    else:
                        unsolved.append([qas_id,question_text,answer_text])

                        predicts_answer = predicts["object1"] +predicts["object2"]
                                          # +predicts["SP_object1"]+predicts["SP_object2"]
                        than_ind = question_text.lower().find("or")
                        o1_f1 = compute_f1(question_text,predicts["object1"])
                        o2_f1 = compute_f1(question_text, predicts["object2"])
                        sp_o1_f1 = compute_f1(question_text, predicts["SP_object1"])
                        sp_o2_f1 = compute_f1(question_text, predicts["SP_object2"])
                        posssible_answer = [predicts["object1"] ,predicts["object2"],predicts["SP_object1"],predicts["SP_object2"]]
                        predicts_answer = posssible_answer[[o1_f1,o2_f1,sp_o1_f1,sp_o2_f1].index(max([o1_f1,o2_f1,sp_o1_f1,sp_o2_f1]))]
                        f1.append(compute_f1(remove_punc(answer_text), predicts_answer))
                predicts_answer = remove_punc(predicts_answer)
                final_answer[qas_id] = predicts_answer
                exact.append(compute_exact(remove_punc(answer_text),remove_punc(predicts_answer)))
    with open(output, "w+") as writer:
        writer.write(json.dumps(final_answer, indent=4) + "\n")
    return final_answer,f1,exact



def make_find_answer_by_rule_uncovered(no_label_synthetic,output):
    Object_question_words ={"which",'in which','whose','who','what','for which','on which'," which",'when','at which','where','during which','when'}
    positive_words = ['more', 'higher', 'increase', 'high', 'harder', 'increasing', 'up', 'larger', 'better', 'faster', 'stronger', 'closer', 'louder', 'correctly']
    negative_words = ['less', 'lower', 'decrease', 'low', 'easier', 'decreasing', 'down', 'smaller', 'worse', 'slower', 'weaker', 'farther', 'quieter', 'incorrectly','fewer','not','avoid']
    comparative_words = [
        'more', 'less', 'higher', 'lower', 'increase', 'decrease', 'harder', 'easier', 'increasing',
        'decreasing', 'larger', 'smaller', 'better', 'worse', 'faster', 'slower', 'weaker', 'stronger',
        'closer', 'farther', 'louder', 'quieter', 'correctly', 'incorrectly', 'not', 'yes', 'no', 'not'
    ]
    pairs = {
        'more': 'less',
        'higher': 'lower',
        'increase': 'decrease',
        'harder': 'easier',
        'increasing': 'decreasing',
        'larger': 'smaller',
        'better': 'worse',
        'faster': 'slower',
        'stronger': 'weaker',
        'closer': 'farther',
        'louder': 'quieter',
        'correctly': 'incorrectly',
        'increased':"reduced",
        "warmer":"colder",
        'high':'low',
        'turn on':"turn off",
        'rise':'fall',
        'up':'down',
        'longer':'shorter',
        'deeper':"shallower",
        'positively':'negatively',
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
    final_answer = {}
    f1 =[]
    unsolved= []
    exact = []
    # writer_ref = open(os.path.join(output, "refs_filtered.txt"), 'w+', encoding='utf-8')
    # writer_hyps = open(os.path.join(out,"hyps_a_1_base.txt"),'w+',encoding='utf-8')
    writer_hyps = open(os.path.join(output,"hyps_rule_based"),'w+',encoding='utf-8')
    with open(no_label_synthetic, "r", encoding="utf-8")as reader:
        input_data = json.load(reader)
    for entry in input_data["data"]:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["background"]
            situation_text = paragraph['situation']
            for qa in paragraph['qas']:
                qas_id = qa['id']
                question_text = qa['question']
                try:
                    answer_text = qa['answers'][0]['text']
                except:
                    answer_text="null"
                predicts = qa['predicts']
                # if qas_id in filtered_id:
                #     continue
                # Object Type:
                if any(question_text.lower().startswith(i) for i in Object_question_words):
                    predicts_answer = make_choice_pos(predicts)
                    for word in positive_words:
                        if word in question_text.lower():
                            predicts_answer = make_choice_pos(predicts)
                            break
                    for word in negative_words:
                        if word in question_text.lower():
                            if word not in predicts['TP in back'].lower():
                                predicts_answer = make_choice_neg(predicts)
                                break
                    f1.append(compute_f1(remove_punc(answer_text), remove_punc(predicts_answer)))

                else:
                    candidate = []
                    for key, val in pairs.items():
                        if key in question_text.lower() and val in question_text.lower():
                            candidate = [key, val]
                            break
                    # comparative
                    if candidate:
                        o1_ind = question_text.lower().find(remove_punc(predicts["object1"].lower()))
                        o2_ind = question_text.lower().find(remove_punc(predicts["object2"].lower()))
                        than_ind = question_text.lower().find("than")
                        indx = [o1_ind, o2_ind, than_ind]

                        # if (o1_ind == -1 and o2_ind == -1) or o1_ind == o2_ind:
                        #     # print(qas_id,question_text)
                        #     continue
                        if o1_ind != -1 and o2_ind == -1:
                            if than_ind != -1:
                                if o1_ind < than_ind:
                                    o2_ind = 1000
                        elif o1_ind == -1 and o2_ind != -1:
                            if than_ind != -1:
                                if o2_ind < than_ind:
                                    o1_ind = 1000

                        if o1_ind < o2_ind:
                            if int(predicts['TP_relevance']) == 0:
                                predicts_answer = candidate[0]
                                f1.append(compute_f1(answer_text, candidate[0]))
                            else:
                                predicts_answer = candidate[1]
                                f1.append(compute_f1(answer_text, candidate[1]))
                        else:
                            if int(predicts['TP_relevance']) == 0:
                                predicts_answer = candidate[1]
                                f1.append(compute_f1(answer_text, candidate[1]))
                            else:
                                predicts_answer = candidate[0]
                                f1.append(compute_f1(answer_text, candidate[0]))

                    else:
                        unsolved.append([qas_id,question_text,answer_text])

                        predicts_answer = predicts["object1"] +predicts["object2"]
                                          # +predicts["SP_object1"]+predicts["SP_object2"]
                        than_ind = question_text.lower().find("or")
                        o1_f1 = compute_f1(question_text,predicts["object1"])
                        o2_f1 = compute_f1(question_text, predicts["object2"])
                        sp_o1_f1 = compute_f1(question_text, predicts["SP_object1"])
                        sp_o2_f1 = compute_f1(question_text, predicts["SP_object2"])
                        posssible_answer = [predicts["object1"] ,predicts["object2"],predicts["SP_object1"],predicts["SP_object2"]]
                        predicts_answer = posssible_answer[[o1_f1,o2_f1,sp_o1_f1,sp_o2_f1].index(max([o1_f1,o2_f1,sp_o1_f1,sp_o2_f1]))]
                        f1.append(compute_f1(remove_punc(answer_text), predicts_answer))
                predicts_answer = remove_punc(predicts_answer)
                ref2 = answer_text
                # writer_ref.write(ref2)
                # writer_ref.write('\n')
                writer_hyps.write(predicts_answer)
                writer_hyps.write('\n')
                final_answer[qas_id] = predicts_answer
                exact.append(compute_exact(remove_punc(answer_text),remove_punc(predicts_answer)))
    # writer_ref.close()
    writer_hyps.close()
    return final_answer,f1,exact