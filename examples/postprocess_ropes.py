# -*- coding：utf-8 -*-
import json
import string
import regex as re


def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

rule = re.compile(r'.* or .*')
all_options = {}
def find_options_from_candidate_answers(input,example):
    for key,value in input.items():
        example_id = key
        top_answer = remove_punc(value[0]['text'])
        options_text =[]
        try:
            if top_answer == rule.match(top_answer).group():
                # print(top_answer)
                # print(top_answer.split(' '))
                ind = top_answer.split(' ').index('or')

                options_text.append(' '.join(top_answer.split(' ')[0:ind]))
                # print(options_text)
        except:
            options_text = [value[0]['text']]

        # options_probs= [value[0]['probability']]
        # rubbish=[]
        successive =0
        for candidates in value[1:]:
            if remove_punc(candidates['text']).lower() in ['the','an']:
                continue

            if remove_punc(candidates['text']) in  remove_punc(options_text[0]):
                successive+=1
                # rubbish.append(remove_punc(candidates['text']))
                continue

            if len(remove_punc(candidates['text'])) > len(options_text[0]) and  remove_punc(options_text[0]) in remove_punc(candidates['text']):
                # rubbish.append(candidates['text'])
                continue

            if len(options_text)==2:
                break
            else:
                successive=0
                options_text.append(candidates['text'])
        if len(options_text) !=2:
            # print(input)
            options_text.append("No answer")
        rewrite_q0,rewrite_q1=rewrite_question_from_candidate_answer(options_text,example)
        label = 0
        try:
            if example.answers[0]['text'] == options_text[0]:
                label=0
            elif example.answers[0]['text']== options_text[1]:
                label=1
            else:
                if options_text[0] not in example.answers[0]['text'] and options_text[1] in example.answers[0]['text']:
                    label=1
                elif options_text[0] in example.answers[0]['text'] and options_text[1] not in example.answers[0]['text']:
                    label=0
                else:
                    label=0
                    # label= 0
        except:
            label=0
    if label !=2:
        all_options = {
            "id":str(example_id),
            "context":example.context_text,
            "orig_answer":example.answers,
            "answer_text": options_text,
            "question_text":example.question_text,
            "rewrite_q0":rewrite_q0,
            "rewrite_q1":rewrite_q1,
            'answerKey':label
        }
    else:
        all_options=None
    return all_options


def rewrite_question_from_candidate_answer(options, example):
    question_text = remove_punc(example.question_text)
    if options[0] in question_text and  options[1] in question_text:
        q0_index = question_text.find(options[0])
        q1_index = question_text.find(options[1])
        first_rewrite_query = question_text[:q0_index] + question_text[q0_index + len(options[0]):]
        second_rewrite_query = question_text[:q1_index] + question_text[q1_index + len(options[1]):]
        first_rewrite_query = remove_or_in_question(first_rewrite_query)
        second_rewrite_query = remove_or_in_question(second_rewrite_query)
    else:
        first_rewrite_query = example.question_text + ' '+ options[0]
        second_rewrite_query = example.question_text + ' ' + options[1]
    return first_rewrite_query,second_rewrite_query

def remove_or_in_question(question):
    question_list =question.split()
    if "or" in question_list:
        question_list.pop(question_list.index('or'))

    return ' '.join(question_list)
