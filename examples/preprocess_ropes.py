# -*- coding：utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.linalg import norm
from nltk.corpus import stopwords
import json
from nltk import tokenize



def tfidf_similarity(s1, s2):
    cv = TfidfVectorizer(stop_words=stopwords.words('english'),ngram_range=[1,2])
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    if norm(vectors[0]) * norm(vectors[1]) == 0:
        return 0.0
    else:
        return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))

# return top k sentence from b in increasing order
def return_top_k(q,b,k=3):
    b = [i for i in tokenize.sent_tokenize(b)]
    final_sent = {}
    for sent in b:
        try:
            sent_scores = tfidf_similarity(q, str(sent))
        except:
            sent_scores = 0.0
        final_sent[sent] = sent_scores
    another_sort_sent = sorted(final_sent, key=final_sent.get,reverse=False)
    corpus_sents = another_sort_sent[-k:]
    return corpus_sents

def make_train_dev_sentence_filter(path,path2,k=3):
    reader = open(path, 'r', encoding='utf-8')
    input_data = json.load(reader)
    input_data = input_data["data"]
    for entry in input_data:
        paras = []
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["background"]
            situation_text = paragraph['situation']
            qas = paragraph['qas']
            new = []
            for qa in qas:
                qas_id = qa['id']
                question_text = qa['question']
                paragraph_text = ' '.join(return_top_k(question_text, paragraph_text, k=k))
                # paragraph_text = ' '.join(return_top_k(question_text+' '+situation_text, paragraph_text, k=k))
                paragraph['background']=paragraph_text
    output = {
        "version": "1.0",
        "data": input_data
    }
    writer = open(path2, 'w', encoding='utf-8')
    writer.write(json.dumps(output, indent=4, ensure_ascii=False))
    writer.close()


def make_answer_prediction(path,path2):
    reader = open(path, 'r', encoding='utf-8')
    input_data = json.load(reader)
    input_data = input_data["data"]
    output ={}
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            qas = paragraph['qas']
            for qa in qas:
                answer = qa["answers"][0]
                answer_text = answer["text"]
                qasid = qa["id"]
                output[qasid]=answer_text

    writer = open(path2, 'w', encoding='utf-8')
    writer.write(json.dumps(output, indent=4, ensure_ascii=False))
    writer.close()


def find_wrong_examples(path,path1,path2,path3,path4,path5):
    reader1 = open(path1, 'r', encoding='utf-8')
    input_data1 = json.load(reader1)
    reader2 = open(path2, 'r', encoding='utf-8')
    input_data2 = json.load(reader2)
    reader3 = open(path3, 'r', encoding='utf-8')
    input_data3 = json.load(reader3)
    reader4 = open(path4, 'r', encoding='utf-8')
    input_data4 = json.load(reader4)

    reader = open(path, 'r', encoding='utf-8')
    input_data = json.load(reader)
    input_data = input_data["data"]
    output ={}
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            qas = paragraph['qas']
            paragraph_text = paragraph["background"]
            situation_text = paragraph['situation']
            for qa in qas:
                question_text = qa['question']
                answer = qa["answers"][0]
                answer_text = answer["text"]
                qasid = qa["id"]
                if input_data1[qasid] != answer_text and input_data2[qasid] != answer_text and input_data3[qasid] != answer_text and input_data4[qasid] != answer_text :
                    output[qasid]={
                        "background": paragraph_text,
                        "situation":situation_text,
                        "question":question_text,
                        "answer_text":answer_text,
                    }
    writer = open(path5, 'w', encoding='utf-8')
    writer.write(json.dumps(output, indent=4, ensure_ascii=False))
    writer.close()


def count_token_number():
    pass