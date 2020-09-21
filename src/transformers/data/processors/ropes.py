# -*- coding：utf-8 -*-
import json
import logging
import os
from functools import partial
from multiprocessing import Pool, cpu_count
import regex as re
import numpy as np
from tqdm import tqdm
from nltk import tokenize
from ...file_utils import is_tf_available, is_torch_available
from ...tokenization_bert import whitespace_tokenize
from .utils import DataProcessor
from preprocess_ropes import return_top_k
from .squad import (_improve_answer_span,
                    _find_last_substring_index,
                    _check_is_max_context,
                    _new_check_is_max_context,
                    _is_whitespace,
                    _sentence_seperator,
                    )

if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)

def ropes_convert_example_to_features(example, max_seq_length, doc_stride, max_query_length, is_training):
    features = []
    if is_training and not example.is_impossible:
    # if not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        if example.answer_in_question is False:
            actual_text = " ".join(example.situation_tokens[start_position : (end_position + 1)])
        else:
            actual_text = " ".join(example.question_tokens[start_position: (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text)).lower()
        if actual_text.lower().find(cleaned_answer_text.lower()) == -1:
            logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
            print(example)
            return []
    # *******************B and S*****************
    b_tok_to_orig_index = []
    b_orig_to_tok_index = []
    all_background_tokens = []
    s_tok_to_orig_index = []
    s_orig_to_tok_index = []
    all_situation_tokens = []
    for (i, token) in enumerate(example.background_tokens):
        b_orig_to_tok_index.append(len(all_background_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            b_tok_to_orig_index.append(i)
            all_background_tokens.append(sub_token)

    for (i, token) in enumerate(example.situation_tokens):
        s_orig_to_tok_index.append(len(all_situation_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            s_tok_to_orig_index.append(i)
            all_situation_tokens.append(sub_token)
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []

    if example.s_first:
        all_doc_tokens = all_situation_tokens+all_background_tokens
        tok_to_orig_index = s_tok_to_orig_index + [x+len(s_tok_to_orig_index) for x in b_tok_to_orig_index]
        orig_to_tok_index = s_orig_to_tok_index + [x+len(s_orig_to_tok_index) for x in b_orig_to_tok_index]
        bs_seperator_position = len(all_situation_tokens)-1
        # print(all_doc_tokens[:bs_seperator_position+1])
        # print(all_situation_tokens)
    else:
        all_doc_tokens = all_background_tokens+ all_situation_tokens
        tok_to_orig_index = s_tok_to_orig_index + [x + len(s_tok_to_orig_index) for x in b_tok_to_orig_index]
        orig_to_tok_index = s_orig_to_tok_index + [x + len(s_orig_to_tok_index) for x in b_orig_to_tok_index]
        bs_seperator_position = len(all_background_tokens) - 1

    # *****************************Question********************
    q_tok_to_orig_index = []
    q_orig_to_tok_index = []
    all_query_tokens = []
    for (i, token) in enumerate(example.question_tokens):
        q_orig_to_tok_index.append(len(all_query_tokens))
        # try:
        #     sub_tokens = tokenizer.tokenize(token,add_prefix_space=True) if i!=0 else tokenizer.tokenize(token)
        # except:
        #     sub_tokens = tokenizer.tokenize(token)
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            q_tok_to_orig_index.append(i)
            all_query_tokens.append(sub_token)
    truncated_query = tokenizer.convert_tokens_to_ids(all_query_tokens)

    #  make guided Object, Object SP, Background_SP, Background_TP attention
    if example.s_first:
        object1_start = s_orig_to_tok_index[example.Object1_word_pos[0]]
        object1_end = s_orig_to_tok_index[example.Object1_word_pos[1]+1]-1 if example.Object1_word_pos[1] < len(example.situation_tokens) - 1 else len(all_situation_tokens) - 1

        object2_start = s_orig_to_tok_index[example.Object2_word_pos[0]]
        object2_end = s_orig_to_tok_index[example.Object2_word_pos[1]+1]-1 if example.Object2_word_pos[1] < len(example.situation_tokens) - 1 else len(all_situation_tokens) - 1

        SP_object1_start = s_orig_to_tok_index[example.SP_Object1_word_pos[0]]
        SP_object1_end = s_orig_to_tok_index[example.SP_Object1_word_pos[1] + 1] - 1 if example.SP_Object1_word_pos[1] < len(
            example.situation_tokens) - 1 else len(all_situation_tokens) - 1

        SP_object2_start = s_orig_to_tok_index[example.SP_Object2_word_pos[0]]
        SP_object2_end = s_orig_to_tok_index[example.SP_Object2_word_pos[1] + 1] - 1 if example.SP_Object2_word_pos[
                                                                                            1] < len(
            example.situation_tokens) - 1 else len(all_situation_tokens) - 1

        SP_Background_start = orig_to_tok_index[example.SP_Background_word_pos[0]+len(example.situation_tokens)]
        SP_Background_end = orig_to_tok_index[example.SP_Background_word_pos[1] +len(example.situation_tokens)+ 1] - 1 if example.SP_Background_word_pos[1] < len(example.background_tokens) - 1 else len(all_doc_tokens) - 1

        TP_Background_start = orig_to_tok_index[example.TP_Background_word_pos[0] + len(example.situation_tokens)]
        TP_Background_end = orig_to_tok_index[
                                example.TP_Background_word_pos[1] + len(example.situation_tokens) + 1] - 1 if \
        example.TP_Background_word_pos[1] < len(example.background_tokens) - 1 else len(all_doc_tokens) - 1



    if is_training and not example.is_impossible:
        if not example.answer_in_question:
            if example.s_first:
                tok_start_position = orig_to_tok_index[example.start_position]
                if example.end_position < len(example.situation_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
                else:
                    tok_end_position = len(all_situation_tokens) - 1
                (tok_start_position, tok_end_position) = _improve_answer_span(
                    all_situation_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
                )
            else:
                tok_start_position = orig_to_tok_index[example.start_position+len(example.background_tokens)]
                if example.end_position < len(example.situation_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position + 1+len(example.background_tokens)] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1

                (tok_start_position, tok_end_position) = _improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
                )
        else:
            tok_start_position = q_orig_to_tok_index[example.start_position]
            if example.end_position < len(example.question_tokens) - 1:
                tok_end_position = q_orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_query_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_query_tokens, tok_start_position, tok_end_position, tokenizer,
                example.answer_text)

    spans = []
    sequence_added_tokens = (
        tokenizer.max_len - tokenizer.max_len_single_sentence + 1
        if "roberta" in str(type(tokenizer))
        else tokenizer.max_len - tokenizer.max_len_single_sentence
    )

    sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair
    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):

        encoded_dict = tokenizer.encode_plus(
            truncated_query if tokenizer.padding_side == "right" else span_doc_tokens,
            span_doc_tokens if tokenizer.padding_side == "right" else truncated_query,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            pad_to_max_length=True,
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            truncation_strategy="only_second" if tokenizer.padding_side == "right" else "only_first",
        )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )
        # print(tokenizer.pad_token_id)
        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
        else:
            non_padded_ids = encoded_dict["input_ids"]


        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)
        token_to_orig_map = {}
        query_len = len(truncated_query)

        for i in range(query_len):
                index = i+1 if tokenizer.padding_side == "right" else paragraph_len+sequence_added_tokens+i
                token_to_orig_map[index] = q_tok_to_orig_index[i]

        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len
        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict:
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]


    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)
        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0) (not sure why...)
        p_mask = np.array(span["token_type_ids"])
        p_mask = np.minimum(p_mask, 1)

        if tokenizer.padding_side == "right":
            # Limit positive values to one
            p_mask = 1 - p_mask

        p_mask[np.where(np.array(span["input_ids"]) == tokenizer.sep_token_id)[0]] = 1
        # Set the CLS index to '0'
        p_mask[cls_index] = 0
        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
        # if not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if bs_seperator_position <= doc_start or bs_seperator_position >= doc_end:
                print("bad span since dont have situation and background at the same time")
                continue
            if not example.answer_in_question and not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True
            if example.answer_in_question and tok_end_position >= len(all_query_tokens):
                print("!"*100)
                # impossible
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:

                if not example.answer_in_question:

                    if tokenizer.padding_side == "left":
                        doc_offset = 0

                    else:
                        doc_offset = len(truncated_query) + sequence_added_tokens
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
                    bs_seperator_position = bs_seperator_position - doc_start + doc_offset



                else:

                    if tokenizer.padding_side == "left":
                        query_offset = span['length']+sequence_added_tokens
                        doc_offset = 0

                    else:
                        query_offset = 1
                        doc_offset = len(truncated_query) + sequence_added_tokens
                    start_position = tok_start_position + query_offset
                    end_position = tok_end_position + query_offset
                    bs_seperator_position = bs_seperator_position - doc_start + doc_offset

        if len(spans) >=2:
            print("woooooooooooooooo:")
            logger.info("answer in question ?  %s" % (example.answer_in_question))
            logger.info("answer text  %s" % (example.answer_text))
            logger.info("input_ids: %s" % " ".join([str(x) for x in span["input_ids"]]))
            logger.info("tokens: %s" % " ".join(span["tokens"]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in span["attention_mask"]]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in span["token_type_ids"]]))
            logger.info("token_to_orig_map: %s" % " ".join([
                "%d:%d" % (x, y) for (x, y) in span["token_to_orig_map"].items()]))
            if is_training:
                answer_text = " ".join(span["tokens"][start_position:(end_position + 1)])
                logger.info("start_position: %d" % (start_position))
                logger.info("end_position: %d" % (end_position))
                logger.info(
                    "current answer: %s" % (answer_text))

        features.append(
            RopesFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
                bs_seperator_position = bs_seperator_position,
                question_len=len(truncated_query),
                SP_relevance= example.relevance_SP,
                SP_TP_Polarity=example.polairty_SP_TP,
                TP_relevance= example.relevance_TP
            )
        )
    return features



def ropes_convert_example_to_features(example, max_seq_length, doc_stride, max_query_length, is_training):
    features = []
    if is_training and not example.is_impossible:

        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        if example.answer_in_question is False:
            actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
        else:
            actual_text = " ".join(example.question_tokens[start_position: (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text)).lower()
        if actual_text.lower().find(cleaned_answer_text.lower()) == -1:
            logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
            print(example)
            return []


    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []

    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        try:
            sub_tokens = tokenizer.tokenize(token,add_prefix_space=True) if i!=0 else tokenizer.tokenize(token)
        except:
            sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    q_tok_to_orig_index = []
    q_orig_to_tok_index = []
    all_query_tokens = []
    for (i, token) in enumerate(example.question_tokens):
        q_orig_to_tok_index.append(len(all_query_tokens))
        try:
            sub_tokens = tokenizer.tokenize(token,add_prefix_space=True) if i!=0 else tokenizer.tokenize(token)
        except:
            sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            q_tok_to_orig_index.append(i)
            all_query_tokens.append(sub_token)

    query_length = len(all_query_tokens)

    truncated_query = tokenizer.convert_tokens_to_ids(all_query_tokens)


    if is_training and not example.is_impossible:
    # if not example.is_impossible:
        if not example.answer_in_question:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1

            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
            )
        else:
            tok_start_position = q_orig_to_tok_index[example.start_position]
            if example.end_position < len(example.question_tokens) - 1:
                tok_end_position = q_orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_query_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_query_tokens, tok_start_position, tok_end_position, tokenizer,
                example.answer_text)
    spans = []


    sequence_added_tokens = (
        tokenizer.max_len - tokenizer.max_len_single_sentence + 1
        if "roberta" in str(type(tokenizer))
        else tokenizer.max_len - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):

        encoded_dict = tokenizer.encode_plus(
            truncated_query if tokenizer.padding_side == "right" else span_doc_tokens,
            span_doc_tokens if tokenizer.padding_side == "right" else truncated_query,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            pad_to_max_length=True,
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            truncation_strategy="only_second" if tokenizer.padding_side == "right" else "only_first",
        )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
        else:
            non_padded_ids = encoded_dict["input_ids"]


        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)
        token_to_orig_map = {}
        query_len = len(truncated_query)

        for i in range(query_len):
                index = i+1 if tokenizer.padding_side == "right" else paragraph_len+sequence_added_tokens+i
                token_to_orig_map[index] = q_tok_to_orig_index[i]

        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict:
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]


    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0) (not sure why...)
        p_mask = np.array(span["token_type_ids"])

        p_mask = np.minimum(p_mask, 1)

        if tokenizer.padding_side == "right":
            # Limit positive values to one
            p_mask = 1 - p_mask

        p_mask[np.where(np.array(span["input_ids"]) == tokenizer.sep_token_id)[0]] = 1

        # Set the CLS index to '0'
        p_mask[cls_index] = 0

        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
        # if not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not example.answer_in_question and not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True
            if example.answer_in_question and tok_end_position >= len(all_query_tokens):
                print("!"*100)
                # impossible
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:

                if not example.answer_in_question:
                    if tokenizer.padding_side == "left":
                        doc_offset = 0


                    else:
                        doc_offset = len(truncated_query) + sequence_added_tokens
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

                else:
                    if tokenizer.padding_side == "left":
                        query_offset = span['length']+sequence_added_tokens

                    else:
                        query_offset = 1
                    start_position = tok_start_position + query_offset
                    end_position = tok_end_position + query_offset

        features.append(
            RopesFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
            )
        )
    return features

def ropes_convert_example_to_features_with_labels(example, max_seq_length, doc_stride, max_query_length, is_training):
    features = []
    if is_training and not example.is_impossible:

        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        if example.answer_in_question is False:
            actual_text = " ".join(example.situation_tokens[start_position : (end_position + 1)])
        else:
            actual_text = " ".join(example.question_tokens[start_position: (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text)).lower()
        if actual_text.lower().find(cleaned_answer_text.lower()) == -1:
            logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
            print(example)
            return []

    # *****************************Question********************
    q_tok_to_orig_index = []
    q_orig_to_tok_index = []
    all_query_tokens = []
    for (i, token) in enumerate(example.question_tokens):
        q_orig_to_tok_index.append(len(all_query_tokens))

        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            q_tok_to_orig_index.append(i)
            all_query_tokens.append(sub_token)
    truncated_query = tokenizer.convert_tokens_to_ids(all_query_tokens)
    # *******************B and S*****************
    b_tok_to_orig_index = []
    b_orig_to_tok_index = []
    all_background_tokens = []
    s_tok_to_orig_index = []
    s_orig_to_tok_index = []
    all_situation_tokens = []
    for (i, token) in enumerate(example.background_tokens):
        b_orig_to_tok_index.append(len(all_background_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            b_tok_to_orig_index.append(i)
            all_background_tokens.append(sub_token)

    for (i, token) in enumerate(example.situation_tokens):
        s_orig_to_tok_index.append(len(all_situation_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            s_tok_to_orig_index.append(i)
            all_situation_tokens.append(sub_token)

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    if example.s_first:
        all_doc_tokens = all_situation_tokens+all_background_tokens
        tok_to_orig_index = s_tok_to_orig_index + [x+s_tok_to_orig_index[-1] for x in b_tok_to_orig_index]
        orig_to_tok_index = s_orig_to_tok_index + [x+s_orig_to_tok_index[-1] for x in b_orig_to_tok_index]
        bs_seperator_position = len(all_situation_tokens)-1

    if is_training and not example.is_impossible:
        if not example.answer_in_question:
            if example.s_first:
                tok_start_position = orig_to_tok_index[example.start_position]
                if example.end_position < len(example.situation_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
                else:
                    tok_end_position = len(all_situation_tokens) - 1
                (tok_start_position, tok_end_position) = _improve_answer_span(
                    all_situation_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
                )
            else:
                tok_start_position = orig_to_tok_index[example.start_position+len(example.background_tokens)]
                if example.end_position < len(example.situation_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position + 1+len(example.background_tokens)] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1

                (tok_start_position, tok_end_position) = _improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
                )
        else:
            tok_start_position = q_orig_to_tok_index[example.start_position]
            if example.end_position < len(example.question_tokens) - 1:
                tok_end_position = q_orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_query_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_query_tokens, tok_start_position, tok_end_position, tokenizer,
                example.answer_text)

    sequence_added_tokens = (
        tokenizer.max_len - tokenizer.max_len_single_sentence + 1
        if "roberta" in str(type(tokenizer))
        else tokenizer.max_len - tokenizer.max_len_single_sentence
    )
    # print(sequence_added_tokens) 3
    sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair

    # span_doc_tokens = all_doc_tokens

    if example.s_first and example.label is not None:
        object1_start = s_orig_to_tok_index[example.Object1_word_pos[0]]
        object1_end = s_orig_to_tok_index[example.Object1_word_pos[1]+1]-1 if example.Object1_word_pos[1] < len(example.situation_tokens) - 1 else len(all_situation_tokens) - 1

        object2_start = s_orig_to_tok_index[example.Object2_word_pos[0]]
        object2_end = s_orig_to_tok_index[example.Object2_word_pos[1]+1]-1 if example.Object2_word_pos[1] < len(example.situation_tokens) - 1 else len(all_situation_tokens) - 1

        SP_object1_start = s_orig_to_tok_index[example.SP_Object1_word_pos[0]]
        SP_object1_end = s_orig_to_tok_index[example.SP_Object1_word_pos[1] + 1] - 1 if example.SP_Object1_word_pos[1] < len(
            example.situation_tokens) - 1 else len(all_situation_tokens) - 1

        SP_object2_start = s_orig_to_tok_index[example.SP_Object2_word_pos[0]]
        SP_object2_end = s_orig_to_tok_index[example.SP_Object2_word_pos[1] + 1] - 1 if example.SP_Object2_word_pos[
                                                                                            1] < len(
            example.situation_tokens) - 1 else len(all_situation_tokens) - 1

        SP_Background_start = b_orig_to_tok_index[example.SP_Background_word_pos[0]]
        SP_Background_end = b_orig_to_tok_index[example.SP_Background_word_pos[1] + 1] - 1 if example.SP_Background_word_pos[1] < len(example.background_tokens) - 1 else len(all_background_tokens) - 1

        TP_Background_start = b_orig_to_tok_index[example.TP_Background_word_pos[0]]
        TP_Background_end = b_orig_to_tok_index[example.TP_Background_word_pos[1]+ 1] - 1 if example.TP_Background_word_pos[1] < len(example.background_tokens) - 1 else len(all_background_tokens) - 1
    else:
        object1_start = -1
        object1_end = -1
        object2_start = -1
        object2_end = -1
        SP_object1_start = -1
        SP_object1_end = -1
        SP_object2_start = -1
        SP_object2_end = -1
        SP_Background_start = -1
        SP_Background_end= -1
        TP_Background_start = -1
        TP_Background_end = -1



    valid_stride_for_background= max_seq_length-len(truncated_query)-sequence_pair_added_tokens-len(all_situation_tokens)
    background_spans = []
    background_tokens = all_background_tokens
    if len(all_background_tokens)>valid_stride_for_background:
        while len(background_spans)*doc_stride <len(background_tokens):
            background_encoded = tokenizer.encode_plus(
                background_tokens,
                max_length=valid_stride_for_background,
                return_overflowing_tokens=True,
                pad_to_max_length=False,
                add_special_tokens = False,
                stride=valid_stride_for_background - doc_stride,
                truncation_strategy="only_first",
            )
            current_back_tokens = tokenizer.convert_ids_to_tokens(background_encoded["input_ids"])
            background_encoded["start"] = len(background_spans) * doc_stride
            background_encoded["tokens"] = current_back_tokens
            background_encoded["len"] = len(background_encoded["tokens"])
            back_tok_to_orig_index =[]
            for i in range(len(background_encoded["tokens"])):
                back_tok_to_orig_index.append(b_tok_to_orig_index[(len(background_spans) * doc_stride)+i])
            background_encoded["b_tok_to_orig_index"] = back_tok_to_orig_index
            background_spans.append(background_encoded)
            if "overflowing_tokens" not in background_encoded:
                break
            background_tokens = background_encoded["overflowing_tokens"]
    else:
        background_encoded={}
        background_encoded["start"] = 0
        background_encoded["tokens"] = background_tokens
        background_encoded["b_tok_to_orig_index"] =b_tok_to_orig_index
        background_spans.append(background_encoded)

    spans = []
    for background_chunk_dict in background_spans:
        background_tokens = background_chunk_dict["tokens"]
        span_doc_tokens = all_situation_tokens+background_tokens
        # print(len(span_doc_tokens)+len(truncated_query)+sequence_pair_added_tokens)
        tok_to_orig_index = s_tok_to_orig_index +background_chunk_dict["b_tok_to_orig_index"]
        # print(tok_to_orig_index)
        n = 0
        while len(spans) * doc_stride < len(span_doc_tokens):
            encoded_dict = tokenizer.encode_plus(
                truncated_query if tokenizer.padding_side == "right" else span_doc_tokens,
                span_doc_tokens if tokenizer.padding_side == "right" else truncated_query,
                max_length=max_seq_length,
                return_overflowing_tokens=True,
                pad_to_max_length=True,
                stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
                truncation_strategy="only_second" if tokenizer.padding_side == "right" else "only_first",
            )

            paragraph_len = min(
                len(span_doc_tokens),
                max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
            )

            # print(tokenizer.pad_token_id)
            if tokenizer.pad_token_id in encoded_dict["input_ids"]:
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                non_padded_ids = encoded_dict["input_ids"]

            tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)
            token_to_orig_map = {}
            query_len = len(truncated_query)

            for i in range(query_len):
                    index = i+1 if tokenizer.padding_side == "right" else paragraph_len+sequence_added_tokens+i
                    token_to_orig_map[index] = q_tok_to_orig_index[i]

            for i in range(paragraph_len):
                index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
                # token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]
                token_to_orig_map[index] = tok_to_orig_index[i]

            encoded_dict["paragraph_len"] = paragraph_len
            encoded_dict["tokens"] = tokens
            encoded_dict["token_to_orig_map"] = token_to_orig_map
            encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
            encoded_dict["token_is_max_context"] = {}
            encoded_dict["start"] = background_chunk_dict["start"]
            encoded_dict["length"] = len(tokens)-len(truncated_query)-sequence_pair_added_tokens
            spans.append(encoded_dict)
            n+=1
            if "overflowing_tokens" not in encoded_dict:
                break
            span_doc_tokens = encoded_dict["overflowing_tokens"]
            assert n <= 1
            # # print(len(span_doc_tokens)+len(truncated_query)+sequence_pair_added_tokens)
            # tok_to_orig_index = s_tok_to_orig_index + background_chunk_dict["b_tok_to_orig_index"]


    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context


    bad_label = []
    # print(len(spans))
    if len(spans) >=2 and not is_training:
        spans = spans[:1]
    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)
        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0) (not sure why...)
        p_mask = np.array(span["token_type_ids"])
        p_mask = np.minimum(p_mask, 1)

        if tokenizer.padding_side == "right":
            # Limit positive values to one
            p_mask = 1 - p_mask

        p_mask[np.where(np.array(span["input_ids"]) == tokenizer.sep_token_id)[0]] = 1
        # Set the CLS index to '0'
        p_mask[cls_index] = 0
        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        situation_start = 0
        situation_end = len(all_situation_tokens)
        doc_start = span["start"]
        doc_end = span["start"] + span["length"] - len(all_situation_tokens) - 1
        situation_offset = len(truncated_query) + sequence_added_tokens
        background_offest = situation_offset + len(all_situation_tokens) - doc_start

        if is_training:
            out_of_span = False
            if not (SP_Background_start >= doc_start and SP_Background_end <= doc_end) or not (TP_Background_start >= doc_start and TP_Background_end <= doc_end):
                continue

            if example.answer_in_question and tok_end_position >= len(all_query_tokens):
                    print("!"*100)
                    out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:

                if not example.answer_in_question:

                    if tokenizer.padding_side == "left":
                        doc_offset = 0

                    else:
                        doc_offset = len(truncated_query) + sequence_added_tokens
                    start_position = tok_start_position - situation_start + doc_offset
                    end_position = tok_end_position - situation_start + doc_offset

                else:

                    if tokenizer.padding_side == "left":
                        query_offset = span['length']+sequence_added_tokens
                        doc_offset = 0

                    else:
                        query_offset = 1
                        doc_offset = len(truncated_query) + sequence_added_tokens
                    start_position = tok_start_position + query_offset
                    end_position = tok_end_position + query_offset

        doc_offset = 0 if tokenizer.padding_side == "left" else len(truncated_query) + sequence_added_tokens
        bs_seperator_position = bs_seperator_position - situation_start + doc_offset
        Object1_label = make_alignment_label(object1_start,object1_end,situation_offset,max_seq_length)
        Object2_label = make_alignment_label(object2_start,object2_end,situation_offset,max_seq_length)
        SP_Object1_label = make_alignment_label(SP_object1_start, SP_object1_end, situation_offset, max_seq_length)
        SP_Object2_label = make_alignment_label(SP_object2_start, SP_object2_end, situation_offset, max_seq_length)
        SP_Background_label = make_alignment_label(SP_Background_start, SP_Background_end, background_offest, max_seq_length)
        TP_Background_label = make_alignment_label(TP_Background_start, TP_Background_end, background_offest, max_seq_length)

        # print(Object1_label,Object2_label,SP_Object1_label,SP_Object2_label,SP_Background_label,TP_Background_label)
            # if len(spans)>=2:
                # logger.info("token_to_orig_map: %s" % " ".join([
            #     "%d:%d" % (x, y) for (x, y) in span["token_to_orig_map"].items()]))
        if is_training and (span["tokens"][TP_Background_start + background_offest:TP_Background_end + background_offest + 1] != all_background_tokens[TP_Background_start:TP_Background_end + 1] or span["tokens"][SP_Background_start + background_offest:SP_Background_end + background_offest + 1]!=all_background_tokens[SP_Background_start:SP_Background_end + 1]):
            print("NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO!")
            print(span["tokens"][TP_Background_start + background_offest:TP_Background_end + background_offest + 1])
            print(all_background_tokens[TP_Background_start:TP_Background_end + 1])
            print(span["tokens"][SP_Background_start + background_offest:SP_Background_end + background_offest + 1])
            print(all_background_tokens[SP_Background_start:SP_Background_end + 1])
            print("background_offest",background_offest)
            print("situation_offset",situation_offset)
            print("doc_start",doc_start)
            print(list(span["tokens"]))
            print(TP_Background_start,TP_Background_end)
            print(SP_Background_start, SP_Background_end)
            print(list(all_background_tokens))
            print([i['start']for i in background_spans])

        # print(len(spans))
        # logger.info("answer in question ?  %s" % (example.answer_in_question))
        # logger.info("answer text  %s" % (example.answer_text))
        # logger.info("input_ids: %s" % " ".join([str(x) for x in span["input_ids"]]))
        # logger.info("tokens: %s" % " ".join(span["tokens"]))
        # logger.info("input_mask: %s" % " ".join([str(x) for x in span["attention_mask"]]))
        # logger.info("segment_ids: %s" % " ".join([str(x) for x in span["token_type_ids"]]))
        # logger.info("token_to_orig_map: %s" % " ".join([
        #     "%d:%d" % (x, y) for (x, y) in span["token_to_orig_map"].items()]))
        # if is_training:
        #     answer_text = " ".join(span["tokens"][start_position:(end_position + 1)])
        #     logger.info("start_position: %d" % (start_position))
        #     logger.info("end_position: %d" % (end_position))
        #     logger.info(
        #         "current answer: %s" % (answer_text))
        # logger.info("para_len: %d" % (span["paragraph_len"]))
        # print(span["tokens"][bs_seperator_position+1:])
        # print(json.dumps(example.label,indent=2))
        # print("SEPERATOR",bs_seperator_position)
        # print(list(span["tokens"]))
        # print("background_offest",background_offest)
        # print("situation_offset",situation_offset)
        # print("doc_start",doc_start)
        # print("TP",span["tokens"][TP_Background_start+background_offest:TP_Background_end+background_offest+1])
        # print("SP_01",span["tokens"][SP_object1_start+situation_offset:SP_object1_end+situation_offset+1])
        # print("O1",span["tokens"][object1_start + situation_offset:object1_end + situation_offset + 1])
        # print(TP_Background_start,TP_Background_end)
        # print(all_background_tokens[TP_Background_start:TP_Background_end + 1])
        # print(all_doc_tokens[
        #       TP_Background_start + len(all_situation_tokens):TP_Background_end + len(all_situation_tokens) + 1])
        # print("SP",span["tokens"][SP_Background_start + background_offest:SP_Background_end + background_offest + 1])
        # print(all_background_tokens[SP_Background_start:SP_Background_end + 1])
        # print(all_doc_tokens[
        #       SP_Background_start + len(all_situation_tokens):SP_Background_end + len(all_situation_tokens) + 1])

        features.append(
            RopesFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
                bs_seperator_position = bs_seperator_position,
                question_len=len(truncated_query),
                SP_relevance= example.relevance_SP,
                SP_TP_Polarity=example.polairty_SP_TP,
                TP_relevance= example.relevance_TP,
                Object1_label = Object1_label,
                Object2_label = Object2_label,
                SP_Object1_label = SP_Object1_label,
                SP_Object2_label=SP_Object2_label,
                SP_Back_label=SP_Background_label,
                TP_Back_label=TP_Background_label,
            )
        )
    assert len(features)!=0
    return features

def chunk_background_into_pieces(valid_length,background_tokens):
    background_chunks = []
    i=0
    while len(background_chunks)*valid_length <len(background_tokens):
        background_chunks.append(background_tokens[i*valid_length:(i+1)*valid_length])
        i+=1
    return background_chunks


def ropes_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def ropes_convert_examples_to_features(
    examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training, return_dataset=False, threads=1
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset,
            if 'tf': returns a tf.data.Dataset
        threads: multiple processing threadsa-smi


    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`

    Example::

        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)

        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """

    # Defining helper methods
    features = []
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=ropes_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            ropes_convert_example_to_features_with_labels,
            # ropes_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=is_training,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert ropes examples to features",
            )
        )
    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(features, total=len(features), desc="add example index and unique id"):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    if return_dataset == "pt":
        if not is_torch_available():
            raise ImportError("Pytorch must be installed to return a pytorch dataset.")

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        # print(all_input_ids.shape)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_question_len = torch.tensor([f.question_len for f in features],dtype = torch.long)
        all_bs_seperator_position = torch.tensor([f.bs_seperator_position for f in features],dtype=torch.long)
        all_SP_relevance= torch.tensor([f.SP_relevance for f in features],dtype=torch.long)
        all_SP_TP_polarity = torch.tensor([f.SP_TP_Polarity for f in features], dtype=torch.long)
        all_TP_relevance = torch.tensor([f.TP_relevance for f in features], dtype=torch.long)
        all_Object1_label = torch.tensor([f.Object1_label for f in features], dtype=torch.long)
        all_Object2_label = torch.tensor([f.Object2_label for f in features], dtype=torch.long)
        all_SP_Object1_label= torch.tensor([f.SP_Object1_label for f in features], dtype=torch.long)
        all_SP_Object2_label= torch.tensor([f.SP_Object2_label for f in features], dtype=torch.long)
        all_SP_Back_label= torch.tensor([f.SP_Back_label for f in features], dtype=torch.long)
        all_TP_Back_label= torch.tensor([f.TP_Back_label for f in features], dtype=torch.long)

        if not is_training:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_example_index,
                all_cls_index,
                all_p_mask,
                all_question_len,
                all_bs_seperator_position,
                all_SP_TP_polarity,
                all_SP_relevance,
                all_Object1_label,
                all_Object2_label,
                all_SP_Object1_label,
                all_SP_Object2_label,
                all_SP_Back_label,
                all_TP_Back_label,
                all_TP_relevance
            )
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions,
                all_cls_index,
                all_p_mask,
                all_question_len,
                all_bs_seperator_position,
                all_SP_relevance,
                all_SP_TP_polarity,
                all_TP_relevance,
                all_Object1_label,
                all_Object2_label,
                all_SP_Object1_label,
                all_SP_Object2_label,
                all_SP_Back_label,
                all_TP_Back_label,
            )

        return features, dataset
    elif return_dataset == "tf":
        if not is_tf_available():
            raise ImportError("TensorFlow must be installed to return a TensorFlow dataset.")

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    {
                        "start_position": ex.start_position,
                        "end_position": ex.end_position,
                        "cls_index": ex.cls_index,
                        "p_mask": ex.p_mask,
                    },
                )

        return tf.data.Dataset.from_generator(
            gen,
            (
                {"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32},
                {"start_position": tf.int64, "end_position": tf.int64, "cls_index": tf.int64, "p_mask": tf.int32},
            ),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                {
                    "start_position": tf.TensorShape([]),
                    "end_position": tf.TensorShape([]),
                    "cls_index": tf.TensorShape([]),
                    "p_mask": tf.TensorShape([None]),
                },
            ),
        )

    return features

class RopesProcessor(DataProcessor):
    """
    Processor for the SQuAD data set.
    Overriden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and version 2.0 of SQuAD, respectively.
    """

    train_file = None
    dev_file = None

    def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
        if not evaluate:
            answer = tensor_dict["answers"]["text"][0].numpy().decode("utf-8")
            answer_start = tensor_dict["answers"]["answer_start"][0].numpy()
            answers = []
        else:
            answers = [
                {"answer_start": start.numpy(), "text": text.numpy().decode("utf-8")}
                for start, text in zip(tensor_dict["answers"]["answer_start"], tensor_dict["answers"]["text"])
            ]

            answer = None
            answer_start = None

        return RopesExample(
            qas_id=tensor_dict["id"].numpy().decode("utf-8"),
            question_text=tensor_dict["question"].numpy().decode("utf-8"),
            context_text=tensor_dict["context"].numpy().decode("utf-8"),
            answer_text=answer,
            start_position_character=answer_start,
            title=tensor_dict["title"].numpy().decode("utf-8"),
            answers=answers,
        )

    def get_examples_from_dataset(self, dataset, evaluate=False):
        """
        Creates a list of :class:`~transformers.data.processors.squad.SquadExample` using a TFDS dataset.

        Args:
            dataset: The tfds dataset loaded from `tensorflow_datasets.load("squad")`
            evaluate: boolean specifying if in evaluation mode or in training mode

        Returns:
            List of SquadExample

        Examples::

            import tensorflow_datasets as tfds
            dataset = tfds.load("squad")

            training_examples = get_examples_from_dataset(dataset, evaluate=False)
            evaluation_examples = get_examples_from_dataset(dataset, evaluate=True)
        """

        if evaluate:
            dataset = dataset["validation"]
        else:
            dataset = dataset["train"]

        examples = []
        for tensor_dict in tqdm(dataset):
            examples.append(self._get_example_from_tensor_dict(tensor_dict, evaluate=evaluate))

        return examples

    def get_train_examples(self, data_dir, filename=None,grounding_type=None):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """
        if data_dir is None:
            data_dir = ""

        if self.train_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
            os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            try:
                input_data = json.load(reader)["data"]
            except:
                input_data = json.load(reader)[0]
        return self._create_examples_with_label(input_data, "train",grounding_type)

    def get_dev_examples(self, data_dir, filename=None,grounding_type=None):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        if data_dir is None:
            data_dir = ""

        if self.dev_file is None:
            raise ValueError("RopesProcessor should be instantiated via RopesProcessor")

        with open(
            os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples_with_label(input_data, "dev",grounding_type)

    def _create_examples_raw(self, input_data, set_type, grounding_type=None):
        is_training = set_type == "train"
        examples = []
        for entry in tqdm(input_data):
            title = entry['title']
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["background"]
                situation_text = paragraph['situation']
                # context_text = paragraph_text + ' ' +situation_text
                for qa in paragraph['qas']:
                    qas_id = qa['id']
                    question_text = qa['question']
                    start_position_character = None
                    answer_text = None
                    answers = []
                    answer_in_question = False
                    synthetic_text = qa['synthetic_text']
                    if  grounding_type == "onlys":
                        context_text = situation_text
                    elif grounding_type =="synthetic":
                        context_text = synthetic_text+ '. '+situation_text
                    else:
                        context_text = paragraph_text + ' ' + situation_text

                    if "is_impossible" in qa:
                        is_impossible = qa["is_impossible"]
                    else:
                        is_impossible = False

                    if not is_impossible:
                        if is_training:
                            if grounding_type != "qfirst":
                                answer = qa["answers"][0]
                                answer_text = answer["text"]

                                answer_offset = _find_last_substring_index(answer_text, context_text)
                                if answer_offset is not None:
                                    answer_in_question = False
                                    start_position_character = answer_offset
                                else:
                                    # continue
                                    answer_offset = _find_last_substring_index(answer_text, question_text)
                                    if answer_offset is not None:
                                        answer_in_question = True
                                        start_position_character = answer_offset

                                if answer_offset is None:
                                    continue
                            else:
                                answer = qa["answers"][0]
                                answer_text = answer["text"]

                                answer_offset = _find_last_substring_index(answer_text, question_text)
                                if answer_offset is not None:
                                    answer_in_question = True
                                    start_position_character = answer_offset
                                else:
                                    # continue
                                    answer_offset = _find_last_substring_index(answer_text, context_text)
                                    if answer_offset is not None:
                                        answer_in_question = False
                                        start_position_character = answer_offset

                                if answer_offset is None:
                                    continue
                        else:
                            try:
                                answers = qa["answers"]
                                answer = answer[0]
                                answer_text = answer["text"]
                            except:
                                answers = [
                                    {
                                        "text": "Null"
                                    }
                                ]

                    example = RopesExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        background=paragraph_text,
                        situation=situation_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=False,
                        answers=answers,
                        answer_in_question=answer_in_question,
                        s_first=False,
                        labels=None,
                    )
                    examples.append(example)
        return examples

    def _create_examples_s(self, input_data, set_type,grounding_type=None):
        is_training = set_type == "train"
        examples = []
        s_first = False
        total = []
        cnt = 0
        for entry in tqdm(input_data):
            title = entry['title']
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["background"]
                situation_text = paragraph['situation']
                for qa in paragraph['qas']:
                    qas_id =qa['id']
                    question_text = qa['question']
                    labels = qa['labels']
                    start_position_character = None
                    answer_text = None
                    answers = []
                    answer_in_question=False
                    synthetic_text = qa['synthetic_text']
                    if grounding_type =="1hop":
                        paragraph_text = ' '.join(return_top_k(question_text,paragraph_text,k=3))
                        context_text = paragraph_text + ' ' + situation_text
                    elif grounding_type =="2hop":
                        paragraph_text = ' '.join(return_top_k(question_text+' '+situation_text,paragraph_text,k=3))
                        context_text = paragraph_text + ' ' + situation_text
                    elif grounding_type == "onlys":
                        context_text = situation_text
                    elif grounding_type == "synthetic":
                        context_text = situation_text
                    elif grounding_type == "s_first":
                        context_text = situation_text + ' ' +paragraph_text
                        s_first = True
                    else:
                        context_text = paragraph_text + ' ' + situation_text

                    if "is_impossible" in qa:
                        is_impossible = qa["is_impossible"]
                    else:
                        is_impossible = False

                    if not is_impossible:
                        if is_training:
                            if grounding_type != "qfirst":
                                answer = qa["answers"][0]
                                answer_text = answer["text"]
                                # if grounding_type == "s_first":
                                #     answer_offset = _find_last_substring_index(answer_text,situation_text)
                                # else:
                                #     answer_offset = _find_last_substring_index(answer_text, context_text)
                                answer_offset = _find_last_substring_index(answer_text, situation_text)

                                if answer_offset is not None:
                                    answer_in_question = False
                                    start_position_character = answer_offset
                                else:
                                    # continue
                                    answer_offset = _find_last_substring_index(answer_text,question_text)
                                    if answer_offset is not None:
                                        answer_in_question = True
                                        start_position_character = answer_offset

                                if answer_offset is None:
                                    print(answer_text)
                                    continue
                            else:
                                answer = qa["answers"][0]
                                answer_text = answer["text"]

                                answer_offset = _find_last_substring_index(answer_text, question_text)
                                if answer_offset is not None:
                                    answer_in_question = True
                                    start_position_character = answer_offset
                                else:
                                    # continue
                                    answer_offset = _find_last_substring_index(answer_text, situation_text)
                                    if answer_offset is not None:
                                        answer_in_question = False
                                        start_position_character = answer_offset

                                if answer_offset is None:
                                    # logger.warning(
                                    #     f"Couldn't find answer '{answer_text}' in " + f"question '{question_text}' or passage '{context_text}'")
                                    print(answer_text)
                                    continue
                        else:
                            try:
                                answers = qa["answers"]
                                answer_text = answers[0]["text"]
                            except:
                                answers = [
								{
									"text": "Null"
								}
							]

                    example = RopesExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        background=paragraph_text,
                        situation=situation_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                        answer_in_question=answer_in_question,
                        s_first = s_first)

                    examples.append(example)
                    total.append(len(example.question_tokens))
                    cnt+=1
        # print(max(total))
        # print(min(total))
        # print(sum(total)/cnt)
        return examples

    def _create_examples_with_label(self, input_data, set_type,grounding_type=None):
        is_training = set_type == "train"
        examples = []
        s_first = False
        total = []
        cnt = 0
        for entry in tqdm(input_data):
            title = entry['title']
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["background"]
                situation_text = paragraph['situation']
                for qa in paragraph['qas']:
                    qas_id =qa['id']
                    question_text = qa['question']
                    start_position_character = None
                    answer_text = None
                    answers = []
                    answer_in_question=False
                    if grounding_type == "s_first":
                        context_text = situation_text + ' ' +paragraph_text
                        s_first = True
                    else:
                        context_text = paragraph_text + ' ' + situation_text

                    if is_training:
                        try:
                            labels = qa['label']
                        except:
                            print("no label",qas_id)
                            continue
                        answer = qa["answers"][0]
                        answer_text = answer["text"]
                        answer_offset = _find_last_substring_index(answer_text, situation_text)

                        if answer_offset is not None:
                            answer_in_question = False
                            start_position_character = answer_offset
                        else:
                            # continue
                            answer_offset = _find_last_substring_index(answer_text,question_text)
                            if answer_offset is not None:
                                answer_in_question = True
                                start_position_character = answer_offset
                        if answer_offset is None:
                            print(answer_text)
                            continue
                    else:
                        try:
                            answers = qa["answers"]
                            answer_text = answers[0]["text"]
                        except:
                            answers = [
                            {
                                "text": "Null"
                            }
                        ]
                            answer_text = answers[0]["text"]
                        try:
                            labels = qa['label']
                        except:
                            labels = None

                    example = RopesExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        background=paragraph_text,
                        situation=situation_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=False,
                        answers=answers,
                        answer_in_question=answer_in_question,
                        s_first = s_first,
                        labels =labels,
                    )

                    examples.append(example)
                    total.append(len(example.question_tokens))
                    cnt+=1
        return examples


class RopesProcessor(RopesProcessor):
    train_file = "train-v1.0.json"
    dev_file = "dev-v1.0.json"


class RopesExample(object):
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
        answer_in_question: answer is in question, the start position is the index of answer appred in question text
    """

    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text,
        background,
        situation,
        start_position_character,
        title,
        answers=[],
        answer_in_question=False,
        is_impossible=False,
        s_first = False,
        labels=None,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.background = background
        self.situation = situation
        self.title = title
        self.answer_in_question = answer_in_question
        self.is_impossible = is_impossible
        self.answers = answers
        self.start_position, self.end_position = 0, 0
        self.s_first = s_first


        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        background_tokens = []
        background_char_to_word_offset = []
        prev_is_whitespace = True
        for c in self.background:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    background_tokens.append(c)
                else:
                    background_tokens[-1] += c
                prev_is_whitespace = False
            background_char_to_word_offset.append(len(background_tokens) - 1)

        situation_tokens = []
        situation_char_to_word_offset = []
        prev_is_whitespace = True
        for c in self.situation:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    situation_tokens.append(c)
                else:
                    situation_tokens[-1] += c
                prev_is_whitespace = False
            situation_char_to_word_offset.append(len(situation_tokens) - 1)


        question_tokens = []
        question_char_to_word_offset = []
        prev_is_whitespace = True
        for c in self.question_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    question_tokens.append(c)
                else:
                    question_tokens[-1] += c
                prev_is_whitespace = False
            question_char_to_word_offset.append(len(question_tokens) - 1)


        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset
        self.question_tokens = question_tokens
        self.question_char_to_word_offset = question_char_to_word_offset
        self.background_tokens = background_tokens
        self.background_char_to_word_offset = background_char_to_word_offset
        self.situation_tokens = situation_tokens
        self.situation_char_to_word_offset = situation_char_to_word_offset

        # Start end end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            if not answer_in_question:
                self.start_position = situation_char_to_word_offset[start_position_character]
                self.end_position = situation_char_to_word_offset[
                    min(start_position_character + len(answer_text) - 1, len(situation_char_to_word_offset) - 1)
                ]
            else:
                self.start_position = question_char_to_word_offset[start_position_character]
                self.end_position = question_char_to_word_offset[
                    min(start_position_character + len(answer_text) - 1, len(question_char_to_word_offset) - 1)
                ]
        if labels is not None:
            self.Object1_word_pos=[situation_char_to_word_offset[labels['Object1_ind'][0]],
                                   situation_char_to_word_offset[min(labels['Object1_ind'][0] + len(labels['Object1']) - 1,
                             len(situation_char_to_word_offset) - 1)]]
            self.Object2_word_pos = [situation_char_to_word_offset[labels['Object2_ind'][0]],
                                     situation_char_to_word_offset[min(labels['Object2_ind'][0] + len(
                                         labels['Object2']),len(situation_char_to_word_offset) - 1)]]
            self.SP_Object1_word_pos = [situation_char_to_word_offset[labels['SP_O1_ind'][0]],
                                        situation_char_to_word_offset[min(labels['SP_O1_ind'][0] + len(
                                            labels['SP_O1']),len(situation_char_to_word_offset) - 1)]]
            self.SP_Object2_word_pos = [situation_char_to_word_offset[labels['SP_O2_ind'][0]],
                                        situation_char_to_word_offset[min(labels['SP_O2_ind'][0] + len(
                                         labels['SP_O2']),
                                         len(situation_char_to_word_offset) - 1)]]
            self.SP_Background_word_pos = [background_char_to_word_offset[labels['Back_SP_ind'][0]],
                                           background_char_to_word_offset[min(labels['Back_SP_ind'][0] + len(
                                            labels['Back_SP']),
                                            len(background_char_to_word_offset) - 1)]]

            self.TP_Background_word_pos = [background_char_to_word_offset[labels['Back_TP_ind'][0]],
                                           background_char_to_word_offset[min(labels['Back_TP_ind'][0] + len(
                                            labels['Back_TP']),
                                            len(background_char_to_word_offset) - 1)]]

            self.relevance_SP = labels["Relevance"]
            self.polairty_SP_TP = labels['Polarity']
            self.relevance_TP = labels["TP_Relevacne"]
            self.label = labels
        else:
            self.Object1_word_pos = None
            self.Object2_word_pos = None
            self.SP_Object1_word_pos=None
            self.SP_Object2_word_pos = None
            self.SP_Background_word_pos = None
            self.TP_Background_word_pos=None
            self.relevance_SP = 0
            self.polairty_SP_TP = 0
            self.relevance_TP = 0
            self.label  = None


def make_alignment_label(orig_start,orig_end,offset,max_seq_len):
    attention = [0] * max_seq_len
    if orig_start == -1 or orig_end == -1:
        # print("you are in dev, some label are zeros!")
        return attention
    else:
        newstart = orig_start+offset
        newend = orig_end +offset
        attention[newstart:newend+1]=[1]*(newend-newstart+1)
        if len(attention) > max_seq_len:
            return attention[0:max_seq_len]
        else:
            return attention


class RopesFeatures(object):
    """
    Single squad example features to be fed to a model.
    Those features are model-specific and can be crafted from :class:`~transformers.data.processors.squad.SquadExample`
    using the :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        start_position,
        end_position,
        bs_seperator_position,
        question_len,
        SP_relevance,
        SP_TP_Polarity,
        TP_relevance,
        Object1_label,
        Object2_label,
        SP_Object1_label,
        SP_Object2_label,
        SP_Back_label,
        TP_Back_label,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position
        self.bs_seperator_position = bs_seperator_position
        self.question_len = question_len
        self.SP_relevance =SP_relevance
        self.SP_TP_Polarity = SP_TP_Polarity
        self.TP_relevance = TP_relevance
        self.Object1_label=Object1_label
        self.Object2_label = Object2_label
        self.SP_Object1_label = SP_Object1_label
        self.SP_Object2_label = SP_Object2_label
        self.SP_Back_label = SP_Back_label
        self.TP_Back_label = TP_Back_label


        # self.situation_len= situation_len
        # self.background_len = background_len



def make_new_file_with_synthetic_text(filename_path,synthetic_dict):
    with open(filename_path, "r", encoding="utf-8") as reader:
        input_data = json.load(reader)
    for entry in input_data["data"]:
        for paragraph in entry["paragraphs"]:
            for qa in paragraph['qas']:
                qas_id = qa['id']
                synthetic_text = synthetic_dict[qas_id]["constructed text"]
                predict = synthetic_dict[qas_id]["predicts"]
                qa['synthetic_text'] = synthetic_text
                qa['predicts'] = predict
    return input_data



