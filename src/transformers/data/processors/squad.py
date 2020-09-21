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

if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

def _find_last_substring_index(pattern_string, string):
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
    return res[-1] if res else None

def _find_all_substring_index(pattern_string, string):
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
    return res if res else None


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def _sentence_seperator(p):
    p_sents = tokenize.sent_tokenize(p)
    new_p = ' '.join([i for i in p_sents])
    return [_find_last_substring_index(i,new_p) for i in p_sents], new_p


def squad_convert_example_to_features(example, max_seq_length, doc_stride, max_query_length, is_training):
    features = []
    if is_training and not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
            return []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )

    spans = []

    truncated_query = tokenizer.encode(example.question_text, add_special_tokens=False, max_length=max_query_length)
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
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        features.append(
            SquadFeatures(
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


def squad_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def squad_convert_examples_to_features(
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
    with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            squad_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=is_training,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert squad examples to features",
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
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

        if not is_training:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_example_index, all_cls_index, all_p_mask
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

def ropes_convert_example_to_features(example, max_seq_length, doc_stride, max_query_length, is_training):
    features = []
    if is_training and not example.is_impossible:
    # if not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

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
        if example.start_positions_first is not None:
            tok_start_positions_first=[]
            tok_end_positions_first = []
            for start_first,end_first in zip(example.start_positions_first,example.end_positions_first):
                tok_start_position_first=q_orig_to_tok_index[start_first]
                if end_first< len(example.question_tokens) - 1:
                    tok_end_position_first = q_orig_to_tok_index[end_first + 1] - 1
                else:
                    tok_end_position_first = len(all_query_tokens) - 1
                (tok_start_position_first, tok_end_position_first) = _improve_answer_span(all_query_tokens, tok_start_position_first, tok_end_position_first, tokenizer,example.answer_text)
                tok_start_positions_first.append(tok_start_position_first)
                tok_end_positions_first.append(tok_end_position_first)
        else:
            tok_start_positions_first =[]
            tok_end_positions_first = []

        if example.start_positions_second is not None:
            tok_start_positions_second=[]
            tok_end_positions_second = []
            for start_second,end_second in zip(example.start_positions_second,example.end_positions_second):
                tok_start_position_second=orig_to_tok_index[start_second]
                if end_second< len(example.doc_tokens) - 1:
                    tok_end_position_second = orig_to_tok_index[end_second + 1] - 1
                else:
                    tok_end_position_second = len(all_doc_tokens) - 1
                (tok_start_position_second, tok_end_position_second) = _improve_answer_span(all_query_tokens, tok_start_position_second, tok_end_position_second, tokenizer,example.answer_text)
                tok_start_positions_second.append(tok_start_position_second)
                tok_end_positions_second.append(tok_end_position_second)
        else:
            tok_start_positions_second =[]
            tok_end_positions_second = []

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
        # 将word编码成id，就是tokenizer 功能，利用词表
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

    # 填补is max context 无所谓
    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context
    # span for 循环
    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)
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
        all_start_position,all_end_position= make_matrix_start_end(None,max_seq_length)
        if is_training and not span_is_impossible:
        # if not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if example.start_positions_first is None:
                if not example.answer_in_question and not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True

            valid_start_end_current_span = [[tok_start_second,tok_end_second] for tok_start_second,tok_end_second in zip(tok_start_positions_second,tok_end_positions_second) if tok_start_second >= doc_start and tok_end_second <= doc_end]


            if example.start_positions_first is not None:
                if not example.answer_in_question or not tok_start_positions_first:
                    if not valid_start_end_current_span:
                            out_of_span= True
                    else:
                        if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                            tok_start_position = valid_start_end_current_span[-1][0]
                            tok_end_position=valid_start_end_current_span[-1][1]

            if example.answer_in_question and tok_end_position >= len(all_query_tokens):
                print("!"*100)
                # impossible
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
                continue
            else:
                all_start_end_positions=[]
                # 如果不在问题中
                if not example.answer_in_question:
                    # 如果是P+Q这种顺序,不用添加任何
                    if tokenizer.padding_side == "left":
                        doc_offset = 0

                    # 如果是Q+P这种顺序
                    else:
                        doc_offset = len(truncated_query) + sequence_added_tokens
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
                # 如果在问题中
                else:
                    # 如果是P+Q, 着offset变成len(p) + 分隔符顺序
                    if tokenizer.padding_side == "left":
                        query_offset = span['length']+sequence_added_tokens
                    # 如果是Q+P,offest就是1(cls)
                    else:
                        query_offset = 1
                    start_position = tok_start_position + query_offset
                    end_position = tok_end_position + query_offset
                if tok_start_positions_first:
                    query_offset =1
                    for s,e in zip(tok_start_positions_first,tok_end_positions_first):
                        all_start_end_positions.append([s+query_offset,e+query_offset])
                if valid_start_end_current_span:
                    doc_offset = len(truncated_query) + sequence_added_tokens
                    for [s, e] in valid_start_end_current_span:
                        all_start_end_positions.append([s- doc_start+doc_offset,e -doc_start+doc_offset])

            # print("heres are all answer index",all_start_end_positions)
            # print("only one orignal answer index",[start_position,end_position])
            # print("answers",example.answer_text)
                all_start_position, all_end_position = make_matrix_start_end(all_start_end_positions, max_seq_length)

        # print("woooooooooooooooo:")
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
                all_start_positions=all_start_position,
                all_end_positions=all_end_position
            )
        )
    return features


def ropes_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def ropes_convert_examples_to_features(
    examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training, return_dataset=False, threads=1
):

    # Defining helper methods
    features = []
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=ropes_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            ropes_convert_example_to_features,
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
    print("len of feature",len(features))
    del new_features
    if return_dataset == "pt":
        if not is_torch_available():
            raise ImportError("Pytorch must be installed to return a pytorch dataset.")

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)


        if not is_training:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_example_index, all_cls_index, all_p_mask
            )
        else:
            print("builting train dataset")
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            all_multi_start_position = torch.tensor([f.all_start_positions for f in features], dtype=torch.float)
            all_multi_end_position = torch.tensor([f.all_end_positions for f in features], dtype=torch.float)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions,
                all_cls_index,
                all_p_mask,
                all_multi_start_position,
                all_multi_end_position,
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

class SquadProcessor(DataProcessor):
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

        return SquadExample(
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

    def get_train_examples(self, data_dir, filename=None):
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
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "train")

    def get_dev_examples(self, data_dir, filename=None):
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
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
            os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "dev")

    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        for entry in tqdm(input_data):
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    if "is_impossible" in qa:
                        is_impossible = qa["is_impossible"]
                    else:
                        is_impossible = False

                    if not is_impossible:
                        if is_training:
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            try:
                                answers = qa["answers"]
                            except:
                                answers = "Unknown"

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )

                    examples.append(example)
        return examples

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

        return SquadExample(
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



    def get_train_examples(self, data_dir, filename=None,grounding_type=None,multi_answer=False):
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
        return self._create_examples(input_data, "train",grounding_type,multi_answer)

    def get_dev_examples(self, data_dir, filename=None,grounding_type=None,multi_answer=False):
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
        return self._create_examples(input_data, "dev",grounding_type,multi_answer)

    def _create_examples(self, input_data, set_type,grounding_type=None,multi_answer=False):
        is_training = set_type == "train"
        examples = []
        multi_answer = multi_answer
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
                    start_position_characters_first=None
                    start_position_characters_second = None
                    try:
                        synthetic_text = qa['synthetic_text']
                    except:
                        synthetic_text = ''
                    if grounding_type is not None:
                        if  grounding_type == "onlys":
                            context_text = situation_text
                        elif grounding_type == "synthetic":
                            context_text = synthetic_text + '. ' + situation_text
                        elif grounding_type == "synthetic_2nd":
                            context_text = synthetic_text
                            question_text = question_text+' '+situation_text
                        elif grounding_type == "synthetic_1st":
                            context_text = question_text + ' ' + situation_text
                            question_text = synthetic_text
                        elif grounding_type == "onlyqsyn":
                            context_text = synthetic_text
                        elif grounding_type == "synback":
                            context_text = synthetic_text+' '+paragraph_text
                            question_text = question_text + ' ' + situation_text
                        elif grounding_type == "sb":
                            context_text = situation_text+' '+paragraph_text
                        elif grounding_type == "synsb":
                            context_text = situation_text +' '+ synthetic_text+'. '+ paragraph_text
                            # context_text = situation_text + ' '+ paragraph_text+' '+synthetic_text
                        elif grounding_type == "qs_synb":
                            context_text = synthetic_text + '.  ' + paragraph_text
                            question_text = question_text+' '+situation_text
                        else:
                            context_text =  paragraph_text+ ' ' + situation_text

                    if "is_impossible" in qa:
                        is_impossible = qa["is_impossible"]
                    else:
                        is_impossible = False

                    if not is_impossible:
                        if is_training:
                            if multi_answer:
                                answer = qa["answers"][0]
                                answer_text = answer["text"]
                                answer_offsets_first = _find_all_substring_index(answer_text, question_text)
                                answer_offsets_second = _find_all_substring_index(answer_text, context_text)
                                if answer_offsets_first is not None:
                                    start_position_characters_first = answer_offsets_first
                                if answer_offsets_second is not None:
                                    start_position_characters_second = answer_offsets_second

                            if grounding_type != "synthetind":
                                answer = qa["answers"][0]
                                answer_text = answer["text"]

                                answer_offset = _find_last_substring_index(answer_text,context_text)
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
                                    logger.warning(f"context first Couldn't find answer '{answer_text}' in " + f"question '{question_text}' or passage '{context_text}'")
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
                                    logger.warning(
                                        f"Question first,Couldn't find answer '{answer_text}' in " + f"question '{question_text}' or passage '{context_text}'")
                                    continue
                        else:
                            try:
                                answers = qa["answers"]
                            except:
                                answers = [
								{
									"text": "Null"
								}
							]

                    # print(context_text)
                    example = RopesExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                        answer_in_question=answer_in_question,
                        start_position_characters_first = start_position_characters_first,
                        start_position_characters_second= start_position_characters_second
                    )
                    examples.append(example)
        return examples





class SquadV1Processor(SquadProcessor):
    train_file = "train-v1.1.json"
    dev_file = "dev-v1.1.json"


class SquadV2Processor(SquadProcessor):
    train_file = "train-v2.0.json"
    dev_file = "dev-v2.0.json"

class RopesProcessor(RopesProcessor):
    train_file = "train-v1.0.json"
    dev_file = "dev-v1.0.json"

class SquadExample(object):
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
    """

    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text,
        start_position_character,
        title,
        answers=[],
        is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = 0, 0

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

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start end end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]

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
        start_position_character,
        title,
        answers=[],
        answer_in_question=False,
        is_impossible=False,
        start_position_characters_first= None,
        start_position_characters_second= None,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.answer_in_question = answer_in_question
        self.is_impossible = is_impossible
        self.answers = answers
        self.start_position, self.end_position = 0, 0
        self.start_positions_first,self.start_positions_second,self.end_positions_first,self.end_positions_second = None,None,None,None

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

        question_tokens = []
        question_char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
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

        # Start end end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            if not answer_in_question:
                self.start_position = char_to_word_offset[start_position_character]
                self.end_position = char_to_word_offset[
                    min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
                ]
            else:
                self.start_position = question_char_to_word_offset[start_position_character]
                self.end_position = question_char_to_word_offset[
                    min(start_position_character + len(answer_text) - 1, len(question_char_to_word_offset) - 1)
                ]
        if start_position_characters_first is not None:
            self.start_positions_first = [question_char_to_word_offset[i] for i in start_position_characters_first]
            self.end_positions_first = [question_char_to_word_offset[
                min(i + len(answer_text) - 1, len(question_char_to_word_offset) - 1)
            ] for i in start_position_characters_first]
        if start_position_characters_second is not None:
            self.start_positions_second = [char_to_word_offset[i] for i in start_position_characters_second]
            self.end_positions_second = [char_to_word_offset[
                                            min(i + len(answer_text) - 1, len(char_to_word_offset) - 1)
                                        ] for i in start_position_characters_second]

class SquadFeatures(object):
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

class RopesFeatures(object):


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
        all_start_positions,
        all_end_positions,
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

        self.all_start_positions = all_start_positions
        self.all_end_positions = all_end_positions

class SquadResult(object):
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.

    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logits=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits


def make_matrix_start_end(all_start_end,max_seq_len):
    start = [0.0] * max_seq_len
    end = [0.0]*max_seq_len
    if all_start_end is not None:
        for [s,e] in all_start_end:
            start[s]=1.0
            end[e]=1.0
    else:
        start[0]=1.0
        end[0]=1.0
    return start,end
