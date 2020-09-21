# coding: utf-8

#  The code for training and evaluating Interpretabile Reasoning Part.


import argparse
import glob
import logging
import os
import random
import timeit
import json
import numpy as np
import collections
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from modeling_ropes import RobertaForRopes
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForQuestionAnswering,
    RobertaTokenizer,
    XLMConfig,
    XLMForQuestionAnswering,
    XLMTokenizer,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits_include_question,
    squad_evaluate,
    get_final_text,
)
# from transformers.data.processors.squad import SquadResult, RopesProcessor, SquadV2Processor,    ropes_convert_examples_to_features
from transformers.data.processors.squad import SquadResult
from transformers.data.processors.ropes import RopesProcessor, ropes_convert_examples_to_features,make_new_file_with_synthetic_text
from anotate_ropes import compute_f1
from rule_based import make_find_answer_by_rule
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter



logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig, XLNetConfig, XLMConfig)),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForRopes, RobertaTokenizer),
    "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # freezed_layer=['roberta','find_object1','find_object2','find_TP']
    # freezed_layer=['find_object1','find_object2','find_TP']
    # freezed_layer = ['roberta']
    freezed_layer = []
    for name, param in model.named_parameters():
        param.requires_grad = False if any(fl in name for fl in freezed_layer) else True

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)) and not(any(fl in n for fl in freezed_layer))],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not(any(fl in n for fl in freezed_layer))], "weight_decay": 0.0},
    ]
    # print(len(optimizer_grouped_parameters[0]['params']))
    # print(optimizer_grouped_parameters)
    # exit()
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )



    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    TP_back_loss= 0.0
    logging_loss_TP = 0.0
    SP_back_loss, logging_loss_SP =0.0,0.0
    Pol_loss, logging_loss_Pol= 0.0,0.0

    SP_o1_loss, logging_loss_SPo1 =0.0,0.0
    SP_o2_loss, logging_loss_SPo2 = 0.0, 0.0
    o1_loss, logging_loss_o1 = 0.0, 0.0
    o2_loss, logging_loss_o2 = 0.0, 0.0
    Rel_loss, logging_loss_Rel= 0.0,0.0
    TP_Rel_loss,logging_loss_TP_Rel = 0.0,0.0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)

    #
    # for name, parms in model.named_parameters():
    #     if parms.requires_grad:
    #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
                "len_q":batch[7],
                "bs_seperator_index":batch[8],
                "sp_relevance":batch[9],
                "sp_tp_polarity":batch[10],
                "tp_relevance":batch[11],
                "object1_label":batch[12],
                "object2_label":batch[13],
                "SP_Object1_label":batch[14],
                "SP_Object2_label":batch[15],
                "SP_Back_label":batch[16],
                "TP_Back_label":batch[17]
            }

            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                if args.version_2_with_negative:
                    inputs.update({"is_impossible": batch[7]})


            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]
            debug_loss = outputs[1]
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            tr_loss += loss.item()
            # print(debug_loss)
            TP_back_loss += debug_loss['TP']
            SP_back_loss += debug_loss['SP']
            Pol_loss+= debug_loss['loss_pol']
            SP_o1_loss+= debug_loss['loss_SP_o1']
            SP_o2_loss += debug_loss['loss_SP_o2']
            o1_loss += debug_loss['object1']
            o2_loss += debug_loss['object2']
            Rel_loss +=debug_loss["loss_rel"]
            TP_Rel_loss += debug_loss["loss_on_TP"]
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info(
                        "Average loss: %s at global step: %s",
                        str((tr_loss - logging_loss) / args.logging_steps),
                        str(global_step),
                    )
                    logger.info(
                        "Average O1 loss: %s, O2 loss: %s, SPloss: %s, TP loss: %s, SP_o1: %s, SP_o2: %s, Rel loss: %s, Polarity loss: %s,TP relevance loss: %s at global step: %s",
                        str((o1_loss - logging_loss_o1) / args.logging_steps),
                        str((o2_loss - logging_loss_o2) / args.logging_steps),
                        str((SP_back_loss - logging_loss_SP) / args.logging_steps),
                        str((TP_back_loss - logging_loss_TP) / args.logging_steps),
                        str((SP_o1_loss - logging_loss_SPo1) / args.logging_steps),
                        str((SP_o2_loss - logging_loss_SPo2) / args.logging_steps),
                        str((Rel_loss - logging_loss_Rel) / args.logging_steps),
                        str((Pol_loss - logging_loss_Pol) / args.logging_steps),
                        str((TP_Rel_loss - logging_loss_TP_Rel) / args.logging_steps),
                        str(global_step),
                    )
                    # logger.info(
                    #     "current all loss: %s at global step: %s",
                    #     str(json.dumps(debug_loss,indent=2)),
                    #     str(global_step),
                    # )
                    logging_loss = tr_loss
                    logging_loss_SP=SP_back_loss
                    logging_loss_TP=TP_back_loss
                    logging_loss_Pol = Pol_loss
                    logging_loss_o1 = o1_loss
                    logging_loss_o2 = o2_loss
                    logging_loss_SPo1 = SP_o1_loss
                    logging_loss_SPo2 = SP_o2_loss
                    logging_loss_Rel= Rel_loss
                    logging_loss_TP_Rel = TP_Rel_loss

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        output_dir = os.path.join(args.output_dir, "epoch-{}".format(global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)

        # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", output_dir)
        # if args.max_steps > 0 and global_step > args.max_steps:
        #     train_iterator.close()
        #     break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def evaluate_all(args, model, tokenizer, prefix="",test=False):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
    results = {}
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Compute predictions
    if args.metadata_dir:
        output_rewrite_file = os.path.join(args.metadata_dir, "nolabels_synthetic.json")
    else:
        output_rewrite_file = os.path.join(args.output_dir, "nolabels_synthetic.json")

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    start_time = timeit.default_timer()
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    out_label_ids_rel = None
    out_label_ids_pol = None
    out_label_ids_tp_rel = None
    preds_rel = None
    preds_pol = None
    preds_tp_rel = None
    all_results=[]
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        # logger.info("one batch done!")
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2],
                "len_q":batch[6],
                "bs_seperator_index":batch[7],
                "sp_tp_polarity":batch[8],
                "sp_relevance":batch[9],
                "object1_label":batch[10],
                "object2_label":batch[11],
                "SP_Object1_label":batch[12],
                "SP_Object2_label":batch[13],
                "SP_Back_label":batch[14],
                "TP_Back_label":batch[15],
                "tp_relevance":batch[16],

            }
            example_indices = batch[3]
            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})

            outputs = model(**inputs)
            total_loss,all_loss,all_predicts = outputs[:3]
            eval_loss += total_loss
            object1 = all_predicts['object1']
            object2 = all_predicts['object2']
            TP = all_predicts['TP']
            SP = all_predicts['SP']
            SP_o1 = all_predicts['SP_o1']
            SP_o2 = all_predicts['SP_o2']
            pred_rel = all_predicts['relevance']
            pred_pol = all_predicts['polarity']
            pred_tp_rel = all_predicts['tp_relevance']
            p_o1 = all_predicts['p_o1']
            p_o2 = all_predicts['p_o2']
            p_TP = all_predicts["p_TP"]
            p_SP = all_predicts["p_SP"]
            p_sp_o1 = all_predicts["p_sp_o1"]
            p_sp_o2 = all_predicts["p_sp_o2"]
        nb_eval_steps += 1
        # find text

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RopesResult(
                unique_id,p_o1[i],p_o2[i],p_TP[i],p_SP[i],p_sp_o1[i],p_sp_o2[i],object1[i],object2[i],TP[i],SP[i],SP_o1[i],SP_o2[i],pred_rel[i].detach().cpu().numpy(),pred_pol[i].detach().cpu().numpy(),pred_tp_rel[i].detach().cpu().numpy()
            ))
        if preds_rel is None:
            preds_rel = pred_rel.detach().cpu().numpy()
            out_label_ids_rel = inputs["sp_relevance"].detach().cpu().numpy()
        else:
            preds_rel = np.append(preds_rel, pred_rel.detach().cpu().numpy(), axis=0)
            out_label_ids_rel = np.append(out_label_ids_rel, inputs["sp_relevance"].detach().cpu().numpy(), axis=0)
        if preds_pol is None:
            preds_pol = pred_pol.detach().cpu().numpy()
            out_label_ids_pol = inputs["sp_tp_polarity"].detach().cpu().numpy()
        else:
            preds_pol = np.append(preds_pol, pred_pol.detach().cpu().numpy(), axis=0)
            out_label_ids_pol = np.append(out_label_ids_pol, inputs["sp_tp_polarity"].detach().cpu().numpy(), axis=0)
        if preds_tp_rel is None:
            # print(pred_tp_rel.detach().cpu().numpy())
            preds_tp_rel = pred_tp_rel.detach().cpu().numpy()
            out_label_ids_tp_rel = inputs["tp_relevance"].detach().cpu().numpy()
        else:
            try:
                preds_tp_rel = np.append(preds_tp_rel, pred_tp_rel.detach().cpu().numpy(), axis=0)
            except:
                print([pred_tp_rel.detach().cpu().numpy()])
                preds_tp_rel = np.append(preds_tp_rel, [pred_tp_rel.detach().cpu().numpy()], axis=0)
            out_label_ids_tp_rel = np.append(out_label_ids_tp_rel, inputs["tp_relevance"].detach().cpu().numpy(), axis=0)
    eval_loss = eval_loss / nb_eval_steps
    # print(preds)
    preds_rel = np.argmax(preds_rel, axis=1)
    preds_pol = np.argmax(preds_pol, axis=1)
    preds_tp_rel = np.argmax(preds_tp_rel, axis=1)
    acc_rel = simple_accuracy(preds_rel, out_label_ids_rel)
    acc_pol = simple_accuracy(preds_pol, out_label_ids_pol)
    acc_tp_rel = simple_accuracy(preds_tp_rel, out_label_ids_tp_rel)
    f1_scores,average_scores,acc = compute_prediction(examples,features,all_results,tokenizer,output_rewrite_file,args)

    # print(preds,out_label_ids)
    result = {"eval_acc_relevance": acc_rel,
              "eval_acc_polarity":acc_pol,
              "eval_acc_tp_relevance":acc_tp_rel,}
    if f1_scores:
        o1_scores = sum(average_scores['object1'])/len(average_scores['object1'])
        o2_scores = sum(average_scores['object2']) / len(average_scores['object2'])
        o1_sp_scores = sum(average_scores['sp_o1']) / len(average_scores['sp_o1'])
        o2_sp_scores = sum(average_scores['sp_o2']) / len(average_scores['sp_o2'])
        SP_scores = sum(average_scores['SP']) / len(average_scores['SP'])
        TP_scores = sum(average_scores['TP']) / len(average_scores['TP'])
        good_o1 = [i for i in average_scores['object1'] if i >=0.5]
        good_o2 = [i for i in average_scores['object2'] if i >=0.5]
        good_tp = [i for i in average_scores['TP'] if i >=0.5]
        zero_tp = [i for i in average_scores['TP'] if i ==0]
        result.update({
            "o1": o1_scores,
            "o2": o2_scores,
            "sp_o1": o1_sp_scores,
            "sp_o2": o2_sp_scores,
            "SP": SP_scores,
            "TP": TP_scores,
            "No.O1 >0.5": len(good_o1) / len(average_scores['object1']),
            "No.O2 >0.5": len(good_o2) / len(average_scores['object2']),
            "No.TP >0.5":len(good_tp)/len(average_scores['TP']),
            "No.TP =0": len(zero_tp) / len(average_scores['TP']),
        })
    if acc:
        result.update(acc)
    if args.rule_based:
        output_prediction_file = os.path.join(args.output_dir, "predictions.json")
        _,f1,em = make_find_answer_by_rule(output_rewrite_file,output_prediction_file)
        print(sum(f1)/len(f1),sum(em)/len(em))
    return result

class RopesResult(object):
    def __init__(self, unique_id,p_o1,p_o2,p_TP,p_SP,p_sp_o1,p_sp_o2,object1,object2, TP,SP,SP_o1,SP_o2,relevance,polarity,tp_relevance):
        self.p_o1=p_o1
        self.p_o2 =p_o2
        self.p_TP=p_TP
        self.p_SP=p_SP
        self.p_sp_o1=p_sp_o1
        self.p_sp_o2=p_sp_o2
        self.unique_id = unique_id
        self.object1 = object1
        self.object2 = object2
        self.TP = TP
        self.SP = SP
        self.SP_o1 = SP_o1
        self.SP_o2 = SP_o2
        self.rel = relevance
        self.pol = polarity
        self.tp_rel = tp_relevance

def beautiful_preidction(tokens,predict_ids,tok_orig_map,parital_tokens,tokenizer,prob):
    # Not beautiful at all....
    direct_predict = tokens[predict_ids[0]:predict_ids[1]+1]
    try:
        orig_start = tok_orig_map[predict_ids[0]]
    except:
        print(tokens, predict_ids, tok_orig_map, parital_tokens)
        print("Start wrong")
        orig_start = tok_orig_map[predict_ids[0]-1]
        # exit()
    try:
        orig_end = tok_orig_map[predict_ids[1]]
    except:
        print("end too long predicts",predict_ids)
        try:
            orig_end = tok_orig_map[predict_ids[1]-1]
        except:
            orig_end = tok_orig_map[list(tok_orig_map.keys())[-1]]
            print(tokens, predict_ids, tok_orig_map, parital_tokens)
    orig_tokens = parital_tokens[orig_start : (orig_end + 1)]

    dirct_predict_text = tokenizer.convert_tokens_to_string(direct_predict)
    dirct_predict_text = " ".join(dirct_predict_text.split())

    orig_text = " ".join(orig_tokens)
    final_text = get_final_text(dirct_predict_text,orig_text,do_lower_case=True)
    if final_text ==".":
        if len(orig_text) > len(dirct_predict_text):
            final_text  = orig_text


    return dirct_predict_text,orig_text,final_text


def compute_prediction(
        all_examples,
        all_features,
        all_results,
        tokenizer,
        output_path,
        args
):
    writer = open(output_path,'w+',encoding='utf-8')
    # writer_final = open(os.path.join(output_path, '.json'), 'w+', encoding='utf-8')

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    output={}
    f1_scores = {}
    average_scores = {
        'object1':[],
        'object2':[],
        'sp_o1':[],
        'sp_o2':[],
        'SP':[],
        'TP':[],
    }
    total = 0
    relevance =0
    polarity = 0
    tprelevance = 0
    relasfinal = 0
    rules_step8 = 0
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        if len(features)>=1:
            for (feature_index, feature) in enumerate(features):
                # print(list(feature.tokens))
                tp_relevance = result.tp_rel
                result = unique_id_to_result[feature.unique_id]
                o1_tok_text, o1_orig_text, o1_final_text = beautiful_preidction(feature.tokens, result.object1,
                                                                               feature.token_to_orig_map,
                                                                               example.situation_tokens, tokenizer,result.p_o1)
                o2_tok_text, o2_orig_text, o2_final_text = beautiful_preidction(feature.tokens, result.object2,
                                                                               feature.token_to_orig_map,
                                                                               example.situation_tokens, tokenizer,result.p_o2)
                sp_o1_tok_text, sp_o1_orig_text, sp_o1_final_text = beautiful_preidction(feature.tokens, result.SP_o1,
                                                                               feature.token_to_orig_map,
                                                                               example.situation_tokens, tokenizer,result.p_sp_o1)
                sp_o2_tok_text, sp_o2_orig_text, sp_o2_final_text = beautiful_preidction(feature.tokens, result.SP_o2,
                                                                               feature.token_to_orig_map,
                                                                               example.situation_tokens, tokenizer,result.p_sp_o2)


                SP_tok_text, SP_orig_text, SP_final_text = beautiful_preidction(feature.tokens, result.SP,
                                                                               feature.token_to_orig_map,
                                                                               example.background_tokens, tokenizer,result.p_SP)
                TP_tok_text, TP_orig_text, TP_final_text = beautiful_preidction(feature.tokens, result.TP,
                                                                               feature.token_to_orig_map,
                                                                               example.background_tokens, tokenizer,result.p_TP)


                pred_rel = np.argmax(result.rel)
                pred_pol = np.argmax(result.pol)
                pred_tp_rel = np.argmax(tp_relevance)
                # print(result.rel,result.pol)
                # if pred_rel ==0:
                #     if pred_pol ==1:
                #         rules_tp_rel = 0
                #     else:
                #         rules_tp_rel =1
                #
                # if pred_rel == 1:
                #     if pred_pol == 1:
                #         rules_tp_rel = 1
                #     else:
                #         rules_tp_rel = 0
                if abs(result.rel[0] - result.rel[1]) >= 0.3:
                    if pred_rel == 0:
                        if pred_pol == 1:
                            rules_tp_rel = 0
                        else:
                            rules_tp_rel = 1
                    if pred_rel == 1:
                        if pred_pol == 1:
                            rules_tp_rel = 1
                        else:
                            rules_tp_rel = 0
                if abs(result.rel[0]-result.rel[1])>=0.3:
                    if pred_rel == 0:
                        if pred_pol == 1:
                            pred_tp_rel = 0
                        else:
                            pred_tp_rel = 1
                    if pred_rel == 1:
                        if pred_pol == 1:
                            pred_tp_rel = 1
                        else:
                            pred_tp_rel = 0

                if pred_tp_rel == 0:
                    predict_text  = o1_final_text+ ' has larger '+ TP_final_text + ' than '+o2_final_text

                if pred_tp_rel == 1:
                    predict_text  = o1_final_text+ ' has smaller '+ TP_final_text + ' than '+o2_final_text


                if example.label is not None:
                    qas_id = example.qas_id
                    o1_f1 = max(compute_f1(example.label['Object1'],o1_final_text),compute_f1(example.label['Object2'],o1_final_text))
                    o2_f1 = max(compute_f1(example.label['Object1'],o2_final_text),compute_f1(example.label['Object2'],o2_final_text))
                    sp_o1_f1 = max(compute_f1(example.label['SP_O1'],sp_o1_final_text),compute_f1(example.label['SP_O2'],sp_o1_final_text))
                    sp_o2_f1 = max(compute_f1(example.label['SP_O2'],sp_o2_final_text),compute_f1(example.label['SP_O1'],sp_o2_final_text))
                    SP_f1 = compute_f1(example.label['Back_SP'], SP_final_text)
                    TP_f1 = compute_f1(example.label['Back_TP'], TP_final_text)
                    if (compute_f1(example.label['Object1'], o1_final_text) < compute_f1(example.label['Object2'],o1_final_text)) and (compute_f1(example.label['Object1'],o2_final_text)>compute_f1(example.label['Object2'],o2_final_text)):
                        if pred_rel != example.label['Relevance']:
                            relevance +=1
                        if pred_pol !=example.label['Polarity']:
                            polarity+=1
                        if pred_tp_rel !=example.label['TP_Relevacne']:
                            tprelevance +=1
                        if pred_rel != example.label['TP_Relevacne']:
                            relasfinal +=1
                        if rules_tp_rel != example.label['TP_Relevacne']:
                            rules_step8 +=1
                        total+=1
                    else:
                        if pred_rel == example.label['Relevance']:
                            relevance +=1
                        if pred_pol ==example.label['Polarity']:
                            polarity+=1
                        # print(pred_tp_rel)
                        # print(example.label['TP_Relevacne'])
                        if pred_tp_rel ==example.label['TP_Relevacne']:
                            tprelevance +=1
                        if pred_rel == example.label['TP_Relevacne']:
                            relasfinal += 1
                        if rules_tp_rel == example.label['TP_Relevacne']:
                            rules_step8 += 1
                        total += 1

                    f1_scores[qas_id] = {
                        "o1": o1_f1,
                        "o2": o2_f1,
                        "sp_o1": sp_o1_f1,
                        "sp_o2": sp_o2_f1,
                        "SP": SP_f1,
                        "TP": TP_f1,
                    }
                    average_scores['object1'].append(o1_f1)
                    average_scores['object2'].append(o2_f1)
                    average_scores['sp_o1'].append(sp_o1_f1)
                    average_scores['sp_o2'].append(sp_o2_f1)
                    average_scores['SP'].append(SP_f1)
                    average_scores['TP'].append(TP_f1)
                    comparison = {
                        "object1":{"label":example.label['Object1'],
                                   "predict":o1_final_text,
                                   "f1": o1_f1,
                                    },
                        "object2": {"label": example.label['Object2'],
                                    "predict": o2_final_text,
                                    "f1": o2_f1,
                                    },
                        "SP_object1": {"label": example.label['SP_O1'],
                                    "predict": sp_o1_final_text,
                                       "f1": sp_o1_f1,
                                    },
                        "SP_object2": {"label": example.label['SP_O2'],
                                    "predict": sp_o2_final_text,
                                       "f1": sp_o2_f1,
                                    },
                        "SP in back": {"label": example.label['Back_SP'],
                                    "predict": SP_final_text,
                                       "f1": SP_f1,
                                    },
                        "TP in back": {"label": example.label['Back_TP'],
                                    "predict": TP_final_text,
                                       "f1":TP_f1,
                                    },
                    }

                    output[example.qas_id]={
                            'TP_relevance':str(pred_tp_rel == example.relevance_TP),
                            "tp_rel":str(pred_tp_rel),
                            "rel":str(pred_rel),
                            "pol":str(pred_pol),
                            "question":example.question_text,
                            "answer":example.answer_text,
                            "constructed text": predict_text,
                            "predicts":comparison
                        }
                else:
                    comparison = {
                        "object1": o1_final_text,
                        "object2": o2_final_text,
                        "SP_object1": sp_o1_final_text,
                        "SP_object2": sp_o2_final_text,
                        "SP in back": SP_final_text,
                        "TP in back": TP_final_text,
                        "rel": str(pred_rel),
                        "pol": str(pred_pol),
                        'TP_relevance': str(pred_tp_rel),
                    }
                    output[example.qas_id]={
                            "question": example.question_text,
                            "answer": example.answer_text,
                            "constructed text": predict_text,
                            "predicts": comparison
                        }
    data_path  = os.path.join(args.data_dir,args.predict_file)
    synthetic_data = make_new_file_with_synthetic_text(data_path,output)

    writer.write(json.dumps(synthetic_data, indent=4))
    writer.close()
    if total >0:
        acc ={
            "rel":relevance/total,
            "pol":polarity/total,
            "tprel":tprelevance/total,
            "relasfinal":relasfinal/total,
            "rules_step8": rules_step8/total,
        }
    else:
        acc = {}
    return f1_scores,average_scores,acc


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.metadata_dir if args.metadata_dir else args.data_dir
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset = features_and_dataset["features"], features_and_dataset["dataset"]
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = RopesProcessor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            processor = SquadV2Processor() if args.version_2_with_negative else RopesProcessor()
            if evaluate:
                examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file,grounding_type=args.grounding_type)
            else:
                examples = processor.get_train_examples(args.data_dir, filename=args.train_file,grounding_type=args.grounding_type)

        features, dataset = ropes_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset}, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--metadata_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )
    parser.add_argument(
        "--grounding_type",
        default=None,
        type=str,
        required=True,
        help="select grounding type: 1hop, 2hop",
    )
    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )

    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--rule_based", action="store_true", help="Whether not to use Rule")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("output file is ",str(os.path.exists(args.output_dir)))
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    if args.metadata_dir:
        print("I make a metadata file",str(args.metadata_dir))
        os.makedirs(args.metadata_dir)

    # Setup CUDA, GPU & distributed training
    if  args.no_cuda:
        device =torch.device("cpu")
        args.n_gpu = 0
    elif args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        # device =torch.device("cpu")
        args.n_gpu = torch.cuda.device_count()
        # args.n_gpu = 0
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    # print("totoal parameters:" + str(pytorch_total_params))

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir, force_download=True)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
        else:
            checkpoints = [args.model_name_or_path]
            if args.eval_all_checkpoints:
                checkpoints = list(os.path.dirname(c) for c in
                                   sorted(glob.glob(args.model_name_or_path + '/**/' + WEIGHTS_NAME, recursive=True)))
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

            # logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        writer = open(args.output_dir+"results.json", 'w+', encoding='utf-8')

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, force_download=True)
            model.to(args.device)

            # Evaluate
            result = evaluate_all(args, model, tokenizer, prefix=global_step)

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)
            # logger.info("Result: {}".format(result))
            # print(global_step)
            # print(result)
            logger.info("Results: {}".format(results))
            writer.write(json.dumps(result,indent=4))
        writer.close()
    # logger.info("Results: {}".format(results))


    # return results


if __name__ == "__main__":
    main()
