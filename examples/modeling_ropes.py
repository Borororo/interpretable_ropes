# -*- coding：utf-8 -*-

from transformers.modeling_bert import  BertEmbeddings, BertLayerNorm, BertModel, BertPreTrainedModel, gelu
from transformers.configuration_roberta import RobertaConfig
from transformers.modeling_roberta import ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,RobertaModel
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from allennlp.modules.matrix_attention import (
    MatrixAttention,
    BilinearMatrixAttention,
    DotProductMatrixAttention,
    LinearMatrixAttention,
)
from torch.nn import CrossEntropyLoss, MSELoss,NLLLoss
import allennlp.nn.util as allenutil

def find_surrounding_with_max(p,max_id,meanvalue,direction = None):
    '''
    :param p: attention probabilities for current sequence
    :param max_id: the id with highest probability
    :param meanvalue: threshold value, used for break loop
    :param direction: To Left, or To Right.
    :return: the margin id of max window
    '''
    current_id = max_id
    if direction =="L":
        while True:
            if current_id - 1 < 0:
                break
            elif p[current_id-1] < meanvalue:
                break
            else:
                current_id -= 1
    if direction =="R":
        while True:
            if current_id + 1 >= len(p):
                break
            elif p[current_id+1] < meanvalue:
                break
            else:
                current_id += 1
    return current_id

def calculate_logits_within_window(window_Length, ids, p_prob):
    # Not used in our paper
    scores = [np.mean(p_prob[max(0,id-window_Length):id+window_Length+1]) for id in ids]
    return ids[scores.index(max(scores))]

def find_max_ind(p_prob,no_idx = 2,window_length=1):
    # Not used in our paper
    p_prob = p_prob.detach().cpu().numpy()
    max_ids = p_prob.argsort()[-no_idx:][::-1]
    return calculate_logits_within_window(window_length,max_ids,p_prob)

def find_max_window(p_prob,mask,offset):
    batch_size = p_prob.size(0)
    out_idx = []
    for b in range(batch_size):
        mean_prob = allenutil.masked_mean(
            p_prob[b], mask[b],dim=-1
        )
        max_idx = np.argmax(p_prob[b].detach().cpu().numpy())
        # There are many possible ways to determine max_id, the above method is simply choosing the highest probability.
        # But you can use some other ideas, like calculating the total probability of a 3-gram window instead of each token.
        # max_idx = find_max_ind(p_prob[b])
        max_value = p_prob[b][max_idx]
        start = find_surrounding_with_max(p_prob[b],max_idx,max(4*mean_prob,0.0),'L')
        end = find_surrounding_with_max(p_prob[b],max_idx,max(4*mean_prob,0.0),'R')
        start +=offset[b].tolist()
        end+=offset[b].tolist()
        out_idx.append([start,end])
    return out_idx

def pad_hiddenstate(hiddenstate,max_length):
    # hidden state : l,D, we want to padded to L,D
    current_len = hiddenstate.size(-2)
    if current_len <=max_length:
        padded_hiddenstate = F.pad(hiddenstate,(0,0,0,max_length-current_len))
        padded_mask = torch.cat((torch.tensor([1]*current_len,dtype=torch.long),torch.tensor([0]*(max_length-current_len),dtype=torch.long)))
    else:
        padded_hiddenstate = hiddenstate[:max_length]
        padded_mask =torch.tensor([1]*max_length,dtype=torch.long)
    return hiddenstate, padded_hiddenstate, padded_mask

def pad_supervison_label(label,max_length):
    # hidden state : l,D, we want to padded to L,D
    current_len = label.size(-1)
    if current_len <=max_length:
        padded_label = F.pad(label,(0,max_length-current_len))
    else:
        padded_label = label[:max_length]

    return label,padded_label


class Relevance(nn.Module):
    '''
    This part is Comparison module explained in the paper.
    '''
    def __init__(self,config):
        super(Relevance,self).__init__()
        self.logits_SPo1 = BilinearMatrixAttention(
            matrix_1_dim=config.hidden_size, matrix_2_dim=config.hidden_size
        )
        self.logits_SPo2 = BilinearMatrixAttention(
            matrix_1_dim=config.hidden_size, matrix_2_dim=config.hidden_size
        )
    def forward(self,HB,HS_O1,HS_O2):
        '''
        :param HB:  Background representation
        :param HS_O1:  Object1(World1) weighted situation representation
        :param HS_O2:  Object2(World2) weighted situation representation
        :return: return relevance(comparing) logits
        '''
        logits_o1 = self.logits_SPo1(HB, HS_O1)
        logits_o2 = self.logits_SPo1(HB, HS_O2)
        p_rel_logits = torch.cat((logits_o1,logits_o2),dim=-1).squeeze(1)
        return p_rel_logits

# class Polarity(nn.Module):
#     def __init__(self,config, nhead=8, num_layers=6,dim_feedforward=2048,
#                  dropout=0.1,type = "transformer"):
#         super(Polarity,self).__init__()
#         if type == "transformer":
#             encoder_layer = nn.TransformerEncoderLayer(d_model=config.hidden_size*2,
#                                                        nhead=nhead,
#                                                        dim_feedforward=dim_feedforward,
#                                                        dropout=dropout,
#                                                        )
#             self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         # self.prob_predict = nn.Linear(config.hidden_size*2,2)
#
#     def forward(self,src,src_key_padding_mask=None):
#         device = src.device
#         mask = torch.ones(src.size(0),src.size(0))
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)
#         out = self.encoder(src = src,mask=mask,src_key_padding_mask= src_key_padding_mask.bool())
#         return out.transpose(0,1)

class MLP(nn.Module):
    '''
    A simple MLP network, used in World detection, relation classification
    '''
    def __init__(self,hidden_size,output_label):
        super(MLP, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256,output_label),
            # nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        x = self.module(x)
        return x

def compute_loss(predict,target,type):
    '''
    as we explained in the paper, we have two categories of loss, one is about span attention, another is binary prediction.
    '''
    normalizer = 0
    binary_label_probs = {"TP_relevance"}
    # binary_label_logits = {"relevance", "polarity"}
    attention_label = {"find_object1","find_object2","find_TP","find_SP","find_SP_object1","find_SP_object2"}
    device = predict.device
    if type in attention_label:
        log_attention = torch.log(predict + 1e-40)
        summed_attention_per_ex = torch.sum(log_attention * target, dim=-1)
        l = - torch.mean(summed_attention_per_ex)
        normalizer +=1
    if type in binary_label_probs:
        loss_fct = NLLLoss()
        log_prob = torch.log(predict)
        try:
            l = loss_fct(log_prob,target)
        except:
            l = loss_fct(predict.unsqueeze(0), target)
    if type =="relevance":
        loss_fct = CrossEntropyLoss()
        try:
            l = loss_fct(predict,target)
        except:
            l = loss_fct(predict.unsqueeze(0), target)
    if type =="polarity":
        negvspos = torch.tensor([3.0, 1.0], dtype=torch.float).to(device)
        loss_fct = CrossEntropyLoss(weight=negvspos)
        try:
            l = loss_fct(predict,target)
        except:
            l = loss_fct(predict.unsqueeze(0), target)
    return l


class RobertaForRopes(BertPreTrainedModel):
    '''
    The model structure for interpretation reasoning component
    Note: There may exists some INconsistency between names appeared in the paper and this code:/.

    1. Find_object = World Detection
    2. Find TP =  target/effect Detection
    3. rel_xxx_xxx = relevance between xxx and xxx = Comparison Module
    3. pol_xx_xx = polarity between xx and xx  = Relation Classification
    ... ...

    '''
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self,config):
        super(RobertaForRopes,self).__init__(config)
        self.roberta = RobertaModel(config)
        self.find_object1 = MLP(config.hidden_size,1)
        self.find_object2 = MLP(config.hidden_size,1)
        self.find_TP = MLP(config.hidden_size,1)
        self.bb_matrix_attention = LinearMatrixAttention(
            tensor_1_dim=config.hidden_size, tensor_2_dim=config.hidden_size, combination="x,y,x*y"
        )
        self.bs_bilinear_imilairty = BilinearMatrixAttention(
            matrix_1_dim=config.hidden_size, matrix_2_dim=config.hidden_size
        )
        self.ss_matrix_attention = LinearMatrixAttention(
            tensor_1_dim=config.hidden_size, tensor_2_dim=config.hidden_size, combination="x,y,x*y"
        )
        self.rel_SPo1_SPo2 = Relevance(config)
        self.pol_TP_SP = MLP(config.hidden_size*2,2)
        # self.rel_TPo1_TPo2 = None
        # self.answer_according_to_question = None
        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            len_q = None,
            bs_seperator_index = None,
            s_first = True,
            max_q_length = 30,
            max_s_length = 200,
            max_b_length = 400,
            sp_relevance = None,
            sp_tp_polarity =None,
            tp_relevance = None,
            object1_label=None,
            object2_label=None,
            SP_Object1_label=None,
            SP_Object2_label=None,
            SP_Back_label=None,
            TP_Back_label=None,
        ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        torch.set_printoptions(precision=8, sci_mode=False)
        # Step1 Encoder, we need to split Background , situation and question from the whole contextual representation, then pad them into a fixed length.
        sequence_output = outputs[0]
        batch_size = sequence_output.size(0)
        max_seq_length = sequence_output.size(1)
        hidden_size = sequence_output.size(2)
        device = outputs[0].device
        padded_HQ = torch.zeros([batch_size,max_q_length,hidden_size]).to(device)
        padded_mask_HQ = torch.zeros([batch_size,max_q_length]).to(device)
        padded_HS = torch.zeros([batch_size, max_s_length, hidden_size]).to(device)
        padded_mask_HS = torch.zeros([batch_size, max_s_length]).to(device)
        padded_HB = torch.zeros([batch_size, max_b_length, hidden_size]).to(device)
        padded_mask_HB = torch.zeros([batch_size, max_b_length]).to(device)
        s_inds = []
        for ind in range(batch_size):
            try:
                mask_ind = (attention_mask[ind] == 0).nonzero()[0][0]-1
            except:
                mask_ind = max_seq_length-1
            HQ,padded_h_q,padded_mask_q= pad_hiddenstate(sequence_output[ind,1:1+len_q[ind],:],max_q_length)
            if s_first:
                HS,padded_h_s, padded_mask_s = pad_hiddenstate(sequence_output[ind,len_q[ind]+3:bs_seperator_index[ind]+1, :], max_s_length)
                HB,padded_h_b, padded_mask_b = pad_hiddenstate(sequence_output[ind, bs_seperator_index[ind]+1:mask_ind, :], max_b_length)
            else:
                HB,padded_h_b, padded_mask_b = pad_hiddenstate(
                    sequence_output[ind, len_q[ind] + 3:bs_seperator_index[ind] + 1, :], max_s_length)
                HS,padded_h_s, padded_mask_s = pad_hiddenstate(
                    sequence_output[ind, bs_seperator_index[ind] + 1:mask_ind, :], max_b_length)
            s_inds.append(bs_seperator_index[ind]-2-len_q[ind])
            padded_HQ[ind,:,:] = padded_h_q
            padded_mask_HQ[ind,:] = padded_mask_q
            padded_HS[ind, :, :] = padded_h_s
            padded_mask_HS[ind, :] = padded_mask_s
            padded_HB[ind, :, :] = padded_h_b
            padded_mask_HB[ind, :] = padded_mask_b

        # auxiliary labels also need padding.
        padded_O1 = torch.zeros([batch_size, max_s_length]).to(device)
        padded_O2 = torch.zeros([batch_size, max_s_length]).to(device)
        padded_SP_o1 = torch.zeros([batch_size, max_s_length]).to(device)
        padded_SP_o2 = torch.zeros([batch_size, max_s_length]).to(device)
        padded_SP = torch.zeros([batch_size, max_b_length]).to(device)
        padded_TP = torch.zeros([batch_size, max_b_length]).to(device)
        for ind in range(batch_size):
            try:
                mask_ind = (attention_mask[ind] == 0).nonzero()[0][0]
            except:
                mask_ind = max_seq_length - 1
            if s_first:
                _, padded_o1 = pad_supervison_label(object1_label[ind, len_q[ind] + 3:bs_seperator_index[ind] + 1],
                                                    max_s_length) if object1_label is not None else [None, None]
                _, padded_o2 = pad_supervison_label(object2_label[ind, len_q[ind] + 3:bs_seperator_index[ind] + 1],
                                                    max_s_length) if object2_label is not None else [None, None]
                _, padded_sp_o1 = pad_supervison_label(
                    SP_Object1_label[ind, len_q[ind] + 3:bs_seperator_index[ind] + 1],
                    max_s_length) if SP_Object1_label is not None else [None, None]
                _, padded_sp_o2 = pad_supervison_label(
                    SP_Object2_label[ind, len_q[ind] + 3:bs_seperator_index[ind] + 1],
                    max_s_length) if SP_Object2_label is not None else [None, None]
                _, padded_sp = pad_supervison_label(SP_Back_label[ind, bs_seperator_index[ind] + 1:mask_ind],
                                                    max_b_length) if SP_Back_label is not None else [None, None]
                _, padded_tp = pad_supervison_label(TP_Back_label[ind, bs_seperator_index[ind] + 1:mask_ind],
                                                    max_b_length) if TP_Back_label is not None else [None, None]

            padded_O1[ind, :] = padded_o1 if padded_o1 is not None else padded_O1[ind, :]
            padded_O2[ind, :] = padded_o2 if padded_o2 is not None else padded_O2[ind, :]
            padded_SP_o1[ind, :] = padded_sp_o1 if padded_sp_o1 is not None else padded_SP_o1[ind, :]
            padded_SP_o2[ind, :] = padded_sp_o2 if padded_sp_o2 is not None else padded_SP_o2[ind, :]
            padded_SP[ind, :] = padded_sp if padded_sp is not None else padded_SP[ind, :]
            padded_TP[ind, :] = padded_tp if padded_tp is not None else padded_TP[ind, :]

        # **************************** STEP 2 Find OBJECT/World ****************************
        # [b,n,d] -> [b,n,1]
        ps_object1 = self.find_object1(padded_HS)
        ps_object2 = self.find_object2(padded_HS)
        ps_object1 = allenutil.masked_softmax(
            ps_object1.squeeze(),padded_mask_HS,memory_efficient=True
        )
        ps_object2 = allenutil.masked_softmax(
            ps_object2.squeeze(), padded_mask_HS, memory_efficient=True
        )



        #  ****************************STEP 3 Find TP/Effect in B ****************************
        # [b,m,d] -> [b,m,1]
        pb_TP = self.find_TP(padded_HB)
        pb_TP = allenutil.masked_softmax(
            pb_TP.squeeze(), padded_mask_HB, memory_efficient=True
        )

        #  ****************************STEP 4 Relocate TP/Effect to SP/Cause ****************************
        mean_HS = allenutil.masked_mean(
            padded_HS, padded_mask_HS.unsqueeze(-1),dim=1
        )
        relocate_bb_similarity_matrix = self.bb_matrix_attention(torch.add(mean_HS.unsqueeze(1),padded_HB),padded_HB)
        b2b_attention_matrix = allenutil.masked_softmax(
            relocate_bb_similarity_matrix,padded_mask_HB,memory_efficient=True,dim=-1
        )
        pb_SP = torch.sum(torch.mul(pb_TP.unsqueeze(-1),b2b_attention_matrix),dim=1)

        # if we dont have labels, we can comment out this two lines
        padded_TP_normal = torch.nn.functional.normalize(padded_TP,p=1,dim=-1)
        pb_SP_gold = torch.sum(torch.mul(padded_TP_normal.unsqueeze(-1),b2b_attention_matrix),dim=1)


        # *************************Step 5: Find SP/cause for object/world 1 and object/world 2 ****************************
        # Explained in Comparison module, treat two worlds as masks.

        s2b_similarity_matrix = self.bs_bilinear_imilairty(
            padded_HS,padded_HB
        )
        s2b_similarity_attention = allenutil.masked_softmax(
            s2b_similarity_matrix,padded_mask_HB,memory_efficient=True,dim=1
        )
        b2s_similarity_matrix = torch.transpose(s2b_similarity_matrix, 1, 2)
        ps_guided_SP = torch.sum(torch.mul(pb_SP.unsqueeze(-1), b2s_similarity_matrix), dim=1)
        mask_score_object1 = ps_object1
        mask_score_object2 = ps_object2
        ps_SP_object1 = torch.mul(mask_score_object1, ps_guided_SP)
        ps_SP_object1 = allenutil.masked_softmax(
            ps_SP_object1, padded_mask_HS, memory_efficient=True, dim=-1
        )
        ps_SP_object2 = torch.mul(mask_score_object2, ps_guided_SP)
        ps_SP_object2 = allenutil.masked_softmax(
            ps_SP_object2, padded_mask_HS, memory_efficient=True, dim=-1
        )


        #  gold label if we have.
        padded_SP_normal = torch.nn.functional.normalize(padded_SP, p=1, dim=-1)
        ps_guided_SP_gold = torch.sum(torch.mul(padded_SP_normal.unsqueeze(-1), b2s_similarity_matrix), dim=1)
        mask_score_object1_gold = torch.nn.functional.normalize(padded_O1+0.01, p=1, dim=-1)
        mask_score_object2_gold = torch.nn.functional.normalize(padded_O2+0.01, p=1, dim=-1)
        ps_SP_object1_gold = torch.mul(mask_score_object1_gold, ps_guided_SP_gold)
        ps_SP_object1_gold = allenutil.masked_softmax(
            ps_SP_object1_gold, padded_mask_HS, memory_efficient=True, dim=-1
        )
        ps_SP_object2_gold = torch.mul(mask_score_object2_gold, ps_guided_SP_gold)
        ps_SP_object2_gold = allenutil.masked_softmax(
            ps_SP_object2_gold, padded_mask_HS, memory_efficient=True, dim=-1
        )


        #  ****************************Step 6 relevance/comparison check ****************************
        summed_HB_weighted_pb_SP = torch.matmul(pb_SP.unsqueeze(1),padded_HB)  # 1XMXMXD => [B,1,D]
        summed_HS_weighted_ps_SP_o1= torch.matmul(ps_SP_object1.unsqueeze(1),padded_HS) # 1XNXNXD => [B,1,D]
        summed_HS_weighted_ps_SP_o2 = torch.matmul(ps_SP_object2.unsqueeze(1), padded_HS)  # 1XNXNXD => [B,1,D]
        p_relevance_logits = self.rel_SPo1_SPo2(summed_HB_weighted_pb_SP,summed_HS_weighted_ps_SP_o1,summed_HS_weighted_ps_SP_o2)
        normal_p_relevance_logits = torch.nn.functional.normalize(p_relevance_logits,p=1)
        p_relevance = torch.softmax(normal_p_relevance_logits,dim=-1)

        #  gold label if we have.
        padded_SP_o1_normal = torch.nn.functional.normalize(padded_SP_o1,p=1,dim=-1)
        padded_SP_o2_normal = torch.nn.functional.normalize(padded_SP_o2,p=1,dim=-1)
        GOLD_summed_HB_weighted_pb_SP = torch.matmul(padded_SP_normal.unsqueeze(1).type(dtype=torch.float),padded_HB)  # 1XMXMXD => [B,1,D]
        GOLD_summed_HS_weighted_ps_SP_o1= torch.matmul(padded_SP_o1_normal.unsqueeze(1).type(dtype=torch.float),padded_HS) # 1XNXNXD => [B,1,D]
        GOLD_summed_HS_weighted_ps_SP_o2 = torch.matmul(padded_SP_o2_normal.unsqueeze(1).type(dtype=torch.float), padded_HS)  # 1XNXNXD => [B,1,D]
        p_relevance_logits_gold = self.rel_SPo1_SPo2(GOLD_summed_HB_weighted_pb_SP, GOLD_summed_HS_weighted_ps_SP_o1,
                                                GOLD_summed_HS_weighted_ps_SP_o2)
        normal_p_relevance_logits_gold = torch.nn.functional.normalize(p_relevance_logits_gold,p=1)
        p_relevance_gold = torch.softmax(normal_p_relevance_logits_gold,dim=-1)

        #  ****************************Step 7 relation classification/polarity ****************************
        summed_HB_weighted_pb_TP = torch.matmul(pb_TP.unsqueeze(1),padded_HB)
        summed_HB_weighted_TP_SP = torch.cat((summed_HB_weighted_pb_SP,summed_HB_weighted_pb_TP),dim=-1).squeeze(1)
        p_polarity_logits = self.pol_TP_SP(summed_HB_weighted_TP_SP)
        p_polarity = torch.softmax(p_polarity_logits,dim=-1)
        p_polarity_negative = p_polarity[:,0]
        p_polarity_positive = p_polarity[:,1]

        #  gold label if we have.
        summed_HB_weighted_pb_TP_gold = torch.matmul(padded_TP_normal.unsqueeze(1), padded_HB)
        summed_HB_weighted_TP_SP_gold = torch.cat((GOLD_summed_HB_weighted_pb_SP, summed_HB_weighted_pb_TP_gold), dim=-1).squeeze(1)
        p_polarity_logits_gold = self.pol_TP_SP(summed_HB_weighted_TP_SP_gold)
        p_polarity_gold = torch.softmax(p_polarity_logits_gold, dim=-1)
        p_polarity_negative_gold = p_polarity_gold[:, 0]
        p_polarity_positive_gold = p_polarity_gold[:, 1]

        # ****************************Step 8 Reasoning ****************************
        object1 = p_relevance[:,0]
        object2 = p_relevance[:,1]
        p_TP_object1 = p_polarity_positive*object1+p_polarity_negative*object2
        p_TP_object2 = p_polarity_negative*object1+p_polarity_positive*object2
        p_TP_objects = torch.stack((p_TP_object1,p_TP_object2),dim=1).squeeze()

        #  gold label if we have.
        object1_gold = p_relevance_gold[:,0]
        object2_gold = p_relevance_gold[:,1]
        p_TP_object1_gold = p_polarity_positive_gold*object1_gold+p_polarity_negative_gold*object2_gold
        p_TP_object2_gold = p_polarity_negative_gold*object1_gold+p_polarity_positive_gold*object2_gold
        p_TP_objects_gold = torch.stack((p_TP_object1_gold,p_TP_object2_gold),dim=1).squeeze()


        try:
            assert torch.sum(padded_O1) != 0
            assert torch.sum(padded_O2) != 0
            assert torch.sum(padded_SP_o1) != 0
            assert torch.sum(padded_SP_o2) != 0
            assert torch.sum(padded_SP) != 0
            assert torch.sum(padded_TP) != 0
        except:
            pass
        loss_o1 = compute_loss(ps_object1,padded_O1,"find_object1") if object1_label is not None else 0.0
        loss_o2 = compute_loss(ps_object2,padded_O2,"find_object2") if object2_label is not None else 0.0
        loss_TP = compute_loss(pb_TP,padded_TP,"find_TP") if TP_Back_label is not None else 0.0
        loss_SP = compute_loss(pb_SP,padded_SP,"find_SP") if SP_Back_label is not None else 0.0
        loss_SP_o1 = compute_loss(ps_SP_object1,padded_SP_o1,"find_SP_object1") if SP_Object1_label is not None else 0.0
        loss_SP_o2 = compute_loss(ps_SP_object2, padded_SP_o2, "find_SP_object2") if SP_Object2_label is not None else 0.0
        loss_rel = compute_loss(normal_p_relevance_logits,sp_relevance,"relevance") if sp_relevance is not None else 0.0
        loss_pol = compute_loss(p_polarity_logits,sp_tp_polarity,"polarity") if sp_tp_polarity is not None else 0.0
        loss_on_TP = compute_loss(p_TP_objects, tp_relevance,"TP_relevance") if tp_relevance is not None else 0.0

        loss2_SP = compute_loss(pb_SP_gold,padded_SP,"find_SP") if SP_Back_label is not None else 0.0
        loss2_SP_o1 = compute_loss(ps_SP_object1_gold,padded_SP_o1,"find_SP_object1") if SP_Object1_label is not None else 0.0
        loss2_SP_o2 = compute_loss(ps_SP_object2_gold, padded_SP_o2, "find_SP_object2") if SP_Object2_label is not None else 0.0
        loss2_rel = compute_loss(normal_p_relevance_logits_gold, sp_relevance,
                                "relevance") if sp_relevance is not None else 0.0
        loss2_pol = compute_loss(p_polarity_logits_gold, sp_tp_polarity,
                                "polarity") if sp_tp_polarity is not None else 0.0
        loss2_on_TP = compute_loss(p_TP_objects, tp_relevance, "TP_relevance") if tp_relevance is not None else 0.0
        out = {
            "object1":(loss_o1).tolist(),
            "object2":(loss_o2).tolist(),
            "TP":(loss_TP).tolist(),
            "SP":(loss_SP).tolist(),
            "loss_SP_o1":(loss_SP_o1).tolist(),
            "loss_SP_o2":(loss_SP_o2).tolist(),
            "loss_rel":(loss_rel).tolist(),
            "loss_pol":(loss_pol).tolist(),
            "loss_on_TP":(loss_on_TP).tolist(),
        }

        # Loss function, play around it.
        loss = 0.05 * loss_o1 + 0.05 * loss_o2 + 0.05 * loss_SP + 0.05 * loss_TP + 0.05 * loss_SP_o1 + 0.05 * loss_SP_o2 + 0.2 * loss_pol + 0.2 * loss_rel + 0.3 * loss_on_TP

        #  The following part works as: returning the necessary numbers for predicting intermediate output for each modules.
        object1_ind = find_max_window(ps_object1,padded_mask_HS,offset=len_q+3)
        object2_ind = find_max_window(ps_object2,padded_mask_HS,offset=len_q+3)
        TP_ind = find_max_window(pb_TP,padded_mask_HB,offset=bs_seperator_index+1)
        try:
            SP_object1_ind = find_max_window(ps_SP_object1,padded_mask_HS,offset=len_q+3)
            SP_object2_ind = find_max_window(ps_SP_object2,padded_mask_HS,offset=len_q+3)
            SP_ind = find_max_window(pb_SP,padded_mask_HB,offset=bs_seperator_index+1)
        except:
            SP_object1_ind = [1,1]
            SP_object2_ind = [1,1]
            SP_ind = [1,1]
        predict = {
            "p_o1":ps_object1.tolist(),
            "p_o2":ps_object2.tolist(),
            "p_TP":pb_TP.tolist(),
            "p_SP":pb_SP.tolist(),
            "p_sp_o1":ps_SP_object1.tolist(),
            "p_sp_o2":ps_SP_object2.tolist(),
            "object1":object1_ind,
            "object2":object2_ind ,
            "TP": TP_ind,
            "SP": SP_ind,
            "SP_o1": SP_object1_ind,
            "SP_o2": SP_object2_ind,
            "relevance": p_relevance,
            "polarity": p_polarity,
            "tp_relevance": p_TP_objects,
        }
        output = [loss, out, predict]
        return output

class RobertaForQuestionAnsweringMultiAnswer(BertPreTrainedModel):
    '''
    Be different from original exmaple in hugging face transformer, we consider multi-answer method.
    Since in ropes, the answer usually appears in the question many times, we should not only use last appearance as labels.
    '''
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        all_multi_start_position = None,
        all_multi_end_position = None,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if (all_multi_start_position is not None and all_multi_end_position is not None) or (start_positions is not None and end_positions is not None):
            if torch.sum(all_multi_start_position) !=0 and torch.sum(all_multi_end_position)!=0:
                if len(all_multi_start_position.size()) > 1:
                    all_start_positions = all_multi_start_position.squeeze(-1)
                if len(all_multi_end_position.size()) > 1:
                    all_end_positions = all_multi_end_position.squeeze(-1)
                all_start_probs = torch.log_softmax(start_logits,dim=-1)
                all_end_probs = torch.log_softmax(end_logits,dim=-1)
                # sum
                all_start_loss = -torch.mean(torch.sum(all_start_probs*all_multi_start_position,dim=-1))
                all_end_loss = -torch.mean(torch.sum(all_end_probs*all_multi_end_position,dim=-1))

                all_total_loss = (all_start_loss+all_end_loss)/2
                outputs = (all_total_loss,) + outputs


            else:
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)
                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
                outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)




