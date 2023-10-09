import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask=None):
        """
            N: batch size, T: sequence length, H: Hidden dimension
            input:
                batch_rep : size (N, T, H)
            attention_weight:
                att_w : size (N, T, 1)
            return:
                utter_rep: size (N, H)
        """
        att_logits = self.W(batch_rep).squeeze(-1)
        if att_mask is not None:
            att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep


class CNNSelfAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        padding,
        pooling,
        dropout,
        output_class_num,
        **kwargs
    ):
        super(CNNSelfAttention, self).__init__()
        self.model_seq = nn.Sequential(
            nn.AvgPool1d(kernel_size, pooling, padding),
            nn.Dropout(p=dropout),
            nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
        )
        self.pooling = SelfAttentionPooling(hidden_dim)
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_class_num),
        )

    def forward(self, features, att_mask):
        features = features.transpose(1, 2)
        features = self.model_seq(features)
        out = features.transpose(1, 2)
        out = self.pooling(out, att_mask).squeeze(-1)
        predicted = self.out_layer(out)
        return predicted


class FCN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        padding,
        pooling,
        dropout,
        output_class_num,
        **kwargs,
    ):
        super(FCN, self).__init__()
        self.model_seq = nn.Sequential(
            nn.Conv1d(input_dim, 96, 11, stride=4, padding=5),
            nn.LocalResponseNorm(96),
            nn.ReLU(),
            nn.MaxPool1d(3, 2),
            nn.Dropout(p=dropout),
            nn.Conv1d(96, 256, 5, padding=2),
            nn.LocalResponseNorm(256),
            nn.ReLU(),
            nn.MaxPool1d(3, 2),
            nn.Dropout(p=dropout),
            nn.Conv1d(256, 384, 3, padding=1),
            nn.LocalResponseNorm(384),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(384, 384, 3, padding=1),
            nn.LocalResponseNorm(384),
            nn.ReLU(),
            nn.Conv1d(384, 256, 3, padding=1),
            nn.LocalResponseNorm(256),
            nn.MaxPool1d(3, 2),
        )
        self.pooling = SelfAttentionPooling(256)
        self.out_layer = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_class_num),
        )

    def forward(self, features, att_mask):
        features = features.transpose(1, 2)
        features = self.model_seq(features)
        out = features.transpose(1, 2)
        out = self.pooling(out).squeeze(-1)
        predicted = self.out_layer(out)
        return predicted


class DeepNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        padding,
        pooling,
        dropout,
        output_class_num,
        **kwargs,
    ):
        super(DeepNet, self).__init__()
        self.model_seq = nn.Sequential(
            nn.Conv1d(input_dim, 10, 9),
            nn.ReLU(),
            nn.Conv1d(10, 10, 5),
            nn.ReLU(),
            nn.Conv1d(10, 10, 3),
            nn.MaxPool1d(3, 1),
            nn.BatchNorm1d(10, affine=False),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(10, 40, 3),
            nn.ReLU(),
            nn.Conv1d(40, 40, 3),
            nn.MaxPool1d(2, 1),
            nn.BatchNorm1d(40, affine=False),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(40, 80, 10),
            nn.ReLU(),
            nn.Conv1d(80, 80, 1),
            nn.MaxPool1d(2, 1),
            nn.BatchNorm1d(80, affine=False),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(80, 80, 1),
        )
        self.pooling = SelfAttentionPooling(80)
        self.out_layer = nn.Sequential(
            nn.Linear(80, 30),
            nn.ReLU(),
            nn.Linear(30, output_class_num),
        )

    def forward(self, features, att_mask):
        features = features.transpose(1, 2)
        features = self.model_seq(features)
        out = features.transpose(1, 2)
        out = self.pooling(out).squeeze(-1)
        predicted = self.out_layer(out)
        return predicted


class DeepModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        model_type,
        pooling,
        **kwargs
    ):
        super(DeepModel, self).__init__()
        self.pooling = pooling
        self.model = eval(model_type)(input_dim=input_dim, output_class_num=output_dim, pooling=pooling, **kwargs)

    def forward(self, features, features_len):
        attention_mask = [
            torch.ones(math.ceil((l / self.pooling)))
            for l in features_len
        ]
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        attention_mask = (1.0 - attention_mask) * -100000.0
        attention_mask = attention_mask.to(features.device)
        predicted = self.model(features, attention_mask)
        return predicted, None

# ..model
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_downstream_model(input_dim, output_dim, config):
    model_cls = eval(config['select'])
    model_conf = config.get(config['select'], {})
    model = model_cls(input_dim, output_dim, **model_conf)
    return model


class FrameLevel(nn.Module):
    def __init__(self, input_dim, output_dim, hiddens=None, activation='ReLU', **kwargs):
        super().__init__()
        latest_dim = input_dim
        self.hiddens = []
        if hiddens is not None:
            for dim in hiddens:
                self.hiddens += [
                    nn.Linear(latest_dim, dim),
                    getattr(nn, activation)(),
                ]
                latest_dim = dim
        self.hiddens = nn.Sequential(*self.hiddens)
        self.linear = nn.Linear(latest_dim, output_dim)

    def forward(self, hidden_state, features_len=None):
        hidden_state = self.hiddens(hidden_state)
        logit = self.linear(hidden_state)

        return logit, features_len


class UtteranceLevel(nn.Module):
    def __init__(self,
        input_dim,
        output_dim,
        pooling='MeanPooling',
        activation='ReLU',
        pre_net=None,
        post_net={'select': 'FrameLevel'},
        **kwargs
    ):
        super().__init__()
        latest_dim = input_dim
        self.pre_net = get_downstream_model(latest_dim, latest_dim, pre_net) if isinstance(pre_net, dict) else None
        self.pooling = eval(pooling)(input_dim=latest_dim, activation=activation)
        self.post_net = get_downstream_model(latest_dim, output_dim, post_net)

    def forward(self, hidden_state, features_len=None):
        if self.pre_net is not None:
            hidden_state, features_len = self.pre_net(hidden_state, features_len)

        pooled, features_len = self.pooling(hidden_state, features_len)
        logit, features_len = self.post_net(pooled, features_len)

        return logit, features_len


class MeanPooling(nn.Module):

    def __init__(self, **kwargs):
        super(MeanPooling, self).__init__()

    def forward(self, feature_BxTxH, features_len, **kwargs):
        ''' 
        Arguments
            feature_BxTxH - [BxTxH]   Acoustic feature with shape 
            features_len  - [B] of feature length
        '''
        agg_vec_list = []
        for i in range(len(feature_BxTxH)):
            agg_vec = torch.mean(feature_BxTxH[i][:features_len[i]], dim=0)
            agg_vec_list.append(agg_vec)

        return torch.stack(agg_vec_list), torch.ones(len(feature_BxTxH)).long()


class AttentivePooling(nn.Module):
    ''' Attentive Pooling module incoporate attention mask'''

    def __init__(self, input_dim, activation, **kwargs):
        super(AttentivePooling, self).__init__()
        self.sap_layer = AttentivePoolingModule(input_dim, activation)

    def forward(self, feature_BxTxH, features_len):
        ''' 
        Arguments
            feature_BxTxH - [BxTxH]   Acoustic feature with shape 
            features_len  - [B] of feature length
        '''
        device = feature_BxTxH.device
        len_masks = torch.lt(torch.arange(features_len.max()).unsqueeze(0).to(device), features_len.unsqueeze(1))
        sap_vec, _ = self.sap_layer(feature_BxTxH, len_masks)

        return sap_vec, torch.ones(len(feature_BxTxH)).long()


class AttentivePoolingModule(nn.Module):
    """
    Implementation of Attentive Pooling 
    """
    def __init__(self, input_dim, activation='ReLU', **kwargs):
        super(AttentivePoolingModule, self).__init__()
        self.W_a = nn.Linear(input_dim, input_dim)
        self.W = nn.Linear(input_dim, 1)
        self.act_fn = getattr(nn, activation)()
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask):
        """
        input:
        batch_rep : size (B, T, H), B: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
        att_w : size (B, T, 1)
        
        return:
        utter_rep: size (B, H)
        """
        att_logits = self.W(self.act_fn(self.W_a(batch_rep))).squeeze(-1)
        att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep, att_w