# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)

from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from wenet.transformer.encoder import TransformerEncoder
import logging
import pdb
from wenet.utils.emotion_metics import ConfusionMetrics
from wenet.transformer.s3prl_emotion_model import *

logging.getLogger().setLevel(logging.INFO)

class StatisticsPooling(nn.Module):
    """This class implements a statistic pooling layer.

    It returns the mean and/or std of input tensor.

    Arguments
    ---------
    return_mean : True
         If True, the average pooling will be returned.
    return_std : True
         If True, the standard deviation will be returned.

    Example
    -------
    >>> inp_tensor = torch.rand([5, 100, 50])
    >>> sp_layer = StatisticsPooling()
    >>> out_tensor = sp_layer(inp_tensor)
    >>> out_tensor.shape
    torch.Size([5, 1, 100])
    """

    def __init__(self, return_mean=True, return_std=True):
        super().__init__()

        # Small value for GaussNoise
        self.eps = 1e-5
        self.return_mean = return_mean
        self.return_std = return_std
        if not (self.return_mean or self.return_std):
            raise ValueError(
                "both of statistics are equal to False \n"
                "consider enabling mean and/or std statistic pooling"
            )

    def forward(self, x, lengths=None):
        """Calculates mean and std for a batch (input tensor).

        Arguments
        ---------
        x : torch.Tensor
            It represents a tensor for a mini-batch.
        """
        # pdb.set_trace()
        if lengths is None:
            if self.return_mean:
                mean = x.mean(dim=1)
            if self.return_std:
                std = x.std(dim=1)
        else:
            mean = []
            std = []
            for snt_id in range(x.shape[0]):
                # Avoiding padded time steps
                # actual_size = int(torch.round(lengths[snt_id] * x.shape[1]))
                actual_size = lengths[snt_id]
                # computing statistics
                if self.return_mean:
                    mean.append(
                        torch.mean(x[snt_id, 0:actual_size, ...], dim=0)
                    )
                if self.return_std:
                    std.append(torch.std(x[snt_id, 0:actual_size, ...], dim=0))
            if self.return_mean:
                mean = torch.stack(mean)
            if self.return_std:
                std = torch.stack(std)

        if self.return_mean:
            gnoise = self._get_gauss_noise(mean.size(), device=mean.device)
            gnoise = gnoise
            mean += gnoise
        if self.return_std:
            std = std + self.eps

        # Append mean and std of the batch
        if self.return_mean and self.return_std:
            pooled_stats = torch.cat((mean, std), dim=1)
            pooled_stats = pooled_stats.unsqueeze(1)
        elif self.return_mean:
            pooled_stats = mean.unsqueeze(1)
        elif self.return_std:
            pooled_stats = std.unsqueeze(1)

        return pooled_stats
    def _get_gauss_noise(self, shape_of_tensor, device="cpu"):
        """Returns a tensor of epsilon Gaussian noise.

        Arguments
        ---------
        shape_of_tensor : tensor
            It represents the size of tensor for generating Gaussian noise.
        """
        gnoise = torch.randn(shape_of_tensor, device=device)
        gnoise -= torch.min(gnoise)
        gnoise /= torch.max(gnoise)
        gnoise = self.eps * ((1 - 9) * gnoise + 9)

        return gnoise
    

class EmotionRecognitionModel(torch.nn.Module):
    def __init__(
        self,
        category_size: int,
        encoder: TransformerEncoder,
        input_ln=False,
        s3prl_flag=True,
        **kwargs
    ):
        super().__init__()
        self.category_size = category_size
        self.encoder = encoder
    
        self.pooling = StatisticsPooling(return_std=False)
        self.linear = torch.nn.Linear(self.encoder.output_size(), self.category_size)
        # self.criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        self.criterion = torch.nn.CrossEntropyLoss()

        self.input_ln = input_ln
        if self.input_ln:
            self.input_ln = torch.nn.LayerNorm(
                80, eps=1e-5, elementwise_affine=False
            )
            
        self.freeze_encoder=False
        if self.freeze_encoder:
            # for param in self.encoder.encoders[:-1].parameters():
            #     param.requires_grad = False
            for param in self.encoder.encoders.parameters():
                param.requires_grad = False

        self.test_metrics = ConfusionMetrics(self.category_size)
        
        # for s3prl
        self.s3prl_flag = s3prl_flag
        if self.s3prl_flag:
            self.modelrc = {
                    'projector_dim': 256,
                    'select': 'UtteranceLevel',
                    'UtteranceLevel': {
                        'pooling': 'MeanPooling'
                    },
                    'DeepModel': {
                        'model_type': 'CNNSelfAttention',
                        'hidden_dim': 80,
                        'kernel_size': 5,
                        'padding': 2,
                        'pooling': 5,
                        'dropout': 0.4
                    }
                 }
            model_cls = eval(self.modelrc['select'])
            model_conf = self.modelrc.get(self.modelrc['select'], {})
            self.projector = nn.Linear(self.encoder.output_size(), self.modelrc['projector_dim'])
            self.s3prl_model = model_cls(
                input_dim = self.modelrc['projector_dim'],
                output_dim = self.category_size,
                **model_conf,
            )
            # for DeepModel
            # model_conf = {
            #             'model_type': 'CNNSelfAttention',
            #             'hidden_dim': 80,
            #             'kernel_size': 5,
            #             'padding': 2,
            #             'pooling': 5,
            #             'dropout': 0.4
            #         }
            # self.projector = nn.Linear(self.encoder.output_size(), self.modelrc['projector_dim'])
            # self.s3prl_model = DeepModel(
            #     input_dim = self.modelrc['projector_dim'],
            #     output_dim = self.category_size,
            #     **model_conf,
            # )
    def metrics_reset(self):
        # metrics
        self.test_metrics.clear()

    def metrics_fit(self,output,label):
         for l, p in zip(label, output):
            self.test_metrics.fit(int(l.argmax()), int(p.argmax()))
    
    def get_acc(self):
        return self.test_metrics.precision
    

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        label: torch.Tensor,
        label_lengh: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            label: (Batch, Length)
            label_lengh: no use
        """
        #logging.info(label)
        #pdb.set_trace()
        
        if self.input_ln:
            assert self.encoder.global_cmvn is None
            speech = self.input_ln(speech)
            #print("laynorm")
            
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        if self.s3prl_flag:
            features_len = encoder_mask.squeeze(1).sum(1).to(device=encoder_out[0].device)
            features = self.projector(encoder_out)
            output, _ = self.s3prl_model(features, features_len)
        else:
            # global avg pooling
            encoder_out = self.pooling(encoder_out,encoder_mask.squeeze(1).sum(1))
            encoder_out = encoder_out.squeeze(1)
            #print("encoder_out:",encoder_out.size())
            output = self.linear(encoder_out)
        label = label.type(torch.FloatTensor).to(output.device)
        self.metrics_fit(output,label)
        loss = self.criterion(output,label)
        #loss = (loss-b).abs()+b
        return {"loss": loss}

    def inference(self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        label=None,):
        if self.input_ln:
            assert self.encoder.global_cmvn is None
            speech = self.input_ln(speech)
        #print("speech:",speech.size())
        #print("speech_lengths:",speech_lengths)
        #print("label:",label.size())
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out = self.linear1(encoder_out)
        #global avg pooling !!!!!!!!!!!!!!!!!!!!!!!!!!
        encoder_out = torch.mean(encoder_out,dim=1)
        #print("encoder_out1:",encoder_out.size())
        output = self.linear2(encoder_out)
        #print("encoder_out2:",output.size())
        softmax_output = torch.nn.functional.softmax(output,dim=-1)
        #print("softmax_output:",softmax_output.size())
        confidence,output= torch.max(softmax_output,dim = -1)
        return output, confidence
    