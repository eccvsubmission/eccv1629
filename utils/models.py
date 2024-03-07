import torch
from torch import nn, Tensor
from typing import Dict, List, Optional
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from .fusion import *
from .contrastive_loss import clip_loss
from functools import partial
import torch
import numpy as np
import copy 
from tsai.all import TST


class OneStreamVMAE(nn.Module):
    def __init__(self,
                 label2id,
                 id2label,
                 model_ckpt =  "MCG-NJU/videomae-base-finetuned-kinetics",
                 stream_name = "video"):
        super().__init__()
        self.vmae = VideoMAEForVideoClassification.from_pretrained(
                            model_ckpt, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True,)
        self.stream = stream_name
    def forward(self, x):
        x = self.vmae(x[self.stream].permute(0,2,1,3,4)).logits
        return x
    
def freeze_params(model):
    # Freeze all parameters of the model
    for param in model.parameters():
        param.requires_grad = False

    # Print to verify that all parameters are frozen
    for name, param in model.named_parameters():
        if "attention" in name:
            param.requires_grad = True
        if "classifier" in name:
            param.requires_grad =True
        if "attn" in name:
            param.requires_grad = True
        if "head" in name:
            param.requires_grad =True
            
class OneStreamMVIT(torch.nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.model = self.load_mvit()
        self.classifier = torch.nn.Linear(768,num_classes)
        
    def load_mvit(self,):
        from torchvision.models.video import mvit_v1_b, MViT_V1_B_Weights
        import torch
        weights = MViT_V1_B_Weights.KINETICS400_V1
        model = mvit_v1_b(weights=weights)
        model.head = torch.nn.Identity()
        return model
    
    def forward(self, x):
        x = self.model(x["video"])
        x = self.classifier(x)
        return x
    
class OneStreamSWIN(torch.nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.model = self.load_mvit()
        self.classifier = torch.nn.Linear(1024,num_classes)
        
    def load_mvit(self,):
        from torchvision.models.video import swin3d_b, Swin3D_B_Weights 
        import torch
        weights = Swin3D_B_Weights.KINETICS400_V1
        model = swin3d_b(weights=weights)
        model.head = torch.nn.Identity()
        return model
    
    def forward(self, x):
        x = self.model(x["video"])
        x = self.classifier(x)
        return x
    
    
class MultiModalVMAE(nn.Module):
    def __init__(self,
                 model_ckpt="MCG-NJU/videomae-base-finetuned-kinetics",
                 feature_dim = 768,
                 rank=30,
                 fusion_type="af",
                 num_classes:int=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.rank = rank
        self.num_classes = num_classes
        self.fusion_type = fusion_type
        self.incabin_model = VideoMAEForVideoClassification.from_pretrained(model_ckpt,
                                                                      ignore_mismatched_sizes=True).to("cuda:1")
        self.incabin_model.classifier = torch.nn.Identity()
        self.exterior_model = VideoMAEForVideoClassification.from_pretrained(model_ckpt,
                                                                      ignore_mismatched_sizes=True).to("cuda:0")
        self.exterior_model.classifier = torch.nn.Identity()
        self.fusion_layer = self.fusion_layer_constructor(fusion_type=fusion_type).to("cuda:0")
        self.fc = torch.nn.Linear(self.feature_dim, self.num_classes).to("cuda:0")
        
    def fusion_layer_constructor(self, fusion_type:str):
        
        if fusion_type == "lrtf": #low rank tensor fusion 
            return LowRankTensorFusion(input_dims=[self.feature_dim, 
                                            self.feature_dim], 
                                output_dim=self.feature_dim, 
                                rank=self.rank)
        if fusion_type == "af":
            return AttentionFusionModule({"video":self.feature_dim,
                                   "incabin_video":self.feature_dim,},
                                         self.feature_dim,)
        if fusion_type == "add":
            return Add()
        
        if fusion_type == "mean":
            return Add(avg=True)
        
        if fusion_type == "concat_w_linear":
            return ConcatWithLinear(self.feature_dim*2,self.feature_dim)
        
        if fusion_type == "deepset":
            return DeepsetFusionModule({"video":self.feature_dim,
                        "incabin_video":self.feature_dim,},
                   mlp=MLP(in_dim=self.feature_dim,
                           out_dim=self.feature_dim),
                   apply_attention=True,
                   pooling_function=torch.mean,
                   use_auto_mapping=True)
        
    def forward(self, x):
        if "incabin_video" in x.keys():
            x_incabin_features = self.incabin_model(x["incabin_video"].permute(0,2,1,3,4).to("cuda:1")).logits
            x_incabin_features = x_incabin_features.to("cuda:0")
        if "video"in x.keys() : 
            x_exterior_features = self.exterior_model(x["video"].permute(0,2,1,3,4)).logits
        if (self.fusion_type == "af") or (self.fusion_type == "deepset"):
            x = self.fusion_layer({"incabin_video": x_incabin_features, 
                "video": x_exterior_features})
        else:
            x = self.fusion_layer([x_incabin_features, x_exterior_features])
        x = self.fc(x)
        return x
    


class TST_wrapper(torch.nn.Module):
    def __init__(self,
                 input_size=8,
                 hidden_size=768,
                 output_size=768,
                 num_layers=1,
                 activation="gelu",
                 dropout=0.1,
                 fc_dropout=0.1,
                 d_ff=768,
                 d_model=768,
                 num_variables = 16,
                 seq_len = 8,
                 n_heads=16,):
        super().__init__()
        self.tst = TST(c_in=num_variables,
            c_out=output_size,
            seq_len=seq_len,
            n_layers=num_layers,
            act=activation,
            fc_dropout=fc_dropout,
            dropout=dropout,
            d_ff=d_ff,
            d_model=d_model,
            n_heads=n_heads)
    def forward(self, x):
        return  self.tst(x)

    
class MM_MultiModalVMAE_shared_encoder(nn.Module):
    def __init__(self,
                 model_ckpt="MCG-NJU/videomae-base-finetuned-kinetics",
                 feature_dim = 768,
                 fusion_type="mean",
                 num_classes:int=2,
                 contrastive_loss:bool=False,
                 logit_scale_init_value = 2.6592):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.fusion_type = fusion_type
        self.incabin_model = VideoMAEForVideoClassification.from_pretrained(model_ckpt,
                                                                      ignore_mismatched_sizes=True).to("cuda:1")
        self.canbus_model = TST_wrapper().to("cuda:1")
        
        self.shared_encoding_layer = copy.deepcopy(self.incabin_model.videomae.encoder.layer[-1]).to("cuda:0")
        del self.incabin_model.videomae.encoder.layer[-1]
        self.incabin_model.classifier = torch.nn.Identity()
        

        self.exterior_model = VideoMAEForVideoClassification.from_pretrained(model_ckpt,
                                                                      ignore_mismatched_sizes=True).to("cuda:0")
        del self.exterior_model.videomae.encoder.layer[-1]
        self.exterior_model.classifier = torch.nn.Identity()
        # always average the available modalities
        self.fusion_layer = Add(avg=True).to("cuda:0")
        self.fc = torch.nn.Linear(self.feature_dim, self.num_classes).to("cuda:0")
        
        self.contrastive_loss = contrastive_loss
        if contrastive_loss:
            self.logit_scale = torch.nn.Parameter(torch.tensor(logit_scale_init_value))
        
        
    def forward(self, x, all_input=True):
        
        try:
            if torch.count_nonzero(x["canbus"])==0:
                del x["canbus"]
        except KeyError:
            pass
        try:
            if torch.count_nonzero(x["incabin_video"])==0:
                del x["incabin_video"]
        except KeyError:
            pass
        
        if ("incabin_video" in x.keys()) and all_input:
            x_features = self.incabin_model(x["incabin_video"].permute(0,2,1,3,4).to("cuda:1")).logits
            x_features = x_features.to("cuda:0")
            x_features = self.shared_encoding_layer(x_features.unsqueeze(dim=1))[0].squeeze()
        
        elif ("canbus" in x.keys()) and all_input:
            x_features = self.canbus_model(x["canbus"].type(torch.FloatTensor).to("cuda:1"))
            x_features = x_features.to("cuda:0")
            x_features = self.shared_encoding_layer(x_features.unsqueeze(dim=1))[0].squeeze()
        
        else:
            x_features=None
        
        if ("video"in x.keys()): 
            x_exterior_features = self.exterior_model(x["video"].permute(0,2,1,3,4).to("cuda:0")).logits
            x_exterior_features = self.shared_encoding_layer(x_exterior_features.unsqueeze(dim=1))[0].squeeze()       
        
        if  x_features is not None:
            x_fusion = self.fusion_layer([x_features, x_exterior_features])
            x = self.fc(x_fusion)

            if self.contrastive_loss and self.training:
                modality1_emb = x_exterior_features / x_exterior_features.norm(p=2, dim=-1, keepdim=True)
                modality2_emb = x_features / x_features.norm(p=2, dim=-1, keepdim=True)
                logit_scale = self.logit_scale.exp()
                logits_per_m1 = torch.matmul(modality1_emb, modality2_emb.t()) * logit_scale
                logits_per_m2 = logits_per_m1.t()
                loss = clip_loss(logits_per_m1)
                return x, loss
            elif self.training:
                return x, None
            else:
                return x
        else:
            x = self.fc(x_exterior_features)
            return x

        
class MM_MultiModalVMAE(nn.Module):
    def __init__(self,
                 model_ckpt="MCG-NJU/videomae-base-finetuned-kinetics",
                 feature_dim = 768,
                 fusion_type="mean",
                 num_classes:int=2,
                 contrastive_loss:bool=False,
                 logit_scale_init_value = 2.6592):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.fusion_type = fusion_type
        self.incabin_model = VideoMAEForVideoClassification.from_pretrained(model_ckpt,
                                                                      ignore_mismatched_sizes=True).to("cuda:1")
        self.canbus_model = TST_wrapper().to("cuda:1")
        self.incabin_model.classifier = torch.nn.Identity()
        self.exterior_model = VideoMAEForVideoClassification.from_pretrained(model_ckpt,
                                                                      ignore_mismatched_sizes=True).to("cuda:0")
        self.exterior_model.classifier = torch.nn.Identity()
        # always average the available modalities
        self.fusion_layer = Add(avg=True).to("cpu")
        self.fc = torch.nn.Linear(self.feature_dim, self.num_classes).to("cuda:0")
        
        self.contrastive_loss = contrastive_loss
        if contrastive_loss:
            self.logit_scale = torch.nn.Parameter(torch.tensor(logit_scale_init_value))
        
    def _compute_clip(self,x1,x2):
        modality1_emb = x1 / x1.norm(p=2, dim=-1, keepdim=True)
        modality2_emb = x2 / x2.norm(p=2, dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_m1 = torch.matmul(modality1_emb, modality2_emb.t()) * logit_scale
        logits_per_m2 = logits_per_m1.t()
        loss = clip_loss(logits_per_m1)
        return loss
    
    def forward(self, x, all_input=True):
        features = []
        
        if ("incabin_video" in x.keys()):
            in_cabin_features = self.incabin_model(x["incabin_video"].permute(0,2,1,3,4).to("cuda:1")).logits
            in_cabin_features = in_cabin_features.to("cuda:0")
            features.append(in_cabin_features)
        else:
            in_cabin_features = None
        if ("canbus" in x.keys()):
            canbus_features = self.canbus_model(x["canbus"].type(torch.FloatTensor).to("cuda:1"))
            canbus_features = canbus_features.to("cuda:0")
            features.append(canbus_features)
        else:
            canbus_features = None    
        if ("video"in x.keys()): 
            x_exterior_features = self.exterior_model(x["video"].permute(0,2,1,3,4).to("cuda:0")).logits    
            features.append(x_exterior_features)
            
        x_fusion = self.fusion_layer(features)
        x = self.fc(x_fusion)
        
        if self.contrastive_loss and self.training:
            clip_loss = 0
            
            if canbus_features is not None:
                clip_loss += self._compute_clip(x_exterior_features, canbus_features)
            if in_cabin_features is not None:
                clip_loss += self._compute_clip(x_exterior_features, in_cabin_features)
            
            return x, clip_loss
            
        elif self.training:
            return x, None
        else:
            return x
