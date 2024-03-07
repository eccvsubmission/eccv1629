# -*- coding: utf-8 -*-
import torch
from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn import functional as F
from typing import Callable, Dict, List, Optional, Tuple


class AttentionFusionModule(nn.Module):
    """
    Fuse embeddings through weighted sum of the corresponding linear projections.
    Linear layer for learning the weights.
    Copied from: https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/modules/layers/attention.py

    Args:
        channel_to_encoder_dim: mapping of channel name to the encoding dimension
        encoding_projection_dim: common dimension to project the encodings to.
        defaults to min of the encoder dim if not set

    """
    def __init__(
        self,
        channel_to_encoder_dim: Dict[str, int],
        encoding_projection_dim: Optional[int] = None,
    ):
        super().__init__()
        attn_in_dim = sum(channel_to_encoder_dim.values())
        self.attention = nn.Sequential(
            nn.Linear(attn_in_dim, len(channel_to_encoder_dim)),
            nn.Softmax(-1),
        )
        if encoding_projection_dim is None:
            encoding_projection_dim = min(channel_to_encoder_dim.values())

        encoding_projection = {}
        for channel in sorted(channel_to_encoder_dim.keys()):
            encoding_projection[channel] = nn.Linear(
                channel_to_encoder_dim[channel], encoding_projection_dim
            )
        self.encoding_projection = nn.ModuleDict(encoding_projection)

    def forward(self, embeddings: Dict[str, Tensor]) -> Tensor:
        concatenated_in = torch.cat(
            [embeddings[k] for k in sorted(embeddings.keys())], dim=-1
        )
        attention_weights = self.attention(concatenated_in)
        projected_embeddings: List[Tensor] = []
        for channel, projection in self.encoding_projection.items():
            projected_embedding = projection(embeddings[channel])
            projected_embeddings.append(projected_embedding)

        for i in range(len(projected_embeddings)):
            projected_embeddings[i] = (
                attention_weights[:, i].unsqueeze(-1) * projected_embeddings[i]
            )

        fused = torch.sum(torch.stack(projected_embeddings), dim=0)
        return fused




class DeepsetFusionModule(nn.Module):
    """
    Fuse embeddings through stacking followed by pooling strategy and MLP
    See https://arxiv.org/pdf/2003.01607.pdf

    Args:
        channel_to_encoder_dim (Dict[str, int]): mapping of channel name to the\
        encoding dimension
        mlp (nn.Module): MLP with in dim as projection dim (min of embed dim).\
        Use MLP for mlp_classifier for default mlp.
        pooling_function (Callable): Pooling function to combine the tensors,\
        like torch.median\
        apply_attention (bool): If self attention (2 layer net) is applied before\
        stacking embeddings, defaults to False.
        attention_dim (int): intermediate dim for attention layer.\
        defaults to projection dim / 2
        modality_normalize (bool): If normalization is applied along the modality axis,\
        defaults to False
        norm_factor(float): norm factor for normalization, defaults to 2.0
        use_auto_mapping(bool): If true, projection layer to min embedding dim \
        is applied to the embeddings. defaults to False

    """

    def __init__(
        self,
        channel_to_encoder_dim: Dict[str, int],
        mlp: nn.Module,
        pooling_function: Callable,
        apply_attention: bool = False,
        attention_dim: Optional[int] = None,
        modality_normalize: bool = False,
        norm_factor: float = 2.0,
        use_auto_mapping: bool = False,
    ):
        super().__init__()
        self.apply_attention = apply_attention
        self.modality_normalize = modality_normalize
        self.norm_factor = norm_factor
        self.use_auto_mapping = use_auto_mapping
        projection_dim = DeepsetFusionModule.get_projection_dim(
            channel_to_encoder_dim, use_auto_mapping
        )
        if self.use_auto_mapping:
            self.projections = nn.ModuleDict(
                {
                    channel: nn.Linear(dim, projection_dim)
                    for channel, dim in channel_to_encoder_dim.items()
                }
            )
        else:
            self.projections = nn.ModuleDict(
                {channel: nn.Identity() for channel in channel_to_encoder_dim}
            )
        if self.apply_attention:
            self.attention: nn.Module
            if attention_dim is None:
                # default value as per older implementation
                attention_dim = projection_dim // 2
            self.attention = nn.Sequential(
                nn.Linear(projection_dim, attention_dim),
                nn.Tanh(),
                nn.Linear(attention_dim, 1),
                # channel axis
                nn.Softmax(dim=-2),
            )
        else:
            self.attention = nn.Identity()

        self.pooling_function = pooling_function
        self.mlp = mlp

    def forward(self, embeddings: Dict[str, Tensor]) -> Tensor:

        projections = {}
        for channel, projection in self.projections.items():
            projections[channel] = projection(embeddings[channel])

        embedding_list = [projections[k] for k in sorted(projections.keys())]

        # bsz x channels x projected_dim
        stacked_embeddings = torch.stack(embedding_list, dim=1)

        if self.apply_attention:
            attn_weights = self.attention(stacked_embeddings)
            stacked_embeddings = stacked_embeddings * attn_weights

        if self.modality_normalize:
            normalized_embeddings = F.normalize(
                stacked_embeddings, p=self.norm_factor, dim=1
            )
        else:
            normalized_embeddings = F.normalize(
                stacked_embeddings, p=self.norm_factor, dim=2
            )

        pooled_features = self._pool_features(normalized_embeddings)
        fused = self.mlp(pooled_features)
        return fused

    @classmethod
    def get_projection_dim(
        cls, channel_to_encoder_dim: Dict[str, int], use_auto_mapping: bool
    ) -> int:
        if use_auto_mapping:
            projection_dim = min(channel_to_encoder_dim.values())
        else:
            encoder_dim = set(channel_to_encoder_dim.values())
            if len(encoder_dim) != 1:
                raise ValueError(
                    "Encoder dimension should be same for all channels \
                    if use_auto_mapping is set to false"
                )
            projection_dim = encoder_dim.pop()
        return projection_dim

    def _pool_features(self, embeddings: Tensor) -> Tensor:
        pooled_embeddings = self.pooling_function(embeddings, dim=1)
        if torch.jit.isinstance(pooled_embeddings, Tuple[Tensor, Tensor]):
            return pooled_embeddings.values
        if not isinstance(pooled_embeddings, Tensor):
            raise ValueError(
                f"Result from pooling function should be a tensor.\
             {self.pooling_function} does not satisfy that"
            )
        return pooled_embeddings


# https://github.com/pliang279/HighMMT/blob/main/fusions/common_fusions.py   

class Concat(nn.Module):
    def __init__(self):
        super(Concat,self).__init__()
    
    def forward(self, modalities, training=False):
        flattened = []
        for modality in modalities:
            flattened.append(torch.flatten(modality, start_dim=1))
        return torch.cat(flattened, dim=1)


class Add(nn.Module):
    def __init__(self,avg=False):
        super(Add,self).__init__()
        self.avg=avg
    def forward(self,modalities,training=False):
        out=modalities[0]
        for i in range(len(modalities)-1):
            out += modalities[i+1]
        if self.avg:
            return out / len(modalities)
        return out

class ConcatWithLinear(nn.Module):
    # input dim, output_dim: the in/out dim of the linear layer
    def __init__(self, input_dim, output_dim, concat_dim=1):
        super(ConcatWithLinear,self).__init__()
        self.concat_dim = concat_dim
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, modalities, training=False):
        return self.fc(torch.cat(modalities, dim=self.concat_dim))


class TensorFusion(nn.Module):
    # https://github.com/Justin1904/TensorFusionNetworks/blob/master/model.py
    def __init__(self):
        super().__init__()

    def forward(self, modalities, training=False):
        if len(modalities) == 1:
            return modalities[0]

        mod0 = modalities[0]
        nonfeature_size = mod0.shape[:-1]

        m = torch.cat((Variable(torch.ones(*nonfeature_size, 1).type(mod0.dtype).to(mod0.device), requires_grad=False), mod0), dim=-1)
        for mod in modalities[1:]:
            mod = torch.cat((Variable(torch.ones(*nonfeature_size, 1).type(mod.dtype).to(mod.device), requires_grad=False), mod), dim=-1)
            fused = torch.einsum('...i,...j->...ij', m, mod)
            m = fused.reshape([*nonfeature_size, -1])

        return m


class LowRankTensorFusion(nn.Module):
    # https://github.com/Justin1904/Low-rank-Multimodal-Fusion
    # input_dims: list or tuple of integers indicating input dimensions of the modalities
    # output_dim: output dimension
    # rank: a hyperparameter of LRTF. See link above for details
    def __init__(self, input_dims, output_dim, rank, flatten=True):
        super(LowRankTensorFusion, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.rank = rank
        self.flatten = flatten

        # low-rank factors
        self.factors = []
        for input_dim in input_dims:
            factor = nn.Parameter(torch.Tensor(self.rank, input_dim+1, self.output_dim))#.cuda()
            nn.init.xavier_normal(factor)
            self.factors.append(factor)

        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank))#.cuda()
        self.fusion_bias = nn.Parameter(torch.Tensor(1, self.output_dim))#.cuda()
        # init the fusion weights
        nn.init.xavier_normal(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, modalities, training=False):
        batch_size = modalities[0].shape[0]
        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        fused_tensor = 1
        for (modality, factor) in zip(modalities, self.factors):
            ones = Variable(torch.ones(batch_size, 1).type(modality.dtype), requires_grad=False)
            
            if self.flatten:
                modality_withones = torch.cat((ones, torch.flatten(modality,start_dim=1)), dim=1)
            else:
                modality_withones = torch.cat((ones, modality), dim=1)
            modality_factor = torch.matmul(modality_withones, factor)
            fused_tensor = fused_tensor * modality_factor

        output = torch.matmul(self.fusion_weights, fused_tensor.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        return output


from typing import Callable, List, Optional, Union

import torch
from torch import nn


class MLP(nn.Module):
    """A multi-layer perceptron module.

    This module is a sequence of linear layers plus activation functions.
    The user can optionally add normalization and/or dropout to each of the layers.

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        hidden_dims (Optional[List[int]]): Output dimension for each hidden layer.
        dropout (float): Probability for dropout layer.
        activation (Callable[..., nn.Module]): Which activation
            function to use. Supports module type or partial.
        normalization (Optional[Callable[..., nn.Module]]): Which
            normalization layer to use (None for no normalization).
            Supports module type or partial.

    Inputs:
        x (Tensor): Tensor containing a batch of input sequences.
    ​
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Optional[Union[int, List[int]]] = None,
        dropout: float = 0.5,
        activation: Callable[..., nn.Module] = nn.ReLU,
        normalization: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")
        layers = nn.ModuleList()

        if hidden_dims is None:
            hidden_dims = []

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if normalization:
                layers.append(normalization(hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
