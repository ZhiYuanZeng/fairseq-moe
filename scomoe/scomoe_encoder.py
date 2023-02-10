from fairseq.modules.fairseq_dropout import FairseqDropout
import fairseq
from fairseq.modules.layer_norm import LayerNorm
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union
from torch.functional import Tensor
from fairseq.distributed.fully_sharded_data_parallel import fsdp_wrap
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
import torch
from fairseq.modules.transformer_layer import FeedForwardNetwork, TransformerEncoderLayer
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP, TransformerEncoder
from fairseq import utils
import torch.nn as nn
from scomoe.gates import BalanceGate
from scomoe.moe import SCoMoELayer

class ScomoeTransformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        self.args = args
        super().__init__(args, dictionary, embed_tokens)
        moe_freq = max(getattr(args, 'encoder_moe_freq', 0), getattr(args, 'moe_freq', 0))
        assert moe_freq==2, 'currently only supports moe-freq=2'
        self.encoder_layers=[self.build_encoder_layer(args, is_moe_layer=False) 
                        for _ in range(args.encoder_layers//2)]
        self.attn_layers=[self.build_layer(args, EncoderSelfAttnLayer) 
                        for _ in range(args.encoder_layers//2)]
        self.ffn_layers=[self.build_layer(args, FFNEncoderLayer),]
        self.moe_layers=[]
        for i in range(args.encoder_layers//2):
            layer_idx=i if i!=(args.encoder_layers//2-1) else -1
            self.moe_layers.append(self.build_layer(args, ScomoeEncoderLayer, layer_idx=layer_idx))
        del self.layers
        if args.token_cluster or getattr(args, 'reorder_layers', False):
            utils.print_r0('**************** rerange encoder layers, put moe layer at last layers ****************')
            if args.token_cluster:
                # put one ffn layer before moe layers
                self.layers=self.encoder_layers+self.attn_layers+self.ffn_layers+self.moe_layers
            else: # reorder layers but without token cluster
                self.layers=self.encoder_layers+self.attn_layers+self.moe_layers
        else:
            # mix encoder_layers, attention layers and moe layers 
            self.layers=[]
            for i in range(len(self.encoder_layers)):
                self.layers.append(self.encoder_layers[i])
                self.layers.append(self.attn_layers[i])
                self.layers.append(self.moe_layers[i])

        self.num_layers = len(self.layers)
        self.layers = nn.ModuleList(self.layers)

    def build_layer(self, args, layer_cls, **kwargs):
        layer = layer_cls(args, **kwargs)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer
    
class EncoderSelfAttnLayer(TransformerEncoderLayer):
    def __init__(self, args) -> None:
        super().__init__(args)
        modules_to_del=[]
        for n,_ in self.named_modules():
            if hasattr(self, n):
                modules_to_del.append(n)
        for n in modules_to_del:
            delattr(self, n)
        
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before

    def forward(self, x, encoder_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor] = None):
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        return x, None

class FFNEncoderLayer(nn.Module):
    def __init__(self, args, ) -> None:
        super().__init__()
        self.is_moe_layer=False
        self.embed_dim = args.encoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        ffn_dim = args.encoder_ffn_embed_dim
        self.layer_norm=LayerNorm(self.embed_dim)
        self.ffn=FeedForwardNetwork(args, self.embed_dim, ffn_dim, self.dropout_module)
    
    def forward(self, x, encoder_padding_mask=None):
        res=x
        x=self.layer_norm(x)
        x=self.ffn(x)
        x=res+x
        return x, None

    def upgrade_state_dict_named(self, *args, **kwargs):
        pass

class ScomoeEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args, layer_idx):        
        self.layer_idx=layer_idx
        super().__init__(args, is_moe_layer=True)
        modules_to_del=[]
        for n,_ in self.named_modules():
            if 'attn' in n and hasattr(self, n):
                modules_to_del.append(n)
        for n in modules_to_del:
            delattr(self, n)
        self.token_cluster=args.token_cluster

    def forward(self, x, encoder_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor] = None):
        layer_idx=self.moe_layer.layer_idx
        norm_before=self.normalize_before

        res=x
        if norm_before:
            x = self.final_layer_norm(x)
        
        if self.token_cluster:
            # x - seq_len, batch_size, model_dim
            if layer_idx==0:
                x = x.transpose(0, 1) # batch_size, seq_len, model_dim
            if getattr(self.args, "use_moe_pad_mask", False):
                x, l_aux = self.moe_layer(x, res_connection=res, input_padding_mask=encoder_padding_mask)
            else:
                x, l_aux = self.moe_layer(x, res_connection=res,)
            if layer_idx==-1:
                x = x.transpose(0, 1) # seq_len, batch_size, model_dim
            # we move res-connection inside moe layer
        else:
            x = x.transpose(0, 1) # batch_size, seq_len, model_dim
            if getattr(self.args, "use_moe_pad_mask", False):
                x, l_aux = self.moe_layer(x, input_padding_mask=encoder_padding_mask)
            else:
                x, l_aux = self.moe_layer(x)
            x = x.transpose(0, 1) # seq_len, batch_size, model_dim
            x = self.residual_connection(x, res)

        if not norm_before:
            x = self.final_layer_norm(x)
        return x, l_aux
    
    def build_moe_layer(self, gate, experts, args):
        return SCoMoELayer(gate, experts, args, self.embed_dim, self.layer_idx, autoregressive=False)
    
    def make_experts(self, args, embed_dim, expert_ffn_dim, dropout_module):
        if args.scomoe_type=='feat' and self.layer_idx!=0:
            local_dim = int(args.ratio1*embed_dim)
            if args.ratio3>0:
                node_dim = int(args.ratio2*embed_dim)
            else:
                node_dim = embed_dim-local_dim
            global_dim = embed_dim-local_dim-node_dim

        expert_dict={
            "local":[],
            "node":[],
            "global":[],
        }


        world_size = 1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size()
        ddp_rank = fairseq.distributed_utils.get_data_parallel_rank()
        start_seed = torch.randint(1000000, (1,)).item()
        # at least as many experts than gpus
        if args.moe_expert_count >= world_size:
            assert args.moe_expert_count % world_size == 0, f'{args.moe_expert_count}, {world_size}'
            local_moe_expert_count = args.moe_expert_count // world_size
            for i in range(local_moe_expert_count):
                with utils.set_torch_seed(start_seed + ddp_rank * local_moe_expert_count + i):
                    if args.scomoe_type=='feat' and self.layer_idx!=0:
                        if local_dim!=0:
                            expert_dict['local'].append(FeedForwardNetwork(args, local_dim, expert_ffn_dim, dropout_module))
                        if node_dim!=0:
                            expert_dict['node'].append(FeedForwardNetwork(args, node_dim, expert_ffn_dim, dropout_module))
                        if global_dim!=0:
                            expert_dict['global'].append(FeedForwardNetwork(args, global_dim, expert_ffn_dim, dropout_module))
                    else:
                        expert_dict['global'].append(FeedForwardNetwork(args, embed_dim, expert_ffn_dim, dropout_module))

        # less experts than gpus
        else:
            assert world_size % args.moe_expert_count == 0, f'{world_size}, {args.moe_expert_count}'
            # initialize each FFN with the same seed on different GPUs
            with utils.set_torch_seed(start_seed + ddp_rank % args.moe_expert_count):
                if args.scomoe_type=='feat' and self.layer_idx!=0:
                    if local_dim!=0:
                        expert_dict['local'].append(FeedForwardNetwork(args, local_dim, expert_ffn_dim, dropout_module))
                    if node_dim!=0:
                        expert_dict['node'].append(FeedForwardNetwork(args, node_dim, expert_ffn_dim, dropout_module))
                    if global_dim!=0:
                        expert_dict['global'].append(FeedForwardNetwork(args, global_dim, expert_ffn_dim, dropout_module))
                else:
                    expert_dict['global'].append(FeedForwardNetwork(args, embed_dim, expert_ffn_dim, dropout_module))
        expert_dict['local']=nn.ModuleList(expert_dict['local'])
        expert_dict['node']=nn.ModuleList(expert_dict['node'])
        expert_dict['global']=nn.ModuleList(expert_dict['global'])
        experts = nn.ModuleDict(expert_dict)
        return experts

    def build_gate(self, args):
        if self.layer_idx==0 and args.token_cluster:
            Gate=BalanceGate
            gate = Gate(
                args,
                self.embed_dim,
                args.moe_expert_count,
                use_fp32=args.moe_gating_use_fp32,
            )
        else:
            gate=super().build_gate(args)
        return gate

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        torch.nn.LayerNorm
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]
