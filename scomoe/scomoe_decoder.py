from fairseq.modules.fairseq_dropout import FairseqDropout
import fairseq
from fairseq.modules.layer_norm import LayerNorm
from typing import Dict, List, Optional
from torch.functional import Tensor
from fairseq.distributed.fully_sharded_data_parallel import fsdp_wrap
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
import torch
from fairseq.modules.transformer_layer import FeedForwardNetwork, TransformerDecoderLayer
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP, TransformerDecoder, div_by_world_size
from fairseq import utils
import torch.nn as nn
from scomoe.gates import BalanceGate
from scomoe.moe import SCoMoELayer

class ScomoeTransformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        self.args = args
        super().__init__(args, dictionary, embed_tokens)
        moe_freq = max(getattr(args, 'decoder_moe_freq', 0), getattr(args, 'moe_freq', 0))
        assert moe_freq==2, 'currently only supports moe-freq=2'
        self.decoder_layers=[self.build_decoder_layer(args, is_moe_layer=False) 
                        for _ in range(args.decoder_layers//2)]
        self.self_cross_attn_layers=[
            [self.build_self_attn_layer(args), self.build_cross_attn_layer(args)] for _ in range(args.decoder_layers//2)
        ]

        self.ffn_layers=[self.build_ffn_layer(args),]
        self.moe_layers=[]
        for i in range(args.decoder_layers//2):
            layer_idx=i if i!=(args.decoder_layers//2-1) else -1
            self.moe_layers.append(self.build_moe_layer(args, layer_idx))
        del self.layers

        if args.token_cluster or getattr(args, 'reorder_layers', False):
            utils.print_r0('**************** reorder decoder layers, put moe layer at last layers ****************')
            self.self_cross_attn_layers=sum(self.self_cross_attn_layers, [])
            assert len(self.self_cross_attn_layers)==args.decoder_layers
            self.layers=self.decoder_layers+self.self_cross_attn_layers+self.ffn_layers+self.moe_layers
        else:
            # mix encoder_layers, attention layers and moe layers 
            self.layers=[]
            for i,l in enumerate(self.decoder_layers):
                self.layers.append(self.decoder_layers[i])
                self.layers.append(self.self_cross_attn_layers[i][0]),
                self.layers.append(self.self_cross_attn_layers[i][1])
                self.layers.append(self.moe_layers[i])

        self.num_layers = len(self.layers)
        self.layers = nn.ModuleList(self.layers)

    def build_cross_attn_layer(self, args):
        layer = DecoderCrossAttnLayer(args)
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

    def build_ffn_layer(self, args):
        layer = DecoderFFNLayer(args)
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

    def build_self_attn_layer(self, args):
        layer = DecoderSelfAttnLayer(args)
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
    
    def build_moe_layer(self, args, layer_idx):
        layer = DecoderMoeLayer(args, layer_idx)
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

class DecoderSelfAttnLayer(TransformerDecoderLayer):
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
        self.normalize_before = args.decoder_normalize_before

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        return x, attn, None, None

class DecoderCrossAttnLayer(TransformerDecoderLayer):
    def __init__(self, args) -> None:
        super().__init__(args)
        modules_to_del=[]
        for n,_ in self.named_modules():
            if hasattr(self, n):
                modules_to_del.append(n)
        for n in modules_to_del:
            delattr(self, n)
        
        self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before
    
    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        if need_head_weights:
            need_attn = True

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
        else:
            attn = None
        return x, attn, None, None

class DecoderFFNLayer(nn.Module):
    # TODO: simplify the module
    def __init__(self, args, ) -> None:
        super().__init__()
        self.is_moe_layer=False
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        ffn_dim = args.decoder_ffn_embed_dim
        self.layer_norm=LayerNorm(self.embed_dim)
        self.ffn=FeedForwardNetwork(args, self.embed_dim, ffn_dim, self.dropout_module)
    
    def forward(
        self,
        x,
        *args, **kwargs
    ):
        res=x
        x=self.layer_norm(x)
        x=self.ffn(x)
        x=res+x
        return x, None, None, None
    
    def upgrade_state_dict_named(self, *args, **kwargs):
        pass

class DecoderMoeLayer(TransformerDecoderLayer):
    # TODO: simplify the module
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

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
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
                x, l_aux = self.moe_layer(x, res_connection=res, input_padding_mask=self_attn_padding_mask)
            else:
                x, l_aux = self.moe_layer(x, res_connection=res,)
            if layer_idx==-1:
                x = x.transpose(0, 1) # seq_len, batch_size, model_dim
            # we move res-connection inside moe layer
        else:
            x = x.transpose(0, 1) # batch_size, seq_len, model_dim
            if getattr(self.args, "use_moe_pad_mask", False):
                x, l_aux = self.moe_layer(x, input_padding_mask=self_attn_padding_mask)
            else:
                x, l_aux = self.moe_layer(x)
            x = x.transpose(0, 1) # seq_len, batch_size, model_dim
            x = self.residual_connection(x, res)
        
        if not norm_before:
            x = self.final_layer_norm(x)
        return x, None, None, l_aux
    
    def build_moe_layer(self, gate, experts, args):
        return SCoMoELayer(gate, experts, args, self.embed_dim, self.layer_idx, autoregressive=True)
    
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
            gate = super().build_gate(args)
        return gate
