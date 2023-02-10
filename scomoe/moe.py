import math
import time
from scomoe.utils import is_inf, is_nan
import torch.distributed as dist
import torch
import fairseq.utils as utils
from torch import Tensor, nn
from fairseq.models.transformer import div_by_world_size
import functools
from fairseq.distributed.fully_sharded_data_parallel import fsdp_wrap
from typing import Any, List, Union
from fairseq.modules.moe.moe_layer import MOELayer
import fairseq.distributed.utils as  distributed_utils
from fairseq.modules.moe.moe_layer import _AllToAll
from scomoe.gates import BalanceGate
from torch.nn.functional import pad

def fsdp_wrap_expert(args, process_group, experts):
    # Wrap MoE layer with FSDP using a process group with all replicated ranks
    world_size = distributed_utils.get_data_parallel_group().size()
    pg_size = process_group.size()
    num_experts = world_size/pg_size
    
    for i, expert in enumerate(experts):
        experts[i] = fsdp_wrap(
            expert, process_group=process_group, min_num_params=0
        )
    if getattr(args, "moe_normalize_expert_grad", "world_size") == "sqrt_world_size":
        expert_normalization_term = math.sqrt(num_experts)
    else:
        expert_normalization_term = num_experts
    for p in experts.parameters():
        p.expert = True
        p.register_hook(functools.partial(div_by_world_size, expert_normalization_term))

    return experts
        

class SCoMoELayer(MOELayer):
    def __init__(self, gate, experts: Union, args, d_model, layer_idx, autoregressive=False) -> None:
        local_experts=experts['local']
        node_experts=experts['node']
        global_experts=experts['global']
        
        all_experts=nn.ModuleList([])
        all_experts.extend(local_experts).extend(node_experts).extend(global_experts)

        setattr(gate, 'autoregressive', autoregressive)
        self.autoregressive=autoregressive
        
        super().__init__(gate, all_experts, args)
        self.token_cluster=args.token_cluster
        self.num_local_experts=args.moe_expert_count//self.all2all_size
        
        for p in local_experts.parameters():
            assert p.expert
        for p in node_experts.parameters():
            assert p.expert
        for p in global_experts.parameters():
            assert p.expert
        
        self.all2all_group = get_all2all_group()
        self.node_group = get_node_group()
        
        self.all2all_size=distributed_utils.get_world_size(self.all2all_group)
        self.local_world_size=distributed_utils.get_world_size(self.node_group)
        assert self.local_world_size<=8
        assert self.all2all_size%self.local_world_size==0
        self.node_num=self.all2all_size//self.local_world_size
        rank=distributed_utils.get_global_rank()
        self.node_rank, self.local_rank=rank//self.local_world_size, rank%self.local_world_size
        # print('rank: {}, node_rank:{}, local_rank:{}, node_num:{}, local_world_size:{}, global_world_size:{}'.format(
        #         rank, self.node_rank, self.local_rank, self.node_num, self.local_world_size, self.all2all_size))
        
        self.ratio1=args.ratio1
        self.ratio2=args.ratio2
        self.greedy_autoregressive_inference=getattr(args, 'greedy_autoregressive_inference', False)
        self.gate_type=getattr(args, 'gate_type', 'softmax')
        
        if layer_idx==0:
            self.scomoe_type = 'none'
        else:
            self.scomoe_type=args.scomoe_type
        if self.scomoe_type=='feat':
            self.local_dim=int(local_experts[0].embed_dim) if (args.ratio1>0.) else 0
            self.node_dim=int(node_experts[0].embed_dim) if (args.ratio2>0.) else 0
            self.global_dim=int(global_experts[0].embed_dim) if (args.ratio3>0.) else 0
            self.local_experts=fsdp_wrap_expert(args, self.expert_group, local_experts) \
                if self.local_dim>0 else nn.ModuleList([])
            self.node_experts=fsdp_wrap_expert(args, self.expert_group, node_experts) \
                if self.node_dim>0 else nn.ModuleList([])
            self.global_experts=fsdp_wrap_expert(args, self.expert_group, global_experts) \
                if self.global_dim>0 else nn.ModuleList([])
            
            del self.experts
        else:
            self.experts=fsdp_wrap_expert(args, self.expert_group, self.experts)
        if getattr(args, 'layernorm_after_moe_layer', False):
            self.post_layernorm=nn.LayerNorm(d_model, elementwise_affine=False)
        else:
            self.post_layernorm=None
        self.layer_idx=layer_idx
        self.has_print_examples=False

        utils.print_r0('###################### moe-layer{} ###################### '.format(layer_idx))
        utils.print_r0('scomoe-type:{}'.format(self.scomoe_type))
        utils.print_r0('token-cluster:{}'.format(self.token_cluster))
        if self.scomoe_type!='none':
            utils.print_r0('ratios: local {}, node {}, global {}'.format(args.ratio1, args.ratio2, args.ratio3))
        if self.scomoe_type=='feat':
            utils.print_r0('local_dim:', self.local_dim)      
            utils.print_r0('node_dim:', self.node_dim)        
            utils.print_r0('global_dim:', self.global_dim)
        utils.print_r0('##########################################################')
        
    def forward(self, *input: Tensor, res_connection=None, input_padding_mask=None, **kwargs: Any):
        assert len(input) == 1, "only single input Tensor supported"
        input = input[0]
        if input_padding_mask is None:
            input_padding_mask=torch.full([input.shape[0], input.shape[1]], False, device=input.device)

        if self.token_cluster:
            return self.forward_with_token_cluster(input, res_connection, input_padding_mask, *kwargs)
        else:
            return self.forward_wo_token_cluster(input, input_padding_mask, *kwargs)
    
    def forward_wo_token_cluster(self, input, input_padding_mask, *kwargs):
        ############################## padding (copy from moe_layer.py) #################################
            if input_padding_mask.shape[1] != input.shape[1]:
                input_padding_mask=input_padding_mask[:,-1]
            reshaped_input=input.reshape(-1, input.shape[-1])
            input_shape = list(reshaped_input.shape)
            reshaped_input_padding_mask=input_padding_mask.reshape(-1)

            expected_dim = int(distributed_utils.all_reduce(
                input_shape[0] * torch.ones((1,), dtype=torch.long, device=input.device),
                group=dist.group.WORLD,
                op="max",
            ).item())
            padded_input = torch.zeros(
                (expected_dim, input_shape[1]),
                dtype=input.dtype, layout=input.layout, device=input.device)
            padded_input[:input_shape[0], :] = reshaped_input
            reshaped_input = padded_input
            padded_input_padding_mask = torch.ones(
                (expected_dim,), dtype=torch.bool, device=padded_input.device
            )
            if reshaped_input_padding_mask is not None:
                padded_input_padding_mask[:input_shape[0]] = reshaped_input_padding_mask
            else:
                padded_input_padding_mask[:input_shape[0]] = False
            reshaped_input_padding_mask = padded_input_padding_mask
            ###############################################################################################

            if self.scomoe_type=='none':
                combined_output, l_aux= self.moe(reshaped_input, reshaped_input_padding_mask, self.all2all_group)
            elif self.scomoe_type=='feat':
                combined_output, l_aux=self.scomoe_dmodel(reshaped_input, reshaped_input_padding_mask)
            else:
                combined_output, l_aux=self.scomoe_seq(reshaped_input, reshaped_input_padding_mask)
            
            result = combined_output[:input_shape[0], :]
            result = result.reshape_as(input)
            self.record_all_to_all_stats()
            return result, l_aux

    def forward_with_token_cluster(self, input: Tensor, res_connection=None, input_padding_mask=None, **kwargs: Any) -> Tensor:
        # utils.print_r0('layer:{} input:{}'.format(self.layer_idx, input.abs().max()))
        d_model = input.shape[-1]
        if self.layer_idx==0:
            # first moe layer do not need padding unless we use balanced gate
            bsz, seq_len, _=input.shape
            if input_padding_mask.shape[1]!=seq_len:
                assert not self.training
                assert seq_len==1
                input_padding_mask=input_padding_mask[:,-1:]
            
            total_expert_num=self.all2all_size*len(self.experts)
            
            #--------------maybe right-pad batch size------------
            # batch size should % num_experts for Balance Gate at validation and inference
            if isinstance(self.gate, BalanceGate) and self.autoregressive and (not self.training) \
                    and (not self.greedy_autoregressive_inference) and bsz % total_expert_num!=0:
                pad_bsz_num=total_expert_num-bsz%total_expert_num
                input=pad(input, pad=(0,0,0,0, 0, pad_bsz_num), value=0) # right pad
                input_padding_mask=pad(input_padding_mask, pad=(0,0, 0, pad_bsz_num), value=True)
            else:
                pad_bsz_num=0
            #-----------------------------------------------
            
            input_shape=input.shape
            reshaped_input=input.reshape(-1, d_model)
            reshaped_input_padding_mask=input_padding_mask.view(-1,)

            #--------------maybe right-pad token------------
            if isinstance(self.gate, BalanceGate) and reshaped_input.shape[0]%total_expert_num!=0:
                pad_num=total_expert_num-reshaped_input.shape[0]%total_expert_num
                reshaped_input=pad(reshaped_input, pad=(0, 0, 0, pad_num), value=0)
                reshaped_input_padding_mask=pad(reshaped_input_padding_mask, pad=(0, pad_num), value=True)
            else:
                pad_num=0
            #-----------------------------------------------

            result, l_aux, i_sort, input_splits, output_splits, dispatched_reshaped_input_padding_mask=\
                self.token_cluering_moe(reshaped_input, reshaped_input_padding_mask, seq_len=seq_len)

            share_mem(action='write', key='pad_num', value=pad_num, )
            share_mem(action='write', key='pad_bsz_num', value=pad_bsz_num, )
            share_mem(action='write', key='i_sort', value=i_sort, )
            share_mem(action='write', key='input_splits', value=input_splits, )
            share_mem(action='write', key='output_splits', value=output_splits, )
            share_mem(action='write', key='input_shape', value=input_shape, )
            share_mem(action='write', key='dispatched_reshaped_input_padding_mask', value=dispatched_reshaped_input_padding_mask, ) # dispatched padding
            assert not is_inf(result) and not is_nan(result)
        else:
            input_shape = list(input.shape)
            reshaped_input=input
            reshaped_input_padding_mask = share_mem(
                action='read', key='dispatched_reshaped_input_padding_mask')
            assert input_shape[0]==reshaped_input_padding_mask.shape[0]
            expected_dim = int(distributed_utils.all_reduce(
                input_shape[0] * torch.ones((1,), dtype=torch.long, device=input.device),
                group=dist.group.WORLD,
                op="max",
            ).item())
            padded_input = torch.zeros(
                (expected_dim, input_shape[1]),
                dtype=input.dtype, layout=input.layout, device=input.device)
            padded_input[:input_shape[0], :] = reshaped_input # right padding
            reshaped_input = padded_input
            padded_input_padding_mask = torch.ones(
                (expected_dim,), dtype=torch.bool, device=padded_input.device
            )
            if reshaped_input_padding_mask is not None:
                padded_input_padding_mask[:input_shape[0]] = reshaped_input_padding_mask
            else:
                padded_input_padding_mask[:input_shape[0]] = False
            reshaped_input_padding_mask = padded_input_padding_mask
       
            if self.scomoe_type=='none':
                combined_output, l_aux= self.moe(reshaped_input, reshaped_input_padding_mask, self.all2all_group)
            elif self.scomoe_type=='feat':
                combined_output, l_aux=self.scomoe_dmodel(reshaped_input, reshaped_input_padding_mask)
            else:
                combined_output, l_aux=self.scomoe_seq(reshaped_input, reshaped_input_padding_mask)
            
            result = combined_output[:input_shape[0], :]
            assert list(result.shape)==input_shape, '{}, {}'.format(result.shape, input_shape)
            result += res_connection
            assert not is_inf(result) and not is_nan(result)
            
        if self.layer_idx==-1:
            pad_num=share_mem(action='read', key='pad_num', )
            pad_bsz_num=share_mem(action='read', key='pad_bsz_num', )
            i_sort=share_mem(action='read', key='i_sort', )
            input_splits=share_mem(action='read', key='input_splits', )
            output_splits=share_mem(action='read', key='output_splits', )
            input_shape=share_mem(action='read', key='input_shape', )
            result = self.all_to_all_wrapper(result, group=None, input_splits=output_splits, output_splits=input_splits)
            assert result.shape[0]==len(i_sort)
            result=result[i_sort]
            if pad_num !=0:
                result=result[:-pad_num]
            
            result=result.reshape(input_shape)
            if pad_bsz_num != 0:
                result=result[:-pad_bsz_num]
        if not hasattr(self, 'metadata'):
            self.metadata={}
        self.record_all_to_all_stats()
        return result, l_aux

    def scomoe_dmodel(self, reshaped_input, reshaped_padding):
        if hasattr(self, 'input_projection'):
            reshaped_input=self.input_projection(reshaped_input)
        local_input=reshaped_input[:, :self.local_dim]
        node_input=reshaped_input[:, self.local_dim:self.local_dim+self.node_dim]
        global_input=reshaped_input[:, -self.global_dim:]

        route_scores=self.gate.wg(reshaped_input) # (s, e)
        reshaped_route_scores=route_scores.view(reshaped_input.shape[0], self.node_num, self.local_world_size, self.num_local_experts)

        local_route_scores=reshaped_route_scores[:, self.node_rank, self.local_rank]
        node_route_scores=reshaped_route_scores[:, self.node_rank].view(route_scores.shape[0], -1)

        if not self.has_print_examples:
            utils.print_r0('>>>>>>>>>>>>>>>>>> hir_dmodel_moe >>>>>>>>>>>>>>>>>>>>>')
            utils.print_r0('local-d:{}, node-d:{}, global-d:{}, all:{}'.format(
                self.local_dim, self.node_dim, self.global_dim, self.local_dim+self.node_dim+self.global_dim))
            utils.print_r0('local-r:{}, node-r:{}, global-r:{}, all:{}'.format(
                local_route_scores.shape, node_route_scores.shape, route_scores.shape, route_scores.shape))
            utils.print_r0('local-x:{}, node-x:{}, global-x:{}, all:{}'.format(
                local_input.shape, node_input.shape, global_input.shape, reshaped_input.shape))
            utils.print_r0('local-e:{}, node-e:{}, global-e:{}, all:{}'.format(
                len(self.local_experts), len(self.node_experts), len(self.global_experts), 
                len(self.local_experts)+len(self.node_experts)+len(self.global_experts)))
            utils.print_r0('##########################################################')

            self.has_print_examples=True

        if self.local_dim>0:
            local_output, l_aux1=self.local_moe(local_input, input_mask=reshaped_padding,  group=None, route_scores=local_route_scores, experts=self.local_experts)
        else:
            local_output, l_aux1=None, 0.

        if self.node_dim>0:
            node_output, l_aux2=self.moe(node_input, input_mask=reshaped_padding, group=self.node_group, route_scores=node_route_scores, experts=self.node_experts)
        else:
            node_output, l_aux2=None, 0.

        if self.global_dim>0:
            global_output, l_aux3=self.moe(global_input, input_mask=reshaped_padding, group=self.all2all_group, route_scores=route_scores, experts=self.global_experts)
        else:
            global_output, l_aux3=None, 0.
        
        output_to_cat=[o for o in [local_output, node_output, global_output] if o is not None]
        
        l_aux1_weight=int(l_aux1!=0)
        l_aux2_weight=int(l_aux2!=0)
        l_aux3_weight=int(l_aux3!=0)
        total_laux=(l_aux1_weight*l_aux1+l_aux2_weight*l_aux2+l_aux3_weight*l_aux3)/(l_aux1_weight+l_aux2_weight+l_aux3_weight)
        
        combined_output=torch.cat(output_to_cat, dim=-1)
        if hasattr(self, 'output_projection'):
            combined_output=self.output_projection(combined_output)
        if self.post_layernorm is not None:
            combined_output=self.post_layernorm(combined_output)
        
        assert combined_output.shape==reshaped_input.shape, 'shape1:{}, shape2:{}'.format(combined_output.shape, reshaped_input.shape)
        return combined_output, total_laux

    def scomoe_seq(self, reshaped_input, reshaped_padding):
        # utils.print_r0('reshaped_input:{}'.format(reshaped_input.abs().max()))
        num_token=reshaped_input.shape[0]
        local_K=int(self.ratio1*num_token)
        if self.ratio1+self.ratio2==1.0:
            node_K=num_token-local_K
        else:
            node_K=int(self.ratio2*num_token)
        global_K=num_token-local_K-node_K
        
        route_scores=self.gate.wg(reshaped_input) # (s, e)
        route_scores=route_scores.view(route_scores.shape[0], self.node_num, self.local_world_size, self.num_local_experts)

        masked_route_scores=route_scores.detach()
        empty_tensor=torch.tensor([])
        # local
        if local_K>0:
            local_route_scores=route_scores[:, self.node_rank, self.local_rank] # route scores on the experts of current device, [num_token, num_local_experts]
            local_route_scores=local_route_scores.mean(dim=-1) # [num_token,]
            _, local_tokens_indices=local_route_scores.topk(k=local_K, dim=0) # (local_K,)
            local_tokens=reshaped_input[local_tokens_indices]
            local_mask=reshaped_padding[local_tokens_indices]
            local_route_scores=route_scores[local_tokens_indices, self.node_rank, self.local_rank]
            masked_route_scores[local_tokens_indices]=float('-inf')
        else:
            local_tokens, local_route_scores=empty_tensor, empty_tensor
        # node
        if node_K>0:
            node_route_scores=masked_route_scores[:, self.node_rank].mean(dim=(-1,-2)) # (num_token,)
            _, node_tokens_indices=node_route_scores.topk(k=node_K, dim=0) # (node_K,)
            node_tokens=reshaped_input[node_tokens_indices]
            node_mask=reshaped_padding[node_tokens_indices]
            node_route_scores=route_scores[node_tokens_indices, self.node_rank].view(node_K, -1)
            masked_route_scores[node_tokens_indices]=float('-inf')
        else:
            node_tokens, node_route_scores=empty_tensor, empty_tensor
        # global
        if global_K>0:
            global_tokens_mask=~(masked_route_scores[:,0,0,0].isinf()) # (num_token, )
            global_tokens=reshaped_input[global_tokens_mask]
            global_mask=reshaped_padding[global_tokens_mask]
            global_route_scores=route_scores[global_tokens_mask].view(global_K, -1)
        else:
            global_tokens, global_route_scores=empty_tensor, empty_tensor
        
        node_group, global_group=self.node_group, self.all2all_group
        combined_output=torch.full_like(reshaped_input, fill_value=float('inf'))

        if not self.has_print_examples:
            utils.print_r0('>>>>>>>>>>>>>>>>>> hir_seq_moe >>>>>>>>>>>>>>>>>>>>>')
            utils.print_r0('local-g:{}, node-g:{}, global-g:{}'.format(
                self.num_local_experts, distributed_utils.get_world_size(self.node_group), distributed_utils.get_world_size(self.all2all_group), ))
            utils.print_r0('local-K:{}, node-K:{}, global-K:{}, all:{}'.format(local_K, node_K, global_K, num_token))
            utils.print_r0('local-r:{}, node-r:{}, global-r:{}, all:{}'.format(
                local_route_scores.shape, node_route_scores.shape, global_route_scores.shape, route_scores.shape))
            utils.print_r0('local-x:{}, node-x:{}, global-x:{}, all:{}'.format(
                local_tokens.shape, node_tokens.shape, global_tokens.shape, reshaped_input.shape))
            utils.print_r0('##########################################################')
            self.has_print_examples=True

        if local_K>0:
            local_outputs, l_aux1=self.local_moe(
                local_tokens, input_mask=local_mask, group=None, route_scores=local_route_scores)
            combined_output[local_tokens_indices]=local_outputs
            # utils.print_r0('local_inputs:{} local_outputs:{}'.format(local_tokens.abs().max(),local_outputs.abs().max()))
        else:
            l_aux1=0.

        if node_K>0:
            node_outputs, l_aux2=self.moe(
                node_tokens, input_mask=node_mask, group=node_group, route_scores=node_route_scores)
            combined_output[node_tokens_indices]=node_outputs
        else:
            l_aux2=0.
        
        if global_K>0:
            global_outputs, l_aux3=self.moe(global_tokens, input_mask=global_mask, group=global_group, route_scores=global_route_scores)
            combined_output[global_tokens_mask]=global_outputs
            # utils.print_r0('global_inputs:{} global_outputs:{}'.format(global_tokens.abs().max(), global_outputs.abs().max()))
        else:
            l_aux3=0.
            
        # utils.print_r0('combined_output:{}--------------------------------'.format(combined_output.abs().max()))

        l_aux1_weight=int(local_K!=0)
        l_aux2_weight=int(node_K!=0)
        l_aux3_weight=int(global_K!=0)
        total_laux=(l_aux1_weight*l_aux1+l_aux2_weight*l_aux2+l_aux3_weight*l_aux3)/(l_aux1_weight+l_aux2_weight+l_aux3_weight)
        return combined_output, total_laux

    def local_moe(self, tokens, input_mask, group=None, route_scores=None, experts=[]):
        if len(experts)==0:
            experts=self.experts
        assert route_scores.shape[-1]==len(experts)
        if len(experts)==1:
            return experts[0](tokens), 0.
        elif len(experts)<3:
            # tokens: (s, m), route_scores: (s, num_experts)
            route_scores=torch.softmax(route_scores, dim=-1)
            expert_outputs = []
            for expert in self.experts:
                expert_outputs += [expert(tokens)]
            expert_output = torch.stack(expert_outputs, dim=1) # (s, num_experts, m)
            return torch.sum(expert_output*route_scores.unsqueeze(dim=-1), dim=1), 0.
        else:
            return self.moe(tokens, input_mask, group=None, route_scores=None, experts=experts)

    def moe(self, tokens, input_mask, group, route_scores=None, experts=[]):
        _, d_model=tokens.shape
        if group is not None:
            world_size=distributed_utils.get_world_size(group)
        else:
            world_size=1
        # assert not node_mask.all(), 'x:{}, p:{}, p:{}, i:{},'.format(node_tokens.shape, node_mask.shape, global_mask.all(), node_tokens_indices)
        l_aux, combine_weights, dispatch_mask, self.metadata = self.gate(tokens, logits=route_scores, mask=input_mask)

        dispatch_mask = dispatch_mask.to(tokens.dtype).permute(1, 2, 0)  # S,E,C -> E,C,S
        E, C, S = dispatch_mask.size()
        assert tokens.size() == (S, d_model)
        # einsum("sec,sm->ecm")
        dispatched_input = torch.mm(dispatch_mask.view(E*C, S), tokens)  # -> (E*C),M

        if world_size!=1:
            dispatched_input = self.all_to_all_wrapper(dispatched_input, group)

        dispatched_input = dispatched_input.reshape(world_size, self.num_local_experts, -1, d_model)
        chunks = dispatched_input.chunk(self.num_local_experts, dim=1)
        expert_outputs = []

        if len(experts)==0:
            experts=self.experts
        for chunk, expert in zip(chunks, experts):
            expert_outputs += [expert(chunk)]
        expert_output = torch.cat(expert_outputs, dim=1)
        if self.post_layernorm is not None:
            expert_output=self.post_layernorm(expert_output)
        if world_size!=1:
            expert_output = self.all_to_all_wrapper(expert_output, group)

        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(world_size * self.num_local_experts, -1, d_model)

        # einsum("sec,ecm->sm")
        combined_output = combine_weights.view(S, E*C).mm(expert_output.view(E*C, d_model))
        assert tokens.shape==combined_output.shape
        assert not is_inf(combined_output) and not is_nan(combined_output)
        return combined_output, l_aux

    def all_to_all_wrapper(self, input: Tensor, group=None, input_splits=None, output_splits=None):
        dummy_a2a = getattr(self.args, 'dummy_a2a', False)
        if dummy_a2a:
            input = input.contiguous()
            output = input.detach().clone()
            return input
        # always record times, since it is not a lot of overhead
        # if we do not log it we simply clear it off in record_all_to_all_stats
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cpu_start = time.time() * 1000
        cuda_start.record()
        if group is None:
            group=self.all2all_group

        output = _AllToAll.apply(group, input, input_splits, output_splits)
        cuda_end.record()
        cpu_end = time.time() * 1000
        self.a2a_cpu_time_ms += (cpu_end - cpu_start)
        self.a2a_cuda_event_intervals.append((cuda_start, cuda_end))
        return output

    def token_cluering_moe(self, tokens, input_padding_mask, group=None, seq_len=None):
        dispatch_sort, soft_assignment, input_splits, output_splits, l_aux, self.metadata  = self.gate(tokens, input_padding_mask, seq_len)
        if self.gate_type=='softmax' and soft_assignment is not None:
            dispatched_input=torch.mm(soft_assignment.type_as(tokens), tokens)
        else:
            dispatched_input=tokens[dispatch_sort]
        
        dispatch_padding_mask=input_padding_mask[dispatch_sort]

        if len(self.experts)>1: 
            # mutiple experts in one device
            merged_input_splits=input_splits.view(self.all2all_size, -1).sum(dim=-1)
            merged_output_splits=output_splits.view(self.all2all_size, -1).sum(dim=-1)
            assert merged_input_splits.shape[0]==merged_output_splits.shape[0]==self.all2all_size
            merged_input_splits=merged_input_splits.tolist()
            merged_output_splits=merged_output_splits.tolist()
            dispatched_input = self.all_to_all_wrapper(
                dispatched_input, group, merged_input_splits, merged_output_splits) # num_token, d_m
            dispatch_padding_mask = self.all_to_all_wrapper(
                dispatch_padding_mask, group, merged_input_splits, merged_output_splits) # num_token, d_m          
        else:
            input_splits=input_splits.tolist()
            output_splits=output_splits.tolist()
            dispatched_input = self.all_to_all_wrapper(
                dispatched_input, group, input_splits, output_splits) # num_token, d_m
            dispatch_padding_mask = self.all_to_all_wrapper(
                dispatch_padding_mask, group, input_splits, output_splits) # num_token, d_m
        
        # collect compute inputs
        if len(self.experts)>1:
            # mutiple experts in one device
            output_splits_cumsum=[0]
            for x in output_splits:
                output_splits_cumsum.append(output_splits_cumsum[-1]+x)
            # output_splits_cumsum: [0, x0, x0+x1, x0+x1+x2, ...]
            indices, chunks=[],[]
            for i in range(len(output_splits_cumsum)-1):
                left_i=output_splits_cumsum[i]
                right_i=output_splits_cumsum[i+1]
                indices.append([left_i,right_i])
            # indices: [[0, x0], [x0, x0+x1],...]
            indices=torch.tensor(indices)
            indices=indices.view(self.all2all_size, len(self.experts), 2)
            # indices:
            #                2 experts
            #   [[[0,x0],          [x0, x0+x1            ]],   all2all_size=2
            #   [[x0+x1,x0+x1+x2],[x0+x1+x2, x0+x1+x2+x3]]], 
            indices=indices.transpose(0,1) # (num_local_experts, all2all_size)
            all_indices=[]
            for expert_indices in indices:
                # expert_indices: indices of one expert
                expert_indices = [list(range(i[0], i[1])) for i in expert_indices]
                expert_indices = sum(expert_indices, [])
                chunks.append(dispatched_input[expert_indices])
                all_indices.append(expert_indices)
            assert sorted(sum(all_indices, []))==list(range(len(dispatched_input))) # all features are processed by expert
            assert len(chunks)==len(self.experts)
            # compute
            result=torch.empty_like(dispatched_input)
            for i,chunk in enumerate(chunks):
                expert_indices=all_indices[i]
                expert_output=self.experts[i](chunk)+chunk
                result[expert_indices]=expert_output
                # expert_outputs.append(chunk)
        else:
            expert_output=self.experts[0](dispatched_input)
            # only one expert in a device
            if self.gate_type=='sigmoid':
                expert_indices=[distributed_utils.get_global_rank()]
                df_gate=torch.sigmoid(torch.mm(dispatched_input, self.gate.wg.weight[expert_indices].t())) # (num_tokens, 1)
                result=expert_output*df_gate+dispatched_input*(1-df_gate)
            else:
                if getattr(self.args, 'ffn_after_cluster', False):
                    result=expert_output # res connection already adds into expert
                else:
                    result=expert_output+dispatched_input

        i_sort=self.inverse_sort(dispatch_sort) # TODO: check error in dispatch_sort
        # assert torch.all(0<=dispatch_sort) and torch.all(dispatch_sort<len(dispatch_sort))
        # assert torch.all(0<=i_sort) and torch.all(i_sort<len(dispatch_sort))

        # result=result[i_sort]

        if len(self.experts)>1:
            return result, l_aux, i_sort, merged_input_splits, merged_output_splits, dispatch_padding_mask
        else:
            return result, l_aux, i_sort, input_splits, output_splits, dispatch_padding_mask

    def inverse_sort(self, order):
        # Creates an index that undoes a sort: xs==xs[order][inverse_sort(order)]
        return torch.empty_like(order).scatter_(0, order, torch.arange(0, order.size(0), device=order.device))

def get_all2all_group():
    if torch.distributed.is_initialized():
        if not hasattr(get_all2all_group, "_all2all_groups"):
            world_size=distributed_utils.get_global_world_size()
            all2all_groups=dist.new_group(list(range(world_size)))
            get_all2all_group._all2all_groups=all2all_groups
        return get_all2all_group._all2all_groups

def get_node_group():
    if torch.distributed.is_initialized():
        if not hasattr(get_node_group, "_node_groups"):
            world_size=distributed_utils.get_global_world_size()
            local_world_size=torch.cuda.device_count()
            assert local_world_size>1
            node_num=world_size//local_world_size
            global_rank=distributed_utils.get_global_rank()
            node_rank=global_rank//local_world_size
            for i in range(node_num):
                ranks=list(range(i*local_world_size, (i+1)*local_world_size))
                node_groups=dist.new_group(ranks)
                if i==node_rank:
                    get_node_group._node_groups=node_groups
        return get_node_group._node_groups

def share_mem(action='write', key:str=None, value:object=None, ):
    if not hasattr(share_mem, '_memory'):
        share_mem._memory=dict()    
    if action=='read':
        assert key in share_mem._memory
        return share_mem._memory[key]
    else:
        if isinstance(value, torch.Tensor):
            share_mem._memory[key]=value.detach()
        else:
            share_mem._memory[key]=value
