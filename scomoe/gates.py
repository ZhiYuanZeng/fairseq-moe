from cmath import isinf
from scomoe.utils import is_inf, is_nan
from fairseq.utils import print_r0
from fairseq.modules.moe.top2gate import entropy, one_hot, top2gating
import torch
from typing import Tuple, Dict
from torch import Tensor
from fairseq.modules.moe.moe_layer import _AllToAll
import torch.nn.functional as F
import fairseq.distributed.utils as distributed_utils

class BalanceGate(torch.nn.Module):
    def __init__(self, args, model_dim, num_experts, **kwargs) -> None:
        super().__init__()
        if args.ratio1==0. and args.ratio2!=1.0 and args.scomoe_type!='none':
            self.route_on_nodes=True
            world_size=distributed_utils.get_global_world_size()
            local_world_size=torch.cuda.device_count()
            assert local_world_size>1
            self.node_num=world_size//local_world_size
        else:
            self.route_on_nodes=False
        self.gate_type=getattr(args, 'gate_type', 'softmax')
        self.temperature=getattr(args, 'temperature', 1.0)
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        torch.nn.init.orthogonal_(self.wg.weight, gain=0.1)
        self.cpp=self._load_assignment()
        self.num_experts=num_experts
        self.autoregressive=False
        self.greedy_autoregressive_inference=getattr(args, 'greedy_autoregressive_inference', False)

    def forward(self, inputs, pad_mask=None, seq_len=0):
        if self.training and self.autoregressive:
            # avoid information broadcast between target tokens
            token_expert_sim=self.wg(inputs.detach())
        else:
            token_expert_sim=self.wg(inputs)

        num_experts=token_expert_sim.shape[-1]
        
        pad_num=pad_mask.sum().item() # true is pad
        if pad_num!=0:
            max_v, min_v=token_expert_sim.max(), token_expert_sim.min()
            random_v=torch.rand([pad_num, num_experts], dtype=inputs.dtype, device=inputs.device) # [0,1)
            random_v = random_v * (min_v-max_v) * 0.1 + min_v # min_v + (min_v-max_v~0)*0.1
            assert torch.all(random_v<=min_v)
            token_expert_sim=torch.masked_scatter(token_expert_sim, pad_mask.view(-1,1), random_v)
            assert torch.all(token_expert_sim[pad_mask]<=min_v)

        reshaped_token_expert_sim=token_expert_sim # default
        gate_func=self.balanced_assignment # default    
        if not self.training and self.autoregressive:
            # autoregressive inference
            if self.greedy_autoregressive_inference:
                gate_func=self.greedy_gating
            else:
                reshaped_token_expert_sim=token_expert_sim.view(-1, seq_len, num_experts)
        
        sort_by_expert, soft_assigment, input_splits, output_splits, l_aux = gate_func(reshaped_token_expert_sim)
        return sort_by_expert, soft_assigment, input_splits, output_splits, l_aux, {}

    def projection_from_sort(self, score, topk_index):
        E, T=score.shape # expert_num, token_num
        topk_index=topk_index.reshape(-1, 1)
        hard_projection=torch.zeros((T,T), device=score.device)
        hard_projection=torch.scatter(
            input=hard_projection, 
            dim=-1 ,
            index=topk_index,
            src=torch.ones_like(topk_index).float())
        """
        scatter:
        input   index  src
        [[0,0]  [[0]   [[1]  => [[1,0]
        [0,0]]  [1]]   [1]] =>  [0,1]]
        """
        # hard projection is a row projection matrix, each row is a projection/selection
        soft_projection=torch.softmax(score/self.temperature, dim=1) # E,T

        soft_projection=soft_projection.unsqueeze(1).expand(E,T//E,T)/(T//E)
        soft_projection=soft_projection.reshape(T,T) # every row is a probability
        mix_projection=hard_projection+soft_projection-soft_projection.detach() # ST trick
        return mix_projection

    def _balanced_assignment(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        if self.route_on_nodes:
            scores=scores.view(scores.shape[0], self.node_num, -1).mean(dim=-1)
        num_tokens, num_group=scores.shape
        assert num_tokens%num_group==0
        assert num_tokens>=num_group
        assert not is_inf(scores) and not is_nan(scores)
        top_index=self.cpp.balanced_assignment(scores.detach(), False)

        if len(set(top_index.tolist()))!=len(top_index):
            top_index=self.cpp.balanced_assignment(scores.detach(), True)
            print_r0('>>>>>>>>>>> sort is not ok, sort again with strict constraints! <<<<<<<<<<<')
            assert len(set(top_index.tolist()))==len(top_index)

        num_tokens_per_experts=num_tokens//self.num_experts
        input_splits=torch.full(size=(self.num_experts,), fill_value=num_tokens_per_experts, device=scores.device)

        # soft assigment1: expert select tokens
        if self.gate_type=='softmax':
            df_projection=self.projection_from_sort(scores.t(), top_index)
        else:
            df_projection=None
        return top_index, df_projection, input_splits, 0.

    def balanced_assignment(self, scores):
        # (bsz, seq, experts)
        if len(scores.shape)==2:
            top_index, soft_assigment, input_splits, l_aux = self._balanced_assignment(scores)
        else:
            bsz, seq_len, num_experts=scores.shape

            all_input_splits=0
            base_indices=torch.arange(bsz*seq_len, device=scores.device).view(bsz, seq_len)
            l_aux, soft_assigment=0, None
            for i in range(seq_len):
                top_index, _, input_splits, _=self._balanced_assignment(scores[:,i])
                # top_index: (bsz,), input_splits: (num_experts)
                base_indices[:, i]=base_indices[:, i][top_index] # sort the column (bsz) indices
                all_input_splits+=input_splits
            
            top_index=base_indices.view(-1)
            input_splits=all_input_splits
            assert len(input_splits)==num_experts
            assert sum(input_splits)==bsz*seq_len, f'input_splits is not correct, sum (input_splits):{sum(input_splits)}, {bsz*seq_len}'
        output_splits = _AllToAll.apply(None, input_splits)
        
        return top_index, soft_assigment, input_splits, output_splits, l_aux

    def _load_assignment(self):
        try:
            from fairseq import libbase

            return libbase

        except ImportError as e:
            sys.stderr.write(
                "ERROR: missing libbase. run `python setup.py build_ext --inplace`\n"
            )
            raise e

    def greedy_gating(
        self,
        logits: torch.Tensor,
    ):
        """
        TODO: support
        - topk gateing (combine weights)
        - mask
        """
        num_experts = logits.shape[1]

        token_to_workers = torch.argmax(logits, dim=-1)  # worker ids

        # routing
        token_to_workers, sort_ordering = torch.sort(token_to_workers)
        # Find how many tokens we're sending to each other worker (being careful for sending 0 tokens to some workers)
        input_splits = torch.zeros((num_experts,), dtype=torch.long, device=logits.device)
        workers, counts = torch.unique_consecutive(token_to_workers, return_counts=True)
        input_splits[workers] = counts
        # Tell other workers how many tokens to expect from us
        assert sum(input_splits) == token_to_workers.shape[0], f'{str(input_splits)} {token_to_workers.shape[0]}'
        output_splits = _AllToAll.apply(None, input_splits)

        return sort_ordering, None, input_splits, output_splits, 0.
