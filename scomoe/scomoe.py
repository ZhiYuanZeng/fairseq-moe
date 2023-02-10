import torch
from fairseq.models.transformer import TransformerModel
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq import distributed_utils, utils
from scomoe.scomoe_encoder import ScomoeTransformerEncoder
from scomoe.scomoe_decoder import ScomoeTransformerDecoder

@register_model("scomoe")
class ScomoeTransformer(TransformerModel):
    @classmethod
    def build_model(cls, args, task):
        world_size=distributed_utils.get_global_world_size()
        local_world_size=torch.cuda.device_count()
        assert world_size%local_world_size==0, f'local_world_size:{local_world_size}, world_size:{world_size}'
        node_num=world_size//local_world_size
        if node_num==1:
            args.ratio2=1-args.ratio1 # which means only ratio1 should be set, when training on 1 node
        args.ratio3=1-args.ratio1-args.ratio2
        utils.print_r0('##########################################################')
        utils.print_r0('world_size:{}, node_num:{}'.format(world_size, node_num))
        utils.print_r0('r1:{} r2:{} r3:{}'.format(args.ratio1, args.ratio2, args.ratio3))
        utils.print_r0('##########################################################')
        return super().build_model(args, task)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return ScomoeTransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return ScomoeTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--ratio1', type=float, default=0., help="ratio for intra-accelerator communication")
        parser.add_argument('--ratio2', type=float, default=0., help="ratio for intra-node communication")
        parser.add_argument('--temperature', type=float, default=1.0, help="temperature for df-gate, only make sense if df-gate-type=sigmoid")
        parser.add_argument('--token-cluster', action='store_true', default=False, help="")
        parser.add_argument('--reorder-layers', action='store_true', default=False, help="")
        parser.add_argument('--scomoe-type', type=str, default='seq', 
            choices=['feat','seq', 'none'], help="")
        parser.add_argument('--gate-type', type=str, default='softmax',
            choices=['softmax','sigmoid',])
        parser.add_argument('--layernorm-after-moe-layer', action='store_true', default=False)
        TransformerModel.add_args(parser)        

@register_model_architecture('scomoe', 'scomoe')
def scomoe(args):
    pass