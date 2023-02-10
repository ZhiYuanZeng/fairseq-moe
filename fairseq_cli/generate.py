#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import ast

from sklearn import random_projection
from fairseq.modules.moe.moe_layer import MOELayer
from typing import Dict
from fairseq.distributed.utils import all_gather_list, all_reduce, get_global_rank, get_global_world_size
from fairseq.criterions.moe_cross_entropy import MoECrossEntropyCriterion
import logging
import math
import os
import sys
from argparse import Namespace
from itertools import chain

import numpy as np
import torch
from fairseq import checkpoint_utils, options, scoring, tasks, utils, distributed_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from omegaconf import DictConfig
from torch.nn.functional import pad
import random
try:
    from fairseq import libbase
except ImportError as e:
    sys.stderr.write(
        "ERROR: missing libbase. run `python setup.py build_ext --inplace`\n"
    )
    raise e


def maybe_set_suffix_with_rank(cfg):
    if cfg.common_eval.is_moe and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        cfg.checkpoint.checkpoint_suffix = f"-rank-{torch.distributed.get_rank()}"

def get_maybe_ddp_output_filename(cfg, filename):
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        suffix=f"-rank-{torch.distributed.get_rank()}"
        filename=filename.replace('.txt', suffix+'.txt')
    return filename

def maybe_merge_output_files(cfg, base_output_path):
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1 and get_global_rank()==0:
        with open(base_output_path, 'w', encoding="utf-8") as fo:
            for rank in range(cfg.distributed_training.distributed_world_size):
                suffix=f"-rank-{rank}"
                filename=base_output_path.replace('.txt', suffix+'.txt')
                with open(filename, 'r', encoding="utf-8") as fi:
                    for line in fi:
                        fo.write(line)

def main(cfg: DictConfig):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        base_output_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-{}.txt".format(cfg.dataset.gen_subset),
        )
        maybe_set_suffix_with_rank(cfg)
        output_path=get_maybe_ddp_output_filename(cfg, base_output_path)
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            _main(cfg, h)
        torch.distributed.barrier()
        maybe_merge_output_files(cfg, base_output_path)
    else:
        assert cfg.distributed_training.distributed_world_size==1
        return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(cfg: DictConfig, output_file):
    moe_logger=MoELogger()
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load dataset splits
    task = tasks.setup_task(cfg.task)


    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        cfg.checkpoint.checkpoint_suffix=""
        if cfg.common_eval.is_moe:
            cfg.checkpoint.checkpoint_suffix += f"-rank-{torch.distributed.get_rank()}"
            moe_freq = 1
        else:
            moe_freq = 0
        if cfg.distributed_training.ddp_backend=='fully_sharded':
            cfg.checkpoint.checkpoint_suffix += f"-shard{torch.distributed.get_rank()}"

    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
        is_moe=moe_freq > 0,
    )

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    if cfg.generation.lm_path is not None:
        overrides["data"] = cfg.task.data

        try:
            lms, _ = checkpoint_utils.load_model_ensemble(
                [cfg.generation.lm_path], arg_overrides=overrides, task=None
            )
        except:
            logger.warning(
                f"Failed to load language model! Please make sure that the language model dict is the same "
                f"as target dict and is located in the data dir ({cfg.task.data})"
            )
            raise

        assert len(lms) == 1
    else:
        lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    # Load dataset (possibly sharded)
    num_shards = cfg.distributed_training.distributed_world_size
    shard_id = cfg.distributed_training.distributed_rank
    # We need all GPUs to process different batch
    # if cfg.common_eval.is_moe:
    #     num_shards = 1
    #     shard_id = 0
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=num_shards,
        shard_id=shard_id,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()

    extra_gen_cls_kwargs = {"lm_model": lms[0], "lm_weight": cfg.generation.lm_weight}
    generator = task.build_generator(
        models, cfg.generation, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(cfg.tokenizer)
    bpe = task.build_bpe(cfg.bpe)

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    scorer = scoring.build_scorer(cfg.scoring, tgt_dict)

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()

    for sample in progress:     
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            # generate dummy batch
            dummy_sample={
                "id": torch.tensor([-1,]).cuda(),
                "nsentences": 1,
                "ntokens": 2,
                "net_input": {
                    "src_tokens": torch.zeros(
                        [1,2]).long().cuda(),
                    "src_lengths": torch.tensor([2, ]).cuda(),
                },
                "target": None,
            }
            sample=dummy_sample

        prefix_tokens = None
        if cfg.generation.prefix_size > 0:
            prefix_tokens = sample["target"][:, : cfg.generation.prefix_size]

        constraints = None
        if "constraints" in sample:
            constraints = sample["constraints"]
        # if not is_finish:
        gen_timer.start()
        with torch.no_grad():
            sample=dispatch_sample(sample, models[0], left_pad=cfg.task.left_pad_source, assignment_type=cfg.generation.assignment_type)
        hypos = task.inference_step(
            generator,
            models,
            sample,
            prefix_tokens=prefix_tokens,
            constraints=constraints,
        )
        moe_logger.log(models[0])
        # if is_finish or sample.get('dummy', False):
        #     continue
        num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
        gen_timer.stop(num_generated_tokens)
        for i, sample_id in enumerate(sample["id"].tolist()):
            if sample_id<0:
                continue
            has_target = sample["target"] is not None
            # Remove padding
            if "src_tokens" in sample["net_input"]:
                src_tokens = utils.strip_pad(
                    sample["net_input"]["src_tokens"][i, :], tgt_dict.pad()
                )
            else:
                src_tokens = None

            target_tokens = None
            if has_target:
                target_tokens = (
                    utils.strip_pad(sample["target"][i, :], tgt_dict.pad()).int().cpu()
                )

            # Either retrieve the original sentences or regenerate them from tokens.
            if align_dict is not None:
                src_str = task.dataset(cfg.dataset.gen_subset).src.get_original_text(
                    sample_id
                )
                target_str = task.dataset(cfg.dataset.gen_subset).tgt.get_original_text(
                    sample_id
                )
            else:
                if src_dict is not None:
                    src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)
                else:
                    src_str = ""
                if has_target:
                    target_str = tgt_dict.string(
                        target_tokens,
                        cfg.common_eval.post_process,
                        escape_unk=True,
                        extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                            generator
                        ),
                    )

            src_str = decode_fn(src_str)
            if has_target:
                target_str = decode_fn(target_str)

            if not cfg.common_eval.quiet:
                if src_dict is not None:
                    print("S-{}\t{}".format(sample_id, src_str), file=output_file)
                if has_target:
                    print("T-{}\t{}".format(sample_id, target_str), file=output_file)

            # Process top predictions
            for j, hypo in enumerate(hypos[i][: cfg.generation.nbest]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=cfg.common_eval.post_process,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )
                detok_hypo_str = decode_fn(hypo_str)
                if not cfg.common_eval.quiet:
                    score = hypo["score"] / math.log(2)  # convert to base 2
                    # original hypothesis (after tokenization and BPE)
                    print(
                        "H-{}\t{}\t{}".format(sample_id, score, hypo_str),
                        file=output_file,
                    )
                    # detokenized hypothesis
                    print(
                        "D-{}\t{}\t{}".format(sample_id, score, detok_hypo_str),
                        file=output_file,
                    )
                    print(
                        "P-{}\t{}".format(
                            sample_id,
                            " ".join(
                                map(
                                    lambda x: "{:.4f}".format(x),
                                    # convert from base e to base 2
                                    hypo["positional_scores"]
                                    .div_(math.log(2))
                                    .tolist(),
                                )
                            ),
                        ),
                        file=output_file,
                    )

                    if cfg.generation.print_alignment == "hard":
                        print(
                            "A-{}\t{}".format(
                                sample_id,
                                " ".join(
                                    [
                                        "{}-{}".format(src_idx, tgt_idx)
                                        for src_idx, tgt_idx in alignment
                                    ]
                                ),
                            ),
                            file=output_file,
                        )
                    if cfg.generation.print_alignment == "soft":
                        print(
                            "A-{}\t{}".format(
                                sample_id,
                                " ".join(
                                    [
                                        ",".join(src_probs)
                                        for src_probs in alignment
                                    ]
                                ),
                            ),
                            file=output_file,
                        )

                    if cfg.generation.print_step:
                        print(
                            "I-{}\t{}".format(sample_id, hypo["steps"]),
                            file=output_file,
                        )

                    if cfg.generation.retain_iter_history:
                        for step, h in enumerate(hypo["history"]):
                            _, h_str, _ = utils.post_process_prediction(
                                hypo_tokens=h["tokens"].int().cpu(),
                                src_str=src_str,
                                alignment=None,
                                align_dict=None,
                                tgt_dict=tgt_dict,
                                remove_bpe=None,
                            )
                            print(
                                "E-{}_{}\t{}".format(sample_id, step, h_str),
                                file=output_file,
                            )

                # Score only the top hypothesis
                if has_target and j == 0:
                    if align_dict is not None or cfg.common_eval.post_process is not None:
                        # Convert back to tokens for evaluation with unk replacement and/or without BPE
                        target_tokens = tgt_dict.encode_line(
                            target_str, add_if_not_exist=True
                        )
                        hypo_tokens = tgt_dict.encode_line(
                            detok_hypo_str, add_if_not_exist=True
                        )
                    if hasattr(scorer, "add_string"):
                        scorer.add_string(target_str, detok_hypo_str)
                    else:
                        scorer.add(target_tokens, hypo_tokens)

        wps_meter.update(num_generated_tokens)
        progress.log({"wps": round(wps_meter.avg)})
        num_sentences += (
            sample["nsentences"] if "nsentences" in sample else sample["id"].numel()
        )
    if gen_timer.sum==0:
        return None

    logger.info("NOTE: hypothesis and token scores are output in base 2")
    utils.print_r0(
        "Translated {:,} sentences ({:,} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)".format(
            num_sentences,
            gen_timer.n,
            gen_timer.sum,
            num_sentences / gen_timer.sum,
            gen_timer.n / gen_timer.sum,
        )
    )
    if has_target:
        if cfg.bpe and not cfg.generation.sacrebleu:
            if cfg.common_eval.post_process:
                logger.warning(
                    "BLEU score is being computed by splitting detokenized string on spaces, this is probably not what you want. Use --sacrebleu for standard 13a BLEU tokenization"
                )
            else:
                logger.warning(
                    "If you are using BPE on the target side, the BLEU score is computed on BPE tokens, not on proper words.  Use --sacrebleu for standard 13a BLEU tokenization"
                )
        # use print to be consistent with other main outputs: S-, H-, T-, D- and so on
        print(
            "Generate {} with beam={}: {}".format(
                cfg.dataset.gen_subset, cfg.generation.beam, scorer.result_string()
            ),
            file=output_file,
        )
    torch.distributed.barrier()
    moe_logger.print()
    return scorer

class MoELogger():
    moe_logging_keys = [
        "overflow_expert1",        # average % of overflowed tokens from 1st expert
        "overflow_expert2",        # average % of overflowed tokens from 2nd expert
        "entropy_gating",          # average entropy of the gating distribution
        # "expert1_balance_top",     # average cumulative % of tokens processed by the most used 20% 1st experts
        # "expert1_balance_bottom",  # average cumulative % of tokens processed by the least used 20% 1st experts
        "unused_expert1_count",    # average number of 1st experts which process no tokens
        # "expert2_balance_top",     # average cumulative % of tokens processed by the most used 20% 2nd experts
        # "expert2_balance_bottom",  # average cumulative % of tokens processed by the least used 20% 2nd experts
        "unused_expert2_count",    # average number of 2nd experts which process no tokens
        # "all_to_all_cpu_time_ms",  # CPU time spent in all to all calls in milliseconds
        # "all_to_all_cuda_time_ms", # CUDA ttime spent in all to all calls in milliseconds
        "unrouted_token_rate",
        # "sparsity",
    ]

    def __init__(self) -> None:
        self.moe_logging_output=dict()
        self.log_count=0

    def log(self, model):
        for key in MoELogger.moe_logging_keys:
            total_val = 0
            count = 0
            for _, module in model.named_modules():
                if isinstance(module, MOELayer):
                    total_val += module.metadata[key] if key in module.metadata else 0
                    count += 1
            self.moe_logging_output[key] = (self.moe_logging_output.get(key, 0)*self.log_count + total_val / count)/(self.log_count+1)
        self.log_count+=1

    def print(self,):
        torch.distributed.barrier()
        for k,v in self.moe_logging_output.items():
            if not isinstance(v, torch.Tensor):
                v=torch.tensor(v).cuda()
            else:
                v=v.clone()
            reduced_v=all_reduce(v, group=None, op='sum')/get_global_world_size()
            utils.print_r0(k+':'+str(reduced_v.item()))

def dispatch_sample(sample, model, left_pad, assignment_type):
    if assignment_type=="none":
        return sample
    torch.distributed.barrier()
    device=sample['net_input']['src_lengths'].device
    # print('before dispatch, rank:{}, lens:{}-{}=>{}'.format(get_global_rank(), 
    #     sample['net_input']['src_lengths'].max(), sample['net_input']['src_lengths'].min(), sample['net_input']['src_lengths'].max() - sample['net_input']['src_lengths'].min()))

    num_experts=model.encoder.layers[1].moe_layer.gate.wg.weight.shape[0]
    all_samples=all_gather_list(sample, group=None, max_size=2097152)
    all_lengths=torch.cat([sample['net_input']['src_lengths'] for sample in all_samples], dim=0).to(device)
    all_ids=torch.cat([sample['id'] for sample in all_samples], dim=0).to(device)
    max_length=all_lengths.max()
    
    if left_pad:
        all_tokens=torch.cat(
            [pad(sample['net_input']['src_tokens'], (max_length-sample['net_input']['src_tokens'].shape[1], 0), value=1) 
                for sample in all_samples], dim=0).to(device)
    else:
        all_tokens=torch.cat(
            [pad(sample['net_input']['src_tokens'], (0, max_length-sample['net_input']['src_tokens'].shape[1]), value=1) 
                for sample in all_samples], dim=0).to(device)

    all_word_embed=model.encoder.embed_tokens(all_tokens)
    mask=torch.arange(max_length, device=device)
    if left_pad:
        mask=mask.max()-mask # reverse
    mask=mask.expand_as(all_tokens)<all_lengths.unsqueeze(dim=-1)
    all_word_embed=torch.masked_fill(all_word_embed, ~mask.unsqueeze(dim=-1).expand_as(all_word_embed), 0.)
    
    all_sent_embeds=all_word_embed.sum(dim=1)/all_lengths.unsqueeze(dim=-1)

    batch_size=all_tokens.shape[0]
    if batch_size%(num_experts)!=0:
        pad_num=num_experts-batch_size%num_experts
    else:
        pad_num=0
    pad_mask=torch.full((batch_size+pad_num, ), fill_value=True, device=device)
    if pad_num!=0:
        pad_mask[-pad_num:]=False
    padded_bsz=batch_size+pad_num

    d_model=all_sent_embeds.shape[-1]
    assert (batch_size+pad_num)%num_experts==0
    

    if assignment_type.startswith('diverse'):
        if assignment_type=='diverse_random':
            cluster_num=(batch_size+pad_num)//num_experts
            random_projection=torch.randn(cluster_num, d_model, device=device)
            torch.nn.init.orthogonal_(random_projection, gain=0.1)
            logits=all_sent_embeds @ random_projection.t()
        elif assignment_type=='diverse_balance':
            cluster_num=num_experts
            logits=model.encoder.layers[1].moe_layer.gate.wg(all_sent_embeds)
        
        if pad_num!=0:
            logits=pad(logits, pad=(0, 0, 0, pad_num), value=logits.min().item())
        
        assert cluster_num==logits.shape[1]
        assert logits.shape[0]%logits.shape[1]==0
        assignment=libbase.balanced_assignment(logits.detach(), False)
        assert assignment.sum()==len(assignment)*(len(assignment)-1)/2
        # utils.print_r0(assignment.reshape(cluster_num, -1).tolist())
        assignment=assignment.reshape(cluster_num, -1).t()
        assignment=assignment.reshape(num_experts, -1)
        # if assignment.shape[0]%num_experts!=0:
        #     num_rows_in_each_process=torch.full(size=(num_experts,), fill_value=assignment.shape[0]//num_experts, device=device)
        #     num_rows_in_each_process[:(assignment.shape[0]%num_experts)]+=1
        #     utils.print_r0(num_rows_in_each_process, assignment.shape[0])
        #     assert num_rows_in_each_process.sum()==assignment.shape[0]
        #     cum_sum_rows_in_each_process=torch.cumsum(num_rows_in_each_process, dim=-1)
        #     start_indices=[0,]+cum_sum_rows_in_each_process[:-1].tolist()
        #     end_indices=cum_sum_rows_in_each_process.tolist()
        #     utils.print_r0(start_indices)
        #     utils.print_r0(end_indices)
        #     assignment=[assignment[s:e] for s,e in zip(start_indices, end_indices)]
        # else:
    elif assignment_type=='random':
        random_indices=list(range(padded_bsz))
        random.shuffle(random_indices)
        assignment=torch.tensor(random_indices, device=device).view(num_experts, -1)
    elif assignment_type=='original':
        assignment=torch.arange(padded_bsz, device=device).view(num_experts, -1)

    indices_for_cur_rank=assignment[get_global_rank()]
    mask_for_cur_rank=pad_mask[indices_for_cur_rank]
    indices_for_cur_rank=indices_for_cur_rank[mask_for_cur_rank]

    samples_num=all_reduce(torch.tensor(indices_for_cur_rank.shape[0]).cuda(), group=None, op='sum')
    assert samples_num.item()==all_ids.shape[0], '{} {}'.format(samples_num, all_ids.shape[0])

    if len(indices_for_cur_rank)==0:
        dummy_sample={
            "id": torch.tensor([-1,]).cuda(),
            "nsentences": 1,
            "ntokens": 2,
            "net_input": {
                "src_tokens": torch.zeros(
                    [1,2]).long().cuda(),
                "src_lengths": torch.tensor([2, ]).cuda(),
            },
            "target": None,
        }
        return dummy_sample
    
    src_tokens=all_tokens[indices_for_cur_rank]
    src_lengths=all_lengths[indices_for_cur_rank]
    ids=all_ids[indices_for_cur_rank]
    # print('rank:{}, indices:{}'.format(get_global_rank(), indices_for_cur_rank))
    if left_pad:
        src_tokens=src_tokens[:, -src_lengths.max():]
    else:
        src_tokens=src_tokens[:, :src_lengths.max()]

    nsentences=src_tokens.shape[0]

    new_sample=dict()
    new_sample['net_input']=dict()
    new_sample['net_input']['src_tokens']=src_tokens
    new_sample['net_input']['src_lengths']=src_lengths
    new_sample['id']=ids
    new_sample['nsentences']=nsentences
    new_sample['target']=None

    # print('after dispatch, rank:{}, lens:{}-{}=>{}'.format(get_global_rank(), src_lengths.max(), src_lengths.min(), src_lengths.max() - src_lengths.min(), ))

    # print('org:{}, new:{}'.format(org_sample['id'].shape, new_sample['id'].shape))
    return new_sample

def cli_main():
    parser = options.get_generation_parser()
    # parser.add_argument('--assignment-type', type=str, choices=['none', 'random', 'original', 'diverse_balance', 'diverse_random'], default='none')
    args = options.parse_args_and_arch(parser)
    cfg = convert_namespace_to_omegaconf(args)
    distributed_utils.call_main(cfg, main)

if __name__ == "__main__":
    cli_main()
