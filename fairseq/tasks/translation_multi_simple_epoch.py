# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import logging
import time
from typing import Dict, List

import torch
from fairseq.data import (
    FairseqDataset,
    LanguagePairDataset,
    ListDataset,
    data_utils,
    iterators,
)
from fairseq.data.multilingual.multilingual_data_manager import (
    MultilingualDatasetManager,
)
from fairseq.data.multilingual.sampling_method import SamplingMethod
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.utils import FileContentsAction, print_r0


###
def get_time_gap(s, e):
    return (
        datetime.datetime.fromtimestamp(e) - datetime.datetime.fromtimestamp(s)
    ).__str__()


###


logger = logging.getLogger(__name__)


@register_task("translation_multi_simple_epoch")
class TranslationMultiSimpleEpochTask(LegacyFairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        langs (List[str]): a list of languages that are being supported
        dicts (Dict[str, fairseq.data.Dictionary]): mapping from supported languages to their dictionaries
        training (bool): whether the task should be configured for training or not

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='inference source language')
        parser.add_argument('--src-max-len', default=None, type=int,
                            help='max lenth of source sentences')
        parser.add_argument('--tgt-max-len', default=None, type=int,
                            help='max lenth of target sentences')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='inference target language')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr',
                            action=FileContentsAction)
        parser.add_argument('--keep-inference-langtok', action='store_true',
                            help='keep language tokens in inference output (e.g. for analysis or debugging)')

        SamplingMethod.add_arguments(parser)
        MultilingualDatasetManager.add_args(parser)
        # fmt: on

    def __init__(self, args, langs, dicts, training):
        super().__init__(args)
        self.langs = langs
        self.dicts = dicts
        self.training = training
        if training:
            self.lang_pairs = args.lang_pairs
        else:
            self.lang_pairs = ["{}-{}".format(args.source_lang, args.target_lang)]
        # eval_lang_pairs for multilingual translation is usually all of the
        # lang_pairs. However for other multitask settings or when we want to
        # optimize for certain languages we want to use a different subset. Thus
        # the eval_lang_pairs class variable is provided for classes that extend
        # this class.
        self.eval_lang_pairs = self.lang_pairs
        # model_lang_pairs will be used to build encoder-decoder model pairs in
        # models.build_model(). This allows multitask type of sub-class can
        # build models other than the input lang_pairs
        self.model_lang_pairs = self.lang_pairs
        self.source_langs = [d.split("-")[0] for d in self.lang_pairs]
        self.target_langs = [d.split("-")[1] for d in self.lang_pairs]
        self.check_dicts(self.dicts, self.source_langs, self.target_langs)

        self.sampling_method = SamplingMethod.build_sampler(args, self)
        self.data_manager = MultilingualDatasetManager.setup_data_manager(
            args, self.lang_pairs, langs, dicts, self.sampling_method
        )
        self.src_max_len=args.src_max_len
        self.tgt_max_len=args.tgt_max_len

    def check_dicts(self, dicts, source_langs, target_langs):
        if self.args.source_dict is not None or self.args.target_dict is not None:
            # no need to check whether the source side and target side are sharing dictionaries
            return
        src_dict = dicts[source_langs[0]]
        tgt_dict = dicts[target_langs[0]]
        for src_lang in source_langs:
            assert (
                src_dict == dicts[src_lang]
            ), "Diffrent dictionary are specified for different source languages; "
            "TranslationMultiSimpleEpochTask only supports one shared dictionary across all source languages"
        for tgt_lang in target_langs:
            assert (
                tgt_dict == dicts[tgt_lang]
            ), "Diffrent dictionary are specified for different target languages; "
            "TranslationMultiSimpleEpochTask only supports one shared dictionary across all target languages"

    @classmethod
    def setup_task(cls, args, **kwargs):
        langs, dicts, training = MultilingualDatasetManager.prepare(
           cls.load_dictionary, args, **kwargs
        )
        return cls(args, langs, dicts, training)

    def has_sharded_data(self, split):
        return self.data_manager.has_sharded_data(split)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        shard_epoch=None
        if split in self.datasets:
            dataset = self.datasets[split]
            if self.has_sharded_data(split):
                if self.args.virtual_epoch_size is not None:
                    if dataset.load_next_shard:
                        shard_epoch = dataset.shard_epoch
                    else:
                        # no need to load next shard so skip loading
                        # also this avoid always loading from beginning of the data
                        return
                else:
                    shard_epoch = epoch
        else:
            # estimate the shard epoch from virtual data size and virtual epoch size
            shard_epoch = self.data_manager.estimate_global_pass_epoch(epoch)
        logger.info(f"loading data for {split} epoch={epoch}/{shard_epoch}")
        logger.info(f"mem usage: {data_utils.get_mem_usage()}")
        if split in self.datasets:
            del self.datasets[split]
            logger.info("old dataset deleted manually")
            logger.info(f"mem usage: {data_utils.get_mem_usage()}")
        self.datasets[split] = self.data_manager.load_dataset(
            split,
            self.training,
            epoch=epoch,
            combine=combine,
            shard_epoch=shard_epoch,
            **kwargs,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None, eval_src_lang=None, eval_tgt_lang=None):
        if constraints is not None:
            raise NotImplementedError(
                "Constrained decoding with the multilingual_translation task is not supported"
            )

        src_data = ListDataset(src_tokens, src_lengths)
        dataset = LanguagePairDataset(src_data, src_lengths, self.source_dictionary)
        src_langtok_spec, tgt_langtok_spec = self.args.langtoks["main"]
        if eval_src_lang is None:
            eval_src_lang=self.args.source_lang
        if eval_tgt_lang is None:
            eval_tgt_lang=self.args.target_lang

        if self.args.lang_tok_replacing_bos_eos:
            dataset = self.data_manager.alter_dataset_langtok(
                dataset,
                src_eos=self.source_dictionary.eos(),
                src_lang=eval_src_lang,
                tgt_eos=self.target_dictionary.eos(),
                tgt_lang=eval_tgt_lang,
                src_langtok_spec=src_langtok_spec,
                tgt_langtok_spec=tgt_langtok_spec,
            )
        else:
            dataset.src = self.data_manager.src_dataset_tranform_func(
                eval_src_lang,
                eval_tgt_lang,
                dataset=dataset.src,
                spec=src_langtok_spec,
            )
        return dataset

    def build_multilingual_dataset_for_inference(self, multilingual_src_tokens:List[Dict], constraints=None):
        if constraints is not None:
            raise NotImplementedError(
                "Constrained decoding with the multilingual_translation task is not supported"
            )
        datasets=[]
        for item in multilingual_src_tokens:
            eval_src_lang=item['src_lang']
            eval_tgt_lang=item['tgt_lang']
            src_tokens=item['src_tokens']
            src_lengths=item['src_lengths']

            src_langtok_spec, tgt_langtok_spec = self.args.langtoks["main"]
            eval_src_lang_id=self.data_manager.get_encoder_langtok(eval_src_lang, eval_tgt_lang, spec='src')
            eval_tgt_lang_id=self.data_manager.get_decoder_langtok(eval_tgt_lang, spec='tgt')
            assert eval_src_lang_id is not None, eval_src_lang+','+src_langtok_spec
            src_data = ListDataset(src_tokens, src_lengths)
            dataset = LanguagePairDataset(
                src=src_data, 
                src_sizes=src_lengths, 
                tgt=src_data, 
                tgt_sizes=src_lengths, 
                src_dict=self.source_dictionary,
                tgt_dict=self.source_dictionary,
                src_lang_id=eval_src_lang_id,
                tgt_lang_id=eval_tgt_lang_id,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target
            )
            if eval_src_lang is None:
                eval_src_lang=self.args.source_lang
            if eval_tgt_lang is None:
                eval_tgt_lang=self.args.target_lang
            if self.args.lang_tok_replacing_bos_eos:
                dataset = self.data_manager.alter_dataset_langtok(
                    dataset,
                    src_eos=self.source_dictionary.eos(),
                    src_lang=eval_src_lang,
                    tgt_eos=self.target_dictionary.eos(),
                    tgt_lang=eval_tgt_lang,
                    src_langtok_spec=src_langtok_spec,
                    tgt_langtok_spec=tgt_langtok_spec,
                )
            else:
                dataset.src = self.data_manager.src_dataset_tranform_func(
                    eval_src_lang,
                    eval_tgt_lang,
                    dataset=dataset.src,
                    spec=src_langtok_spec,
                )
            datasets.append((eval_src_lang+'-'+eval_tgt_lang,dataset))
        # for _,d in datasets:
        #     print_r0(len(d))
        datasets=self.data_manager.load_into_concat_dataset(split='infer', datasets=datasets, data_param_list=None)
        return datasets

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        if not getattr(args, "keep_inference_langtok", False):
            _, tgt_langtok_spec = self.args.langtoks["main"]
            if tgt_langtok_spec:
                tgt_lang_tok = self.data_manager.get_decoder_langtok(
                    self.args.target_lang, tgt_langtok_spec
                )
                extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
                extra_gen_cls_kwargs["symbols_to_strip_from_output"] = {tgt_lang_tok}

        return super().build_generator(
            models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def build_model(self, args):
        return super().build_model(args)

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        return loss, sample_size, logging_output

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None, eval_tgt_lang=None,
    ):
        src_tokens=sample["net_input"]["src_tokens"]
        bsz=src_tokens.size(0)

        if 'tgt_lang_id' in sample:
            tgt_lang_toks=sample["tgt_lang_id"].squeeze(dim=-1)
        else:
            if eval_tgt_lang is None:
                eval_tgt_lang=self.args.target_lang
            eval_tgt_langs=[eval_tgt_lang for _ in range(bsz)]
            _, tgt_langtok_spec = self.args.langtoks["main"]
            tgt_lang_toks = [self.data_manager.get_decoder_langtok(lang, tgt_langtok_spec) for lang in eval_tgt_langs]
            tgt_lang_toks=torch.LongTensor(tgt_lang_toks).to(src_tokens)
        with torch.no_grad():
            if not self.args.lang_tok_replacing_bos_eos:
                if prefix_tokens is None and tgt_langtok_spec:
                    src_tokens = src_tokens
                    prefix_tokens = tgt_lang_toks.unsqueeze(dim=-1)
                return generator.generate(
                    models,
                    sample,
                    prefix_tokens=prefix_tokens,
                    constraints=constraints,
                )
            else:
                return generator.generate(
                    models,
                    sample,
                    prefix_tokens=prefix_tokens,
                    bos_token=tgt_lang_toks
                )

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        return self.data_manager.get_source_dictionary(self.source_langs[0])

    @property
    def target_dictionary(self):
        return self.data_manager.get_target_dictionary(self.target_langs[0])

    def create_batch_sampler_func(
        self,
        max_positions,
        ignore_invalid_inputs,
        max_tokens,
        max_sentences,
        required_batch_size_multiple=1,
        seed=1,
    ):
        def construct_batch_sampler(dataset, epoch):
            splits = [
                s for s, _ in self.datasets.items() if self.datasets[s] == dataset
            ]
            split = splits[0] if len(splits) > 0 else None
            # NEW implementation
            if epoch is not None:
                # initialize the dataset with the correct starting epoch
                dataset.set_epoch(epoch)

            # get indices ordered by example size
            start_time = time.time()
            logger.info(f"start batch sampler: mem usage: {data_utils.get_mem_usage()}")

            with data_utils.numpy_seed(seed):
                indices = dataset.ordered_indices()
            logger.info(
                f"[{split}] @batch_sampler order indices time: {get_time_gap(start_time, time.time())}"
            )
            logger.info(f"mem usage: {data_utils.get_mem_usage()}")

            # filter examples that are too large
            if max_positions is not None:
                src_max_len, tgt_max_len=max_positions
                if self.src_max_len is not None and self.tgt_max_len is not None:
                    src_max_len=min(src_max_len, self.src_max_len)
                    tgt_max_len=min(tgt_max_len, self.tgt_max_len)

                my_time = time.time()
                indices = self.filter_indices_by_size(
                    indices, dataset, (src_max_len, tgt_max_len), ignore_invalid_inputs
                )
                logger.info(
                    f"[{split}] @batch_sampler filter_by_size time: {get_time_gap(my_time, time.time())}"
                )
                logger.info(f"mem usage: {data_utils.get_mem_usage()}")

            # create mini-batches with given size constraints
            my_time = time.time()
            batch_sampler = dataset.batch_by_size(
                indices,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )

            logger.info(
                f"[{split}] @batch_sampler batch_by_size time: {get_time_gap(my_time, time.time())}"
            )
            logger.info(
                f"[{split}] per epoch batch_sampler set-up time: {get_time_gap(start_time, time.time())}"
            )
            logger.info(f"mem usage: {data_utils.get_mem_usage()}")

            return batch_sampler

        return construct_batch_sampler

    # we need to override get_batch_iterator because we want to reset the epoch iterator each time
    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 0).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        # initialize the dataset with the correct starting epoch
        assert isinstance(dataset, FairseqDataset)
        if dataset in self.dataset_to_epoch_iter:
            return self.dataset_to_epoch_iter[dataset]
        if self.args.sampling_method == "RoundRobin":
            batch_iter = super().get_batch_iterator(
                dataset,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                max_positions=max_positions,
                ignore_invalid_inputs=ignore_invalid_inputs,
                required_batch_size_multiple=required_batch_size_multiple,
                seed=seed,
                num_shards=num_shards,
                shard_id=shard_id,
                num_workers=num_workers,
                epoch=epoch,
                data_buffer_size=data_buffer_size,
                disable_iterator_cache=disable_iterator_cache,
            )
            self.dataset_to_epoch_iter[dataset] = batch_iter
            return batch_iter

        construct_batch_sampler = self.create_batch_sampler_func(
            max_positions,
            ignore_invalid_inputs,
            max_tokens,
            max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
            seed=seed,
        )

        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=construct_batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )
        return epoch_iter
