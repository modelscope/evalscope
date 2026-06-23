# flake8: noqa
import numpy as np
import os
import pandas as pd
import time
import torch
import warnings
from collections import defaultdict

from .utils import (
    bert_cos_score_idf,
    get_bert_embedding,
    get_hash,
    get_idf_dict,
    get_model,
    get_tokenizer,
    lang2model,
    model2layers,
    sent_encode,
)


class BERTScorer:
    """
    BERTScore Scorer Object.
    """

    def __init__(
        self,
        model_id_or_path=None,
        model_type=None,
        num_layers=None,
        batch_size=64,
        nthreads=4,
        all_layers=False,
        idf=False,
        idf_sents=None,
        device=None,
        lang=None,
        rescale_with_baseline=False,
        baseline_path=None,
        use_fast_tokenizer=False,
    ):
        """
        Args:
            - :param: `model_type` (str): contexual embedding model specification, default using the suggested
                      model for the target langauge; has to specify at least one of
                      `model_type` or `lang`
            - :param: `num_layers` (int): the layer of representation to use.
                      default using the number of layer tuned on WMT16 correlation data
            - :param: `verbose` (bool): turn on intermediate status update
            - :param: `idf` (bool): a booling to specify whether to use idf or not (this should be True even if `idf_sents` is given)
            - :param: `idf_sents` (List of str): list of sentences used to compute the idf weights
            - :param: `device` (str): on which the contextual embedding model will be allocated on.
                      If this argument is None, the model lives on cuda:0 if cuda is available.
            - :param: `batch_size` (int): bert score processing batch size
            - :param: `nthreads` (int): number of threads
            - :param: `lang` (str): language of the sentences; has to specify
                      at least one of `model_type` or `lang`. `lang` needs to be
                      specified when `rescale_with_baseline` is True.
            - :param: `return_hash` (bool): return hash code of the setting
            - :param: `rescale_with_baseline` (bool): rescale bertscore with pre-computed baseline
            - :param: `baseline_path` (str): customized baseline file
            - :param: `use_fast_tokenizer` (bool): `use_fast` parameter passed to HF tokenizer
        """

        assert (lang is not None or model_type is not None), 'Either lang or model_type should be specified'

        if rescale_with_baseline:
            assert (lang is not None), 'Need to specify Language when rescaling with baseline'

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self._lang = lang
        self._rescale_with_baseline = rescale_with_baseline
        self._idf = idf
        self.batch_size = batch_size
        self.nthreads = nthreads
        self.all_layers = all_layers
        self.model_id_or_path = model_id_or_path

        if model_type is None:
            lang = lang.lower()
            self._model_type = lang2model[lang]
        else:
            self._model_type = model_type

        if num_layers is None:
            self._num_layers = model2layers[self.model_type]
        else:
            self._num_layers = num_layers

        # Building model and tokenizer
        self._use_fast_tokenizer = use_fast_tokenizer
        self._tokenizer = get_tokenizer(self.model_id_or_path, self._use_fast_tokenizer)
        self._model = get_model(self.model_id_or_path, self.num_layers, self.all_layers)
        self._model.to(self.device)

        self._idf_dict = None
        if idf_sents is not None:
            self.compute_idf(idf_sents)

        self._baseline_vals = None
        self.baseline_path = baseline_path
        self.use_custom_baseline = self.baseline_path is not None
        if self.baseline_path is None:
            self.baseline_path = os.path.join(
                os.path.dirname(__file__),
                f'rescale_baseline/{self.lang}/{self.model_type}.tsv',
            )

    @property
    def lang(self):
        return self._lang

    @property
    def idf(self):
        return self._idf

    @property
    def model_type(self):
        return self._model_type

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def rescale_with_baseline(self):
        return self._rescale_with_baseline

    @property
    def baseline_vals(self):
        if self._baseline_vals is None:
            if os.path.isfile(self.baseline_path):
                if not self.all_layers:
                    self._baseline_vals = torch.from_numpy(
                        pd.read_csv(self.baseline_path).iloc[self.num_layers].to_numpy()
                    )[1:].float()
                else:
                    self._baseline_vals = (
                        torch.from_numpy(pd.read_csv(self.baseline_path).to_numpy())[:, 1:].unsqueeze(1).float()
                    )
            else:
                raise ValueError(f'Baseline not Found for {self.model_type} on {self.lang} at {self.baseline_path}')

        return self._baseline_vals

    @property
    def use_fast_tokenizer(self):
        return self._use_fast_tokenizer

    @property
    def hash(self):
        return get_hash(
            self.model_type,
            self.num_layers,
            self.idf,
            self.rescale_with_baseline,
            self.use_custom_baseline,
            self.use_fast_tokenizer,
        )

    def compute_idf(self, sents):
        """
        Args:

        """
        if self._idf_dict is not None:
            warnings.warn('Overwriting the previous importance weights.')

        self._idf_dict = get_idf_dict(sents, self._tokenizer, nthreads=self.nthreads)

    def score(self, cands, refs, verbose=False, batch_size=64, return_hash=False):
        """
        Args:
            - :param: `cands` (list of str): candidate sentences
            - :param: `refs` (list of str or list of list of str): reference sentences

        Return:
            - :param: `(P, R, F)`: each is of shape (N); N = number of input
                      candidate reference pairs. if returning hashcode, the
                      output will be ((P, R, F), hashcode). If a candidate have
                      multiple references, the returned score of this candidate is
                      the *best* score among all references.
        """

        ref_group_boundaries = None
        if not isinstance(refs[0], str):
            ref_group_boundaries = []
            ori_cands, ori_refs = cands, refs
            cands, refs = [], []
            count = 0
            for cand, ref_group in zip(ori_cands, ori_refs):
                cands += [cand] * len(ref_group)
                refs += ref_group
                ref_group_boundaries.append((count, count + len(ref_group)))
                count += len(ref_group)

        if verbose:
            print('calculating scores...')
            start = time.perf_counter()

        if self.idf:
            assert self._idf_dict, 'IDF weights are not computed'
            idf_dict = self._idf_dict
        else:
            idf_dict = defaultdict(lambda: 1.0)
            idf_dict[self._tokenizer.sep_token_id] = 0
            idf_dict[self._tokenizer.cls_token_id] = 0

        all_preds = bert_cos_score_idf(
            self._model,
            refs,
            cands,
            self._tokenizer,
            idf_dict,
            verbose=verbose,
            device=self.device,
            batch_size=batch_size,
            all_layers=self.all_layers,
        ).cpu()

        if ref_group_boundaries is not None:
            max_preds = []
            for start, end in ref_group_boundaries:
                max_preds.append(all_preds[start:end].max(dim=0)[0])
            all_preds = torch.stack(max_preds, dim=0)

        if self.rescale_with_baseline:
            all_preds = (all_preds - self.baseline_vals) / (1 - self.baseline_vals)

        out = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]  # P, R, F

        if verbose:
            time_diff = time.perf_counter() - start
            print(f'done in {time_diff:.2f} seconds, {len(refs) / time_diff:.2f} sentences/sec')

        if return_hash:
            out = tuple([out, self.hash])

        return out

    def plot_example(self, candidate, reference, fname=''):
        """
        Args:
            - :param: `candidate` (str): a candidate sentence
            - :param: `reference` (str): a reference sentence
            - :param: `fname` (str): path to save the output plot
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        assert isinstance(candidate, str)
        assert isinstance(reference, str)

        idf_dict = defaultdict(lambda: 1.0)
        idf_dict[self._tokenizer.sep_token_id] = 0
        idf_dict[self._tokenizer.cls_token_id] = 0

        hyp_embedding, masks, padded_idf = get_bert_embedding(
            [candidate],
            self._model,
            self._tokenizer,
            idf_dict,
            device=self.device,
            all_layers=False,
        )
        ref_embedding, masks, padded_idf = get_bert_embedding(
            [reference],
            self._model,
            self._tokenizer,
            idf_dict,
            device=self.device,
            all_layers=False,
        )
        ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
        hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))
        sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
        sim = sim.squeeze(0).cpu()

        r_tokens = [self._tokenizer.decode([i]) for i in sent_encode(self._tokenizer, reference)][1:-1]
        h_tokens = [self._tokenizer.decode([i]) for i in sent_encode(self._tokenizer, candidate)][1:-1]
        sim = sim[1:-1, 1:-1]

        if self.rescale_with_baseline:
            sim = (sim - self.baseline_vals[2].item()) / (1 - self.baseline_vals[2].item())

        fig, ax = plt.subplots(figsize=(len(r_tokens), len(h_tokens)))
        im = ax.imshow(sim, cmap='Blues', vmin=0, vmax=1)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(r_tokens)))
        ax.set_yticks(np.arange(len(h_tokens)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(r_tokens, fontsize=10)
        ax.set_yticklabels(h_tokens, fontsize=10)
        ax.grid(False)
        plt.xlabel('Reference (tokenized)', fontsize=14)
        plt.ylabel('Candidate (tokenized)', fontsize=14)
        title = 'Similarity Matrix'
        if self.rescale_with_baseline:
            title += ' (after Rescaling)'
        plt.title(title, fontsize=14)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.2)
        fig.colorbar(im, cax=cax)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        # Loop over data dimensions and create text annotations.
        for i in range(len(h_tokens)):
            for j in range(len(r_tokens)):
                text = ax.text(
                    j,
                    i,
                    '{:.3f}'.format(sim[i, j].item()),
                    ha='center',
                    va='center',
                    color='k' if sim[i, j].item() < 0.5 else 'w',
                )

        fig.tight_layout()
        if fname != '':
            plt.savefig(fname, dpi=100)
            print('Saved figure to file: ', fname)
        plt.show()

    def __repr__(self):
        return f'{self.__class__.__name__}(hash={self.hash}, batch_size={self.batch_size}, nthreads={self.nthreads})'

    def __str__(self):
        return self.__repr__()
