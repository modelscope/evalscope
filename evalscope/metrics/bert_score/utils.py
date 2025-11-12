# flake8: noqa
import os
import sys
import torch
from collections import Counter, defaultdict
from functools import partial
from itertools import chain
from math import log
from modelscope import AutoModel, AutoTokenizer
from multiprocessing import Pool
from packaging import version
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import GPT2Tokenizer, RobertaTokenizer
from transformers import __version__ as trans_version

from evalscope import __version__

__all__ = []

SCIBERT_URL_DICT = {
    'scibert-scivocab-uncased':
    'https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_uncased.tar',  # recommend by the SciBERT authors
    'scibert-scivocab-cased':
    'https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_cased.tar',
    'scibert-basevocab-uncased':
    'https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_basevocab_uncased.tar',
    'scibert-basevocab-cased':
    'https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_basevocab_cased.tar',
}

lang2model = defaultdict(lambda: 'bert-base-multilingual-cased')
lang2model.update({
    'en': 'roberta-large',
    'zh': 'bert-base-chinese',
    'tr': 'dbmdz/bert-base-turkish-cased',
    'en-sci': 'allenai/scibert_scivocab_uncased',
})

model2layers = {
    'bert-base-uncased': 9,  # 0.6925188074454226
    'bert-large-uncased': 18,  # 0.7210358126642836
    'bert-base-cased-finetuned-mrpc': 9,  # 0.6721947475618048
    'bert-base-multilingual-cased': 9,  # 0.6680687802637132
    'bert-base-chinese': 8,
    'roberta-base': 10,  # 0.706288719158983
    'roberta-large': 17,  # 0.7385974720781534
    'roberta-large-mnli': 19,  # 0.7535618640417984
    'roberta-base-openai-detector': 7,  # 0.7048158349432633
    'roberta-large-openai-detector': 15,  # 0.7462770207355116
    'xlnet-base-cased': 5,  # 0.6630103662114238
    'xlnet-large-cased': 7,  # 0.6598800720297179
    'xlm-mlm-en-2048': 6,  # 0.651262570131464
    'xlm-mlm-100-1280': 10,  # 0.6475166424401905
    # "scibert-scivocab-uncased": 8,  # 0.6590354319927313
    # "scibert-scivocab-cased": 9,  # 0.6536375053937445
    # "scibert-basevocab-uncased": 9,  # 0.6748944832703548
    # "scibert-basevocab-cased": 9,  # 0.6524624150542374
    'allenai/scibert_scivocab_uncased': 8,  # 0.6590354393124127
    'allenai/scibert_scivocab_cased': 9,  # 0.6536374902465466
    'nfliu/scibert_basevocab_uncased': 9,  # 0.6748945076082333
    'distilroberta-base': 5,  # 0.6797558139322964
    'distilbert-base-uncased': 5,  # 0.6756659152782033
    'distilbert-base-uncased-distilled-squad': 4,  # 0.6718318036382493
    'distilbert-base-multilingual-cased': 5,  # 0.6178131050889238
    'albert-base-v1': 10,  # 0.654237567249745
    'albert-large-v1': 17,  # 0.6755890754323239
    'albert-xlarge-v1': 16,  # 0.7031844211905911
    'albert-xxlarge-v1': 8,  # 0.7508642218461096
    'albert-base-v2': 9,  # 0.6682455591837927
    'albert-large-v2': 14,  # 0.7008537594374035
    'albert-xlarge-v2': 13,  # 0.7317228357869254
    'albert-xxlarge-v2': 8,  # 0.7505160257184014
    'xlm-roberta-base': 9,  # 0.6506799445871697
    'xlm-roberta-large': 17,  # 0.6941551437476826
    'google/electra-small-generator': 9,  # 0.6659421842117754
    'google/electra-small-discriminator': 11,  # 0.6534639151385759
    'google/electra-base-generator': 10,  # 0.6730033453857188
    'google/electra-base-discriminator': 9,  # 0.7032089590812965
    'google/electra-large-generator': 18,  # 0.6813370013104459
    'google/electra-large-discriminator': 14,  # 0.6896675824733477
    'google/bert_uncased_L-2_H-128_A-2': 1,  # 0.5887998733228855
    'google/bert_uncased_L-2_H-256_A-4': 1,  # 0.6114863547661203
    'google/bert_uncased_L-2_H-512_A-8': 1,  # 0.6177345529192847
    'google/bert_uncased_L-2_H-768_A-12': 2,  # 0.6191261237956839
    'google/bert_uncased_L-4_H-128_A-2': 3,  # 0.6076202863798991
    'google/bert_uncased_L-4_H-256_A-4': 3,  # 0.6205239036810148
    'google/bert_uncased_L-4_H-512_A-8': 3,  # 0.6375351621856903
    'google/bert_uncased_L-4_H-768_A-12': 3,  # 0.6561849979644787
    'google/bert_uncased_L-6_H-128_A-2': 5,  # 0.6200458425360283
    'google/bert_uncased_L-6_H-256_A-4': 5,  # 0.6277501629539081
    'google/bert_uncased_L-6_H-512_A-8': 5,  # 0.641952305130849
    'google/bert_uncased_L-6_H-768_A-12': 5,  # 0.6762186226247106
    'google/bert_uncased_L-8_H-128_A-2': 7,  # 0.6186876506711779
    'google/bert_uncased_L-8_H-256_A-4': 7,  # 0.6447993208267708
    'google/bert_uncased_L-8_H-512_A-8': 6,  # 0.6489729408169956
    'google/bert_uncased_L-8_H-768_A-12': 7,  # 0.6705203359541737
    'google/bert_uncased_L-10_H-128_A-2': 8,  # 0.6126762064125278
    'google/bert_uncased_L-10_H-256_A-4': 8,  # 0.6376350032576573
    'google/bert_uncased_L-10_H-512_A-8': 9,  # 0.6579006292799915
    'google/bert_uncased_L-10_H-768_A-12': 8,  # 0.6861146692220176
    'google/bert_uncased_L-12_H-128_A-2': 10,  # 0.6184105693383591
    'google/bert_uncased_L-12_H-256_A-4': 11,  # 0.6374004994430261
    'google/bert_uncased_L-12_H-512_A-8': 10,  # 0.65880012149526
    'google/bert_uncased_L-12_H-768_A-12': 9,  # 0.675911357700092
    'amazon/bort': 0,  # 0.41927911053036643
    'facebook/bart-base': 6,  # 0.7122259132414092
    'facebook/bart-large': 10,  # 0.7448671872459683
    'facebook/bart-large-cnn': 10,  # 0.7393148105835096
    'facebook/bart-large-mnli': 11,  # 0.7531665445691358
    'facebook/bart-large-xsum': 9,  # 0.7496408866539556
    't5-small': 6,  # 0.6813843919496912
    't5-base': 11,  # 0.7096044814981418
    't5-large': 23,  # 0.7244153820191929
    'vinai/bertweet-base': 9,  # 0.6529471006118857
    'microsoft/deberta-base': 9,  # 0.7088459455930344
    'microsoft/deberta-base-mnli': 9,  # 0.7395257063907247
    'microsoft/deberta-large': 16,  # 0.7511806792052013
    'microsoft/deberta-large-mnli': 18,  # 0.7736263649679905
    'microsoft/deberta-xlarge': 18,  # 0.7568670944373346
    'microsoft/deberta-xlarge-mnli': 40,  # 0.7780600929333213
    'YituTech/conv-bert-base': 10,  # 0.7058253551080789
    'YituTech/conv-bert-small': 10,  # 0.6544473011107349
    'YituTech/conv-bert-medium-small': 9,  # 0.6590097075123257
    'microsoft/mpnet-base': 8,  # 0.724976539498804
    'squeezebert/squeezebert-uncased': 9,  # 0.6543868703018726
    'squeezebert/squeezebert-mnli': 9,  # 0.6654799051284791
    'squeezebert/squeezebert-mnli-headless': 9,  # 0.6654799051284791
    'tuner007/pegasus_paraphrase': 15,  # 0.7188349436772694
    'google/pegasus-large': 8,  # 0.63960462272448
    'google/pegasus-xsum': 11,  # 0.6836878575233349
    'sshleifer/tiny-mbart': 2,  # 0.028246072231946733
    'facebook/mbart-large-cc25': 12,  # 0.6582922975802958
    'facebook/mbart-large-50': 12,  # 0.6464972230103133
    'facebook/mbart-large-en-ro': 12,  # 0.6791285137459857
    'facebook/mbart-large-50-many-to-many-mmt': 12,  # 0.6904136529270892
    'facebook/mbart-large-50-one-to-many-mmt': 12,  # 0.6847906439540236
    'allenai/led-base-16384': 6,  # 0.7122259170564179
    'facebook/blenderbot_small-90M': 7,  # 0.6489176335400088
    'facebook/blenderbot-400M-distill': 2,  # 0.5874774070540008
    'microsoft/prophetnet-large-uncased': 4,  # 0.586496184234925
    'microsoft/prophetnet-large-uncased-cnndm': 7,  # 0.6478379437729287
    'SpanBERT/spanbert-base-cased': 8,  # 0.6824006863686848
    'SpanBERT/spanbert-large-cased': 17,  # 0.705352690855603
    'microsoft/xprophetnet-large-wiki100-cased': 7,  # 0.5852499775879524
    'ProsusAI/finbert': 10,  # 0.6923213940752796
    'Vamsi/T5_Paraphrase_Paws': 12,  # 0.6941611753807352
    'ramsrigouthamg/t5_paraphraser': 11,  # 0.7200917597031539
    'microsoft/deberta-v2-xlarge': 10,  # 0.7393675784473045
    'microsoft/deberta-v2-xlarge-mnli': 17,  # 0.7620620803716714
    'microsoft/deberta-v2-xxlarge': 21,  # 0.7520547670281869
    'microsoft/deberta-v2-xxlarge-mnli': 22,  # 0.7742603457742682
    'allenai/longformer-base-4096': 7,  # 0.7089559593129316
    'allenai/longformer-large-4096': 14,  # 0.732408493548181
    'allenai/longformer-large-4096-finetuned-triviaqa': 14,  # 0.7365882744744722
    'zhiheng-huang/bert-base-uncased-embedding-relative-key': 4,  # 0.5995636595368777
    'zhiheng-huang/bert-base-uncased-embedding-relative-key-query': 7,  # 0.6303599452145718
    'zhiheng-huang/bert-large-uncased-whole-word-masking-embedding-relative-key-query': 19,  # 0.6896878492850327
    'google/mt5-small': 8,  # 0.6401166527273479
    'google/mt5-base': 11,  # 0.5663956536597241
    'google/mt5-large': 19,  # 0.6430931371732798
    'google/mt5-xl': 24,  # 0.6707200963021145
    'google/bigbird-roberta-base': 10,  # 0.6695606423502717
    'google/bigbird-roberta-large': 14,  # 0.6755874042374509
    'google/bigbird-base-trivia-itc': 8,  # 0.6930725491629892
    'princeton-nlp/unsup-simcse-bert-base-uncased': 10,  # 0.6703066531921142
    'princeton-nlp/unsup-simcse-bert-large-uncased': 18,  # 0.6958302800755326
    'princeton-nlp/unsup-simcse-roberta-base': 8,  # 0.6436615893535319
    'princeton-nlp/unsup-simcse-roberta-large': 13,  # 0.6812864385585965
    'princeton-nlp/sup-simcse-bert-base-uncased': 10,  # 0.7068074935240984
    'princeton-nlp/sup-simcse-bert-large-uncased': 18,  # 0.7111049471332378
    'princeton-nlp/sup-simcse-roberta-base': 10,  # 0.7253123806661946
    'princeton-nlp/sup-simcse-roberta-large': 16,  # 0.7497820277237173
    'dbmdz/bert-base-turkish-cased': 10,  # WMT18 seg en-tr 0.5522827687776142
    'dbmdz/distilbert-base-turkish-cased': 4,  # WMT18 seg en-tr 0.4742268041237113
    'google/byt5-small': 1,  # 0.5100025975052146
    'google/byt5-base': 17,  # 0.5810347173565313
    'google/byt5-large': 30,  # 0.6151895697554877
    'microsoft/deberta-v3-xsmall': 10,  # 0.6941803815412021
    'microsoft/deberta-v3-small': 4,  # 0.6651551203179679
    'microsoft/deberta-v3-base': 9,  # 0.7261586651018335
    'microsoft/mdeberta-v3-base': 10,  # 0.6778713684091584
    'microsoft/deberta-v3-large': 12,  # 0.6927693082293821
    'khalidalt/DeBERTa-v3-large-mnli': 18,  # 0.7428756686018376
}


def sent_encode(tokenizer, sent):
    'Encoding as sentence based on the tokenizer'
    sent = sent.strip()
    if sent == '':
        return tokenizer.build_inputs_with_special_tokens([])
    elif isinstance(tokenizer, GPT2Tokenizer) or isinstance(tokenizer, RobertaTokenizer):
        # for RoBERTa and GPT-2
        if version.parse(trans_version) >= version.parse('4.0.0'):
            if tokenizer.model_max_length > 10000000:
                tokenizer.model_max_length = 512

            return tokenizer.encode(
                sent,
                add_special_tokens=True,
                add_prefix_space=True,
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
        elif version.parse(trans_version) >= version.parse('3.0.0'):
            return tokenizer.encode(
                sent,
                add_special_tokens=True,
                add_prefix_space=True,
                max_length=tokenizer.max_len,
                truncation=True,
            )
        elif version.parse(trans_version) >= version.parse('2.0.0'):
            return tokenizer.encode(
                sent,
                add_special_tokens=True,
                add_prefix_space=True,
                max_length=tokenizer.max_len,
            )
        else:
            raise NotImplementedError(f'transformers version {trans_version} is not supported')
    else:
        if version.parse(trans_version) >= version.parse('4.0.0'):
            if tokenizer.model_max_length > 10000000:
                tokenizer.model_max_length = 512

            return tokenizer.encode(
                sent,
                add_special_tokens=True,
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
        elif version.parse(trans_version) >= version.parse('3.0.0'):
            return tokenizer.encode(
                sent,
                add_special_tokens=True,
                max_length=tokenizer.max_len,
                truncation=True,
            )
        elif version.parse(trans_version) >= version.parse('2.0.0'):
            return tokenizer.encode(sent, add_special_tokens=True, max_length=tokenizer.max_len)
        else:
            raise NotImplementedError(f'transformers version {trans_version} is not supported')


def get_model(model_type, num_layers, all_layers=None):
    if model_type.startswith('scibert'):
        model = AutoModel.from_pretrained(cache_scibert(model_type))
    elif 't5' in model_type:
        from transformers import T5EncoderModel

        model = T5EncoderModel.from_pretrained(model_type)
    else:
        model = AutoModel.from_pretrained(model_type)
    model.eval()

    if hasattr(model, 'decoder') and hasattr(model, 'encoder'):
        model = model.encoder

    # drop unused layers
    if not all_layers:
        if hasattr(model, 'n_layers'):  # xlm
            assert (
                0 <= num_layers <= model.n_layers
            ), f'Invalid num_layers: num_layers should be between 0 and {model.n_layers} for {model_type}'
            model.n_layers = num_layers
        elif hasattr(model, 'layer'):  # xlnet
            assert (
                0 <= num_layers <= len(model.layer)
            ), f'Invalid num_layers: num_layers should be between 0 and {len(model.layer)} for {model_type}'
            model.layer = torch.nn.ModuleList([layer for layer in model.layer[:num_layers]])
        elif hasattr(model, 'encoder'):  # albert
            if hasattr(model.encoder, 'albert_layer_groups'):
                assert (
                    0 <= num_layers <= model.encoder.config.num_hidden_layers
                ), f'Invalid num_layers: num_layers should be between 0 and {model.encoder.config.num_hidden_layers} for {model_type}'
                model.encoder.config.num_hidden_layers = num_layers
            elif hasattr(model.encoder, 'block'):  # t5
                assert (
                    0 <= num_layers <= len(model.encoder.block)
                ), f'Invalid num_layers: num_layers should be between 0 and {len(model.encoder.block)} for {model_type}'
                model.encoder.block = torch.nn.ModuleList([layer for layer in model.encoder.block[:num_layers]])
            else:  # bert, roberta
                assert (
                    0 <= num_layers <= len(model.encoder.layer)
                ), f'Invalid num_layers: num_layers should be between 0 and {len(model.encoder.layer)} for {model_type}'
                model.encoder.layer = torch.nn.ModuleList([layer for layer in model.encoder.layer[:num_layers]])
        elif hasattr(model, 'transformer'):  # bert, roberta
            assert (
                0 <= num_layers <= len(model.transformer.layer)
            ), f'Invalid num_layers: num_layers should be between 0 and {len(model.transformer.layer)} for {model_type}'
            model.transformer.layer = torch.nn.ModuleList([layer for layer in model.transformer.layer[:num_layers]])
        elif hasattr(model, 'layers'):  # bart
            assert (
                0 <= num_layers <= len(model.layers)
            ), f'Invalid num_layers: num_layers should be between 0 and {len(model.layers)} for {model_type}'
            model.layers = torch.nn.ModuleList([layer for layer in model.layers[:num_layers]])
        else:
            raise ValueError('Not supported')
    else:
        if hasattr(model, 'output_hidden_states'):
            model.output_hidden_states = True
        elif hasattr(model, 'encoder'):
            model.encoder.output_hidden_states = True
        elif hasattr(model, 'transformer'):
            model.transformer.output_hidden_states = True
        # else:
        #     raise ValueError(f"Not supported model architecture: {model_type}")

    return model


def get_tokenizer(model_type, use_fast=False):
    if model_type.startswith('scibert'):
        model_type = cache_scibert(model_type)

    if version.parse(trans_version) >= version.parse('4.0.0'):
        tokenizer = AutoTokenizer.from_pretrained(model_type, use_fast=use_fast)
    else:
        assert not use_fast, 'Fast tokenizer is not available for version < 4.0.0'
        tokenizer = AutoTokenizer.from_pretrained(model_type)

    return tokenizer


def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, :lens[i]] = 1
    return padded, lens, mask


def bert_encode(model, x, attention_mask, all_layers=False):
    model.eval()
    with torch.no_grad():
        out = model(x, attention_mask=attention_mask, output_hidden_states=all_layers)
    if all_layers:
        emb = torch.stack(out[-1], dim=2)
    else:
        emb = out[0]
    return emb


def process(a, tokenizer=None):
    if tokenizer is not None:
        a = sent_encode(tokenizer, a)
    return set(a)


def get_idf_dict(arr, tokenizer, nthreads=4):
    """
    Returns mapping from word piece index to its inverse document frequency.


    Args:
        - :param: `arr` (list of str) : sentences to process.
        - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
        - :param: `nthreads` (int) : number of CPU threads to use
    """
    idf_count = Counter()
    num_docs = len(arr)

    process_partial = partial(process, tokenizer=tokenizer)

    if nthreads > 0:
        with Pool(nthreads) as p:
            idf_count.update(chain.from_iterable(p.map(process_partial, arr)))
    else:
        idf_count.update(chain.from_iterable(map(process_partial, arr)))

    idf_dict = defaultdict(lambda: log((num_docs + 1) / (1)))
    idf_dict.update({idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()})
    return idf_dict


def collate_idf(arr, tokenizer, idf_dict, device='cuda:0'):
    """
    Helper function that pads a list of sentences to hvae the same length and
    loads idf score for words in the sentences.

    Args:
        - :param: `arr` (list of str): sentences to process.
        - :param: `tokenize` : a function that takes a string and return list
                  of tokens.
        - :param: `numericalize` : a function that takes a list of tokens and
                  return list of token indexes.
        - :param: `idf_dict` (dict): mapping a word piece index to its
                               inverse document frequency
        - :param: `pad` (str): the padding token.
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """
    arr = [sent_encode(tokenizer, a) for a in arr]

    idf_weights = [[idf_dict[i] for i in a] for a in arr]

    pad_token = tokenizer.pad_token_id

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, 0, dtype=torch.float)

    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, padded_idf, lens, mask


def get_bert_embedding(
    all_sens,
    model,
    tokenizer,
    idf_dict,
    batch_size=-1,
    device='cuda:0',
    all_layers=False,
):
    """
    Compute BERT embedding in batches.

    Args:
        - :param: `all_sens` (list of str) : sentences to encode.
        - :param: `model` : a BERT model from `pytorch_pretrained_bert`.
        - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
        - :param: `idf_dict` (dict) : mapping a word piece index to its
                               inverse document frequency
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """

    padded_sens, padded_idf, lens, mask = collate_idf(all_sens, tokenizer, idf_dict, device=device)

    if batch_size == -1:
        batch_size = len(all_sens)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_sens), batch_size):
            batch_embedding = bert_encode(
                model,
                padded_sens[i:i + batch_size],
                attention_mask=mask[i:i + batch_size],
                all_layers=all_layers,
            )
            embeddings.append(batch_embedding)
            del batch_embedding

    total_embedding = torch.cat(embeddings, dim=0)

    return total_embedding, mask, padded_idf


def greedy_cos_idf(
    ref_embedding,
    ref_masks,
    ref_idf,
    hyp_embedding,
    hyp_masks,
    hyp_idf,
    all_layers=False,
):
    """
    Compute greedy matching based on cosine similarity.

    Args:
        - :param: `ref_embedding` (torch.Tensor):
                   embeddings of reference sentences, BxKxd,
                   B: batch size, K: longest length, d: bert dimenison
        - :param: `ref_lens` (list of int): list of reference sentence length.
        - :param: `ref_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                   reference sentences.
        - :param: `ref_idf` (torch.Tensor): BxK, idf score of each word
                   piece in the reference setence
        - :param: `hyp_embedding` (torch.Tensor):
                   embeddings of candidate sentences, BxKxd,
                   B: batch size, K: longest length, d: bert dimenison
        - :param: `hyp_lens` (list of int): list of candidate sentence length.
        - :param: `hyp_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                   candidate sentences.
        - :param: `hyp_idf` (torch.Tensor): BxK, idf score of each word
                   piece in the candidate setence
    """
    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

    if all_layers:
        B, _, L, D = hyp_embedding.size()
        hyp_embedding = (
            hyp_embedding.transpose(1, 2).transpose(0, 1).contiguous().view(L * B, hyp_embedding.size(1), D)
        )
        ref_embedding = (
            ref_embedding.transpose(1, 2).transpose(0, 1).contiguous().view(L * B, ref_embedding.size(1), D)
        )
    batch_size = ref_embedding.size(0)
    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
    if all_layers:
        masks = masks.unsqueeze(0).expand(L, -1, -1, -1).contiguous().view_as(sim)
    else:
        masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)

    masks = masks.float().to(sim.device)
    sim = sim * masks

    word_precision = sim.max(dim=2)[0]
    word_recall = sim.max(dim=1)[0]

    hyp_idf.div_(hyp_idf.sum(dim=1, keepdim=True))
    ref_idf.div_(ref_idf.sum(dim=1, keepdim=True))
    precision_scale = hyp_idf.to(word_precision.device)
    recall_scale = ref_idf.to(word_recall.device)
    if all_layers:
        precision_scale = (precision_scale.unsqueeze(0).expand(L, B, -1).contiguous().view_as(word_precision))
        recall_scale = (recall_scale.unsqueeze(0).expand(L, B, -1).contiguous().view_as(word_recall))
    P = (word_precision * precision_scale).sum(dim=1)
    R = (word_recall * recall_scale).sum(dim=1)
    F = 2 * P * R / (P + R)

    hyp_zero_mask = hyp_masks.sum(dim=1).eq(2)
    ref_zero_mask = ref_masks.sum(dim=1).eq(2)

    if all_layers:
        P = P.view(L, B)
        R = R.view(L, B)
        F = F.view(L, B)

    if torch.any(hyp_zero_mask):
        print(
            'Warning: Empty candidate sentence detected; setting raw BERTscores to 0.',
            file=sys.stderr,
        )
        P = P.masked_fill(hyp_zero_mask, 0.0)
        R = R.masked_fill(hyp_zero_mask, 0.0)

    if torch.any(ref_zero_mask):
        print(
            'Warning: Empty reference sentence detected; setting raw BERTScores to 0.',
            file=sys.stderr,
        )
        P = P.masked_fill(ref_zero_mask, 0.0)
        R = R.masked_fill(ref_zero_mask, 0.0)

    F = F.masked_fill(torch.isnan(F), 0.0)

    return P, R, F


def bert_cos_score_idf(
    model,
    refs,
    hyps,
    tokenizer,
    idf_dict,
    verbose=False,
    batch_size=64,
    device='cuda:0',
    all_layers=False,
):
    """
    Compute BERTScore.

    Args:
        - :param: `model` : a BERT model in `pytorch_pretrained_bert`
        - :param: `refs` (list of str): reference sentences
        - :param: `hyps` (list of str): candidate sentences
        - :param: `tokenzier` : a BERT tokenizer corresponds to `model`
        - :param: `idf_dict` : a dictionary mapping a word piece index to its
                               inverse document frequency
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `batch_size` (int): bert score processing batch size
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """
    preds = []

    def dedup_and_sort(l):
        return sorted(list(set(l)), key=lambda x: len(x.split(' ')), reverse=True)

    sentences = dedup_and_sort(refs + hyps)
    embs = []
    iter_range = range(0, len(sentences), batch_size)
    if verbose:
        print('computing bert embedding.')
        iter_range = tqdm(iter_range)
    stats_dict = dict()
    for batch_start in iter_range:
        sen_batch = sentences[batch_start:batch_start + batch_size]
        embs, masks, padded_idf = get_bert_embedding(
            sen_batch, model, tokenizer, idf_dict, device=device, all_layers=all_layers
        )
        embs = embs.cpu()
        masks = masks.cpu()
        padded_idf = padded_idf.cpu()
        for i, sen in enumerate(sen_batch):
            sequence_len = masks[i].sum().item()
            emb = embs[i, :sequence_len]
            idf = padded_idf[i, :sequence_len]
            stats_dict[sen] = (emb, idf)

    def pad_batch_stats(sen_batch, stats_dict, device):
        stats = [stats_dict[s] for s in sen_batch]
        emb, idf = zip(*stats)
        emb = [e.to(device) for e in emb]
        idf = [i.to(device) for i in idf]
        lens = [e.size(0) for e in emb]
        emb_pad = pad_sequence(emb, batch_first=True, padding_value=2.0)
        idf_pad = pad_sequence(idf, batch_first=True)

        def length_to_mask(lens):
            lens = torch.tensor(lens, dtype=torch.long)
            max_len = max(lens)
            base = torch.arange(max_len, dtype=torch.long).expand(len(lens), max_len)
            return base < lens.unsqueeze(1)

        pad_mask = length_to_mask(lens).to(device)
        return emb_pad, pad_mask, idf_pad

    device = next(model.parameters()).device
    iter_range = range(0, len(refs), batch_size)
    if verbose:
        print('computing greedy matching.')
        iter_range = tqdm(iter_range)

    with torch.no_grad():
        for batch_start in iter_range:
            batch_refs = refs[batch_start:batch_start + batch_size]
            batch_hyps = hyps[batch_start:batch_start + batch_size]
            ref_stats = pad_batch_stats(batch_refs, stats_dict, device)
            hyp_stats = pad_batch_stats(batch_hyps, stats_dict, device)

            P, R, F1 = greedy_cos_idf(*ref_stats, *hyp_stats, all_layers)
            preds.append(torch.stack((P, R, F1), dim=-1).cpu())
    preds = torch.cat(preds, dim=1 if all_layers else 0)
    return preds


def get_hash(
    model,
    num_layers,
    idf,
    rescale_with_baseline,
    use_custom_baseline,
    use_fast_tokenizer,
):
    msg = '{}_L{}{}_version={}(hug_trans={})'.format(
        model, num_layers, '_idf' if idf else '_no-idf', __version__, trans_version
    )
    if rescale_with_baseline:
        if use_custom_baseline:
            msg += '-custom-rescaled'
        else:
            msg += '-rescaled'
    if use_fast_tokenizer:
        msg += '_fast-tokenizer'
    return msg


def cache_scibert(model_type, cache_folder='~/.cache/torch/transformers'):
    if not model_type.startswith('scibert'):
        return model_type

    underscore_model_type = model_type.replace('-', '_')
    cache_folder = os.path.abspath(os.path.expanduser(cache_folder))
    filename = os.path.join(cache_folder, underscore_model_type)

    # download SciBERT models
    if not os.path.exists(filename):
        cmd = f'mkdir -p {cache_folder}; cd {cache_folder};'
        cmd += f'wget {SCIBERT_URL_DICT[model_type]}; tar -xvf {underscore_model_type}.tar;'
        cmd += f'rm -f {underscore_model_type}.tar ; cd {underscore_model_type}; tar -zxvf weights.tar.gz; mv weights/* .;'
        cmd += f'rm -f weights.tar.gz; rmdir weights; mv bert_config.json config.json;'
        print(cmd)
        print(f'downloading {model_type} model')
        os.system(cmd)

    # fix the missing files in scibert
    json_file = os.path.join(filename, 'special_tokens_map.json')
    if not os.path.exists(json_file):
        with open(json_file, 'w') as f:
            print(
                '{"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]", "mask_token": "[MASK]"}',
                file=f,
            )

    json_file = os.path.join(filename, 'added_tokens.json')
    if not os.path.exists(json_file):
        with open(json_file, 'w') as f:
            print('{}', file=f)

    if 'uncased' in model_type:
        json_file = os.path.join(filename, 'tokenizer_config.json')
        if not os.path.exists(json_file):
            with open(json_file, 'w') as f:
                print('{"do_lower_case": true, "max_len": 512, "init_inputs": []}', file=f)

    return filename
