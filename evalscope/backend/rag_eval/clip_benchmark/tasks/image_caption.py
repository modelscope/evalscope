import os
import pandas as pd
from tqdm import tqdm

from evalscope.backend.rag_eval.utils.tools import save_to_jsonl, save_to_tsv
from evalscope.utils.logger import get_logger

logger = get_logger()


def evaluate(model, dataloader, limit=None, output_path=''):
    """
    Evaluate the model on the dataset
    Parameters
    ----------
    model: MultiModalModel
        model to caption the image
    dataloader: torch.utils.data.Dataloader
    limit: int
        limit the number of samples to evaluate
    Returns
    -------
    dict of retrieval metrics
    """
    sample_count = 0
    dataloader = dataloader_with_indices(dataloader)
    query_caption_index = []
    total_captions = []
    total_querys = []
    for batch_images, batch_texts, inds in tqdm(dataloader):
        captions = model.encode_image(batch_images)
        querys = [text for texts in batch_texts for text in texts]

        batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]

        total_captions.extend(captions)
        total_querys.extend(querys)
        query_caption_index.extend(batch_texts_image_index)

        if limit is not None:
            # Update sample counter
            sample_count += len(batch_images)

            if sample_count >= limit:
                break

    write_file(total_querys, total_captions, query_caption_index, output_path)
    return {'convertion_successful': True, 'save_path': output_path}


def write_file(query_list, corpus_list, qrels_list, output_path):
    # 处理 query_list
    query_df = pd.DataFrame(query_list, columns=['text'])
    query_df['_id'] = query_df.index
    query_df = query_df[['_id', 'text']]
    save_to_jsonl(query_df, os.path.join(output_path, 'queries.jsonl'))

    # 处理 corpus_list
    corpus_df = pd.DataFrame(corpus_list, columns=['text'])
    corpus_df['_id'] = corpus_df.index
    corpus_df = corpus_df[['_id', 'text']]
    save_to_jsonl(corpus_df, os.path.join(output_path, 'corpus.jsonl'))

    # 处理 qrels_list
    qrels_df = pd.DataFrame(qrels_list, columns=['corpus-id'])
    qrels_df['query-id'] = qrels_df.index
    qrels_df['score'] = 1
    qrels_df = qrels_df[['query-id', 'corpus-id', 'score']]
    save_to_tsv(qrels_df, os.path.join(output_path, 'qrels', 'test.tsv'))

    logger.info('Write files to {}'.format(output_path))
    return


def dataloader_with_indices(dataloader):
    start = 0
    for x, y in dataloader:
        end = start + len(x)
        inds = list(range(start, end))
        yield x, y, inds
        start = end
