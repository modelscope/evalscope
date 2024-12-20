import json


def generate_mixed_dataset(evaluator_collection, samples):
    mixed_data = []
    for sample in samples:
        dataset_name = sample['source']
        evaluation_result = evaluator_collection.evaluate(dataset_name, sample)
        mixed_data.append({
            'id': sample['id'],
            'row_data': {
                'prompt': sample['row_data']['prompt'],
                'answer': sample['row_data']['answer']
            },
            'tags': sample['tags'],
            'task': sample['task_type'],
            'source': dataset_name,
            'evaluation': evaluation_result
        })
    return mixed_data


def save_to_jsonl(data, file_path):
    with open(file_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
