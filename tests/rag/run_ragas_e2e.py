"""End-to-end RAGAS 0.4.x evaluation test."""
import os
import sys
import traceback

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Load .env
from pathlib import Path

env_file = Path(__file__).resolve().parents[2] / '.env'
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, val = line.split('=', 1)
            os.environ.setdefault(key.strip(), val.strip())

from evalscope.backend.rag_eval.ragas import RAGASEvalConfig, rag_eval

eval_config = RAGASEvalConfig(
    testset_file=str(Path(__file__).parent / 'ragas_mini_testset.json'),
    critic_llm={
        'model_name': 'qwen-plus',
        'provider': 'openai',
        'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': os.environ.get('DASHSCOPE_API_KEY', ''),
    },
    embeddings={
        'model_name_or_path': 'BAAI/bge-base-zh-v1.5',
        'provider': 'huggingface',
    },
    metrics=['faithfulness'],
    batch_size=1,
    raise_exceptions=True,
)

try:
    result = rag_eval(eval_config)
    print('=' * 60)
    print('RAGAS evaluation result:')
    print(result)
    print('=' * 60)

    # Validate results
    if 'faithfulness' in result:
        score = result['faithfulness']
        print(f'\nfaithfulness score: {score}')
        if isinstance(score, (int, float)) and 0.0 <= score <= 1.0:
            print('✓ Score is in valid range [0.0, 1.0]')
        else:
            print(f'✗ Score out of range or not numeric: {score}')
    else:
        print(f"✗ 'faithfulness' not found in result keys: {list(result.keys())}")

    print('\n✓ RAGAS E2E test PASSED')
except Exception as e:
    print('=' * 60)
    print(f'✗ RAGAS E2E test FAILED: {e}')
    print('=' * 60)
    traceback.print_exc()
    sys.exit(1)
