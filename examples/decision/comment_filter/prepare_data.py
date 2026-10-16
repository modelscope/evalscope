"""Prepare unchanged public moderation tasks and explicit Chinese tutorial fixtures."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evalscope import TaskConfig
from evalscope.api.dataset import load_dataset_from_hub
from evalscope.api.model.decision import DecisionInput
from evalscope.api.registry import get_benchmark
from evalscope.models.utils.decision import build_systemone_request

POLICY = (
    '这是生活与摄影经验交流评论区。折叠商家招揽客户、推销付费商品或服务、返佣导购、'
    '刷单兼职和索取支付密码或验证码的引流。正常使用体验、价格讨论、提问、'
    '揭露或引用广告以提醒他人、无商业销售的免费社区活动应保留。'
    '即使自称不是广告，只要实际推销或引流仍应折叠；没有营销证据时保留。'
)
CHOICES = {
    'normal': '正常交流、提问、免费社区活动或反诈提醒，没有实际营销引流。',
    'promotion': '推销商品或付费服务、商家获客或返佣推广，但不满足诈骗诱导条件。',
    'scam': '刷单返现、保证高收益并要求先付款，或索取支付密码、短信验证码。',
}
LEVELS = list(CHOICES.values())
SUBSETS = ['zh_noul', 'zh_choice', 'zh_score']


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write a UTF-8 dataset without changing record values or order."""
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records), encoding='utf-8')


def main() -> None:
    """Resolve public data through EvalScope and verify both native dataset loaders."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/comment-filter/data'))
    parser.add_argument('--model', default='decision-model-preview')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset_from_hub('evalscope/OpenDecisions', split='train')
    records = [row for row in dataset if row['use_case'] == 'moderation_trust_safety']
    if not records:
        raise ValueError('The source contains no moderation_trust_safety records.')
    public_dir = args.output / 'open_decisions_moderation'
    public_dir.mkdir(exist_ok=True)
    dataset.filter(lambda row: row['use_case'] == 'moderation_trust_safety').to_parquet(public_dir / 'train.parquet')
    write_jsonl(args.output / 'open_decisions_records.jsonl', records)

    fixtures = json.loads(Path(__file__).with_name('chinese_comments.json').read_text(encoding='utf-8'))
    questions = {
        'noul': {'type': 'noul', 'instructions': '根据评论区规则，这条评论是否应该折叠？yes 表示折叠。'},
        'choice': {'type': 'choice', 'instructions': '根据评论区规则，对这条评论分类。', 'criteria': CHOICES},
        'score': {'type': 'score', 'instructions': '根据评论区规则，判断这条评论的干扰等级。', 'criteria': LEVELS},
    }
    for kind, cases in fixtures.items():
        rows = [
            {
                'id': source_id,
                'state': {'policy': POLICY, 'comment': comment},
                'question': questions[kind],
                'expected': expected,
                'fixed_classes': kind == 'choice',
                'language': 'zh',
            }
            for source_id, comment, expected in cases
        ]
        write_jsonl(args.output / f'zh_{kind}_test.jsonl', rows)

    config = TaskConfig(
        model=args.model,
        eval_type='systemone_api',
        datasets=['open_decisions', 'general_decision'],
        dataset_args={
            'open_decisions': {'local_path': str(public_dir)},
            'general_decision': {'local_path': str(args.output), 'subset_list': SUBSETS},
        },
    )
    counts = {}
    for name in config.datasets:
        loaded = get_benchmark(name, config).load_dataset()
        counts[name] = {subset: len(samples) for subset, samples in loaded.items()}
        if name == 'general_decision':
            bodies = []
            for sample in loaded['zh_noul']:
                message = next(message for message in sample.input if message.role == 'user')
                request = DecisionInput.model_validate(message.internal['systemone'])
                bodies.append(build_systemone_request(request, config.model))
            write_jsonl(args.output.parent / 'perf_noul.jsonl', bodies)
    manifest = {
        'source': 'evalscope/OpenDecisions',
        'upstream': 'jgeuter/OpenDecisions',
        'source_split': 'train',
        'source_rows': len(dataset),
        'source_fingerprint': dataset._fingerprint,
        'selection': {'use_case': 'moderation_trust_safety'},
        'selected_kinds': dict(Counter(row['question']['type'] for row in records)),
        'counts': counts,
        'public_license': 'CC BY-SA 4.0; per-item provenance retained in exported records',
        'custom_data': 'Synthetic tutorial fixtures, rule-labeled; no independent human review or user-log sampling.',
        'sha256': {
            str(path.relative_to(args.output)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.output.rglob('*'))
            if path.is_file() and path.name != 'manifest.json'
        },
    }
    (args.output / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
