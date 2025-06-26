"""
Arena mode implementation for model comparison
"""
import json
import numpy as np
import os
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from evalscope.metrics.llm_judge import LLMJudge
from evalscope.utils.logger import get_logger

logger = get_logger()


def read_reviews(review_path: str) -> List[Dict]:
    """
    Read model review results from a jsonl file.

    Args:
        review_path: Path to the review file (jsonl)

    Returns:
        List of review dictionaries
    """
    reviews = []
    if not os.path.exists(review_path):
        logger.warning(f'Review file does not exist: {review_path}')
        return reviews

    with open(review_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    review = json.loads(line)
                    reviews.append(review)
                except json.JSONDecodeError:
                    logger.warning(f'Failed to parse line: {line}')

    return reviews


def extract_model_outputs(reviews: List[Dict], output_key: str = 'content') -> Dict[str, Dict]:
    """
    Extract model outputs from review results.

    Args:
        reviews: List of review dictionaries
        output_key: Key for the model output in the review (default: 'content')

    Returns:
        Dictionary mapping question_id to model outputs
    """
    outputs = {}

    for review in reviews:
        model_name = review.get('model', '')
        subset_name = review.get('subset_name', '')
        question_id = review.get('index', None)

        if question_id is None:
            # Try to extract from raw_input
            raw_input = review.get('raw_input', {})
            if raw_input and 'id' in raw_input:
                question_id = raw_input['id']

        # Skip if no question_id found
        if question_id is None:
            continue

        # Get the question
        question = ''
        raw_input = review.get('raw_input', {})
        if raw_input and 'prompt' in raw_input:
            question = raw_input['prompt']

        # Get category/type
        category = subset_name
        if raw_input and 'category' in raw_input:
            category = raw_input['category']

        # Get model output
        output = ''
        choices = review.get('choices', [])
        if choices and len(choices) > 0:
            message = choices[0].get('message', {})
            if message and output_key in message:
                output = message[output_key]

        key = f'{subset_name}_{question_id}'
        if key not in outputs:
            outputs[key] = {'question_id': question_id, 'question': question, 'category': category, 'models': {}}

        outputs[key]['models'][model_name] = output

    return outputs


def extract_model_scores(reviews: List[Dict]) -> Dict[str, Dict]:
    """
    Extract model scores from review results.

    Args:
        reviews: List of review dictionaries

    Returns:
        Dictionary mapping question_id to model scores
    """
    scores = {}

    for review in reviews:
        model_name = review.get('model', '')
        subset_name = review.get('subset_name', '')
        question_id = review.get('index', None)

        if question_id is None:
            # Try to extract from raw_input
            raw_input = review.get('raw_input', {})
            if raw_input and 'id' in raw_input:
                question_id = raw_input['id']

        # Skip if no question_id found
        if question_id is None:
            continue

        # Get the question
        question = ''
        raw_input = review.get('raw_input', {})
        if raw_input and 'prompt' in raw_input:
            question = raw_input['prompt']

        # Get category/type
        category = subset_name
        if raw_input and 'category' in raw_input:
            category = raw_input['category']

        # Get review scores
        result = None
        choices = review.get('choices', [])
        if choices and len(choices) > 0:
            review_data = choices[0].get('review', {})
            if review_data and 'result' in review_data:
                result = review_data['result']

        key = f'{subset_name}_{question_id}'
        if key not in scores:
            scores[key] = {'question_id': question_id, 'question': question, 'category': category, 'models': {}}

        if result:
            scores[key]['models'][model_name] = result

    return scores


def compare_outputs_with_llm_judge(outputs: Dict[str, Dict],
                                   model_a: str,
                                   model_b: str,
                                   judge_config: Optional[Dict] = None) -> List[Dict]:
    """
    Compare outputs from two models using LLM judge.

    Args:
        outputs: Dictionary mapping question_id to model outputs
        model_a: Name of model A
        model_b: Name of model B
        judge_config: Configuration for LLM judge

    Returns:
        List of battle results
    """
    judge_config = judge_config or {}
    judge = LLMJudge(**judge_config)

    battles = []

    for key, data in outputs.items():
        question = data['question']
        category = data['category']
        question_id = data['question_id']

        if model_a not in data['models'] or model_b not in data['models']:
            logger.warning(f'Skipping {key} - missing model output')
            continue

        output_a = data['models'][model_a]
        output_b = data['models'][model_b]

        # Build the comparison prompt
        prompt = f"""Your job is to compare two AI assistants' responses to a user question and determine which one is better overall.

Question: {question}

Assistant A's response: {output_a}

Assistant B's response: {output_b}

Which response is better? Choose one of the following options:
A: Assistant A's response is better
B: Assistant B's response is better
tie: Both responses are equally good

Just return the letter "A", "B", or "tie", with no text around it.
"""  # noqa: E501

        # Get judgment
        result = judge(prompt)
        result = result.strip().lower()

        # Determine winner
        winner = 'tie'
        if result == 'a':
            winner = 'model_a'
        elif result == 'b':
            winner = 'model_b'

        battle = {
            'model_a': model_a,
            'model_b': model_b,
            'scores': [1.0, 1.0] if winner == 'tie' else ([1.0, 0.0] if winner == 'model_a' else [0.0, 1.0]),
            'win': winner,
            'tstamp': None,
            'language': 'NA',
            'question_id': question_id,
            'category': category,
            'question': question,
            'output_a': output_a,
            'output_b': output_b
        }

        battles.append(battle)

    return battles


def compare_outputs_with_scores(scores: Dict[str, Dict],
                                model_a: str,
                                model_b: str,
                                metric_name: str = None) -> List[Dict]:
    """
    Compare model performance using scores from reviews.

    Args:
        scores: Dictionary mapping question_id to model scores
        model_a: Name of model A
        model_b: Name of model B
        metric_name: Name of the metric to use for comparison (uses first metric if None)

    Returns:
        List of battle results
    """
    battles = []

    for key, data in scores.items():
        question = data['question']
        category = data['category']
        question_id = data['question_id']

        if model_a not in data['models'] or model_b not in data['models']:
            logger.warning(f'Skipping {key} - missing model scores')
            continue

        score_a = data['models'][model_a]
        score_b = data['models'][model_b]

        # If metric_name is not specified, use the first metric key
        if not metric_name:
            metric_keys = list(score_a.keys())
            if not metric_keys:
                logger.warning(f'Skipping {key} - no metrics found')
                continue

            # Try to find an overall score metric
            metric_name = next((k for k in metric_keys if 'overall' in k.lower()), metric_keys[0])

        # Get scores for the specified metric
        if metric_name not in score_a or metric_name not in score_b:
            logger.warning(f"Skipping {key} - metric '{metric_name}' not found")
            continue

        metric_score_a = score_a[metric_name]
        metric_score_b = score_b[metric_name]

        # Determine winner
        winner = 'tie'
        if metric_score_a > metric_score_b:
            winner = 'model_a'
        elif metric_score_b > metric_score_a:
            winner = 'model_b'

        battle = {
            'model_a': model_a,
            'model_b': model_b,
            'scores': [metric_score_a, metric_score_b],
            'win': winner,
            'tstamp': None,
            'language': 'NA',
            'question_id': question_id,
            'category': category,
            'question': question,
            'output_a': str(metric_score_a),  # Use score as output for reporting
            'output_b': str(metric_score_b)
        }

        battles.append(battle)

    return battles


def compute_elo_ratings(battles: List[Dict],
                        baseline_model: str = None,
                        init_rating: int = 1000,
                        scale: int = 400) -> Dict[str, float]:
    """
    Compute ELO ratings for models based on battle results.

    Args:
        battles: List of battle results
        baseline_model: Name of baseline model (if any)
        init_rating: Initial ELO rating for all models
        scale: ELO scale factor

    Returns:
        Dictionary mapping model names to ELO ratings
    """
    # Convert battles to DataFrame
    df = pd.DataFrame(battles)

    # Get unique models
    models = pd.concat([df['model_a'], df['model_b']]).unique()
    model_indices = pd.Series(np.arange(len(models)), index=models)

    # Duplicate battles for logistic regression
    df_dup = pd.concat([df, df], ignore_index=True)
    n = df_dup.shape[0]
    p = len(models)

    # Create feature matrix for logistic regression
    X = np.zeros([n, p])
    base = 10  # Base for ELO calculation
    X[np.arange(n), model_indices[df_dup['model_a']]] = +np.log(base)
    X[np.arange(n), model_indices[df_dup['model_b']]] = -np.log(base)

    # Create target vector
    Y = np.zeros(n)
    Y[df_dup['win'] == 'model_a'] = 1.0

    # Handle ties - mark first half of ties as wins for model_a
    tie_idx = (df_dup['win'] == 'tie')
    tie_idx[len(tie_idx) // 2:] = False
    Y[tie_idx] = 1.0

    # Check if we have enough data for logistic regression
    if len(np.unique(Y)) < 2:
        logger.warning('Warning: Only one class in the data - using simple ELO calculation')
        elo_scores = pd.Series(init_rating, index=models)

        # Simple calculation for one-sided results
        if np.all(Y == 1.0):
            # All wins for model_a
            for model in df['model_a'].unique():
                elo_scores[model] += scale
        elif np.all(Y == 0.0):
            # All wins for model_b
            for model in df['model_b'].unique():
                elo_scores[model] += scale
    else:
        # Use logistic regression to compute ELO ratings
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-8)
        lr.fit(X, Y)

        # Convert coefficients to ELO ratings
        elo_scores = pd.Series(scale * lr.coef_[0] + init_rating, index=models)

    # Set baseline model rating to init_rating if specified
    if baseline_model and baseline_model in elo_scores.index:
        baseline_rating = elo_scores[baseline_model]
        elo_scores = elo_scores + (init_rating - baseline_rating)

    return elo_scores.to_dict()


def get_leaderboard(elo_ratings: Dict[str, float],
                    battles: List[Dict] = None,
                    baseline_model: str = None) -> Dict[str, pd.DataFrame]:
    """
    Generate leaderboard from ELO ratings.

    Args:
        elo_ratings: Dictionary mapping model names to ELO ratings
        battles: List of battle results (for category breakdown)
        baseline_model: Name of baseline model (for win rate calculation)

    Returns:
        Dictionary mapping leaderboard type to DataFrame
    """
    # Overall leaderboard
    overall_df = pd.DataFrame({
        'Model': list(elo_ratings.keys()),
        'ELO': list(elo_ratings.values())
    }).sort_values(
        'ELO', ascending=False).reset_index(drop=True)

    # Round ELO scores
    overall_df['ELO'] = overall_df['ELO'].round().astype(int)

    # Calculate win rate against baseline if baseline_model is provided
    if baseline_model and baseline_model in elo_ratings:
        win_rates = {}
        base = 10  # Base for ELO calculation
        scale = 400  # Scale for ELO calculation

        for model, elo in elo_ratings.items():
            if model != baseline_model:
                # Win probability formula: 1 / (1 + 10^((R_b - R_a) / 400))
                win_prob = 1 / (1 + base**((elo_ratings[baseline_model] - elo) / scale))
                win_rates[model] = win_prob

        # Add win rate column
        overall_df['Win Rate vs Baseline'] = overall_df['Model'].map(lambda x: win_rates.get(x, 0.5)
                                                                     if x != baseline_model else 0.5).round(3)

    leaderboards = {'overall': overall_df}

    # Category leaderboards if battles are provided
    if battles:
        df = pd.DataFrame(battles)
        categories = df['category'].unique()

        for category in categories:
            category_battles = df[df['category'] == category].to_dict('records')
            category_ratings = compute_elo_ratings(category_battles, baseline_model)

            category_df = pd.DataFrame({
                'Model': list(category_ratings.keys()),
                'ELO': list(category_ratings.values())
            }).sort_values(
                'ELO', ascending=False).reset_index(drop=True)

            # Round ELO scores
            category_df['ELO'] = category_df['ELO'].round().astype(int)

            leaderboards[category] = category_df

    return leaderboards


def generate_battle_report(battles: List[Dict],
                           elo_ratings: Dict[str, float],
                           leaderboards: Dict[str, pd.DataFrame],
                           output_dir: str,
                           baseline_model: str = None) -> str:
    """
    Generate a detailed battle report.

    Args:
        battles: List of battle results
        elo_ratings: Dictionary mapping model names to ELO ratings
        leaderboards: Dictionary mapping leaderboard type to DataFrame
        output_dir: Directory to save the report
        baseline_model: Name of baseline model (if any)

    Returns:
        Path to the saved report
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save battle results
    battles_file = os.path.join(output_dir, 'battles.json')
    with open(battles_file, 'w', encoding='utf-8') as f:
        json.dump(battles, f, ensure_ascii=False, indent=2)

    # Save ELO ratings
    ratings_file = os.path.join(output_dir, 'elo_ratings.json')
    with open(ratings_file, 'w', encoding='utf-8') as f:
        json.dump(elo_ratings, f, ensure_ascii=False, indent=2)

    # Save leaderboards
    leaderboards_dir = os.path.join(output_dir, 'leaderboards')
    os.makedirs(leaderboards_dir, exist_ok=True)

    for board_name, df in leaderboards.items():
        board_file = os.path.join(leaderboards_dir, f'{board_name}_leaderboard.csv')
        df.to_csv(board_file, index=False)

    # Generate HTML report
    report_file = os.path.join(output_dir, 'arena_report.html')

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f'''<!DOCTYPE html>
<html>
<head>
    <title>Arena Battle Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
    </style>
</head>
<body>
    <h1>Arena Battle Report</h1>

    <h2>Overall Leaderboard</h2>
    <table>
        <tr>
            {' '.join(f'<th>{col}</th>' for col in leaderboards['overall'].columns)}
        </tr>
        {''.join(
            f'<tr>{" ".join(f"<td>{cell}</td>" for cell in row.values)}</tr>'
            for _, row in leaderboards['overall'].iterrows()
        )}
    </table>
''')

        # Add category leaderboards
        for board_name, df in leaderboards.items():
            if board_name != 'overall':
                f.write(f'''
    <h2>{board_name} Leaderboard</h2>
    <table>
        <tr>
            {' '.join(f'<th>{col}</th>' for col in df.columns)}
        </tr>
        {''.join(
            f'<tr>{" ".join(f"<td>{cell}</td>" for cell in row.values)}</tr>'
            for _, row in df.iterrows()
        )}
    </table>
''')

        # Add battle statistics
        # battle_df = pd.DataFrame(battles)

        # Count wins by model
        model_wins = defaultdict(int)
        for battle in battles:
            if battle['win'] == 'model_a':
                model_wins[battle['model_a']] += 1
            elif battle['win'] == 'model_b':
                model_wins[battle['model_b']] += 1

        # Calculate win percentages
        model_battles = defaultdict(int)
        for battle in battles:
            model_battles[battle['model_a']] += 1
            model_battles[battle['model_b']] += 1

        win_pcts = {model: wins / model_battles[model] for model, wins in model_wins.items()}

        # Create win statistics table
        stats_df = pd.DataFrame({
            'Model': list(model_battles.keys()),
            'Battles': list(model_battles.values()),
            'Wins': [model_wins.get(model, 0) for model in model_battles.keys()],
            'Win %': [round(win_pcts.get(model, 0) * 100, 1) for model in model_battles.keys()]
        }).sort_values(
            'Wins', ascending=False).reset_index(drop=True)

        f.write(f'''
    <h2>Battle Statistics</h2>
    <table>
        <tr>
            {' '.join(f'<th>{col}</th>' for col in stats_df.columns)}
        </tr>
        {''.join(
            f'<tr>{" ".join(f"<td>{cell}</td>" for cell in row.values)}</tr>'
            for _, row in stats_df.iterrows()
        )}
    </table>

    <h2>Battle Details</h2>
    <p>Total battles: {len(battles)}</p>
''')

        if baseline_model:
            f.write(f'<p>Baseline model: {baseline_model}</p>')

        f.write('''
</body>
</html>
''')

    logger.info(f'Battle report saved to {report_file}')
    return report_file


def run_arena_battle(output_paths: List[str],
                     comparison_method: str = 'llm_judge',
                     baseline_model: str = None,
                     judge_config: Optional[Dict] = None,
                     metric_name: str = None,
                     output_dir: str = None) -> Dict:
    """
    Run arena battle between models based on evaluation results.

    Args:
        output_paths: List of paths to evaluation output directories
        comparison_method: Method for comparing model outputs ('llm_judge' or 'score')
        baseline_model: Name of baseline model for ELO calculation
        judge_config: Configuration for LLM judge (if using 'llm_judge' method)
        metric_name: Name of metric to use for score comparison (if using 'score' method)
        output_dir: Directory to save battle results and report

    Returns:
        Dictionary with battle results and ELO ratings
    """
    # Ensure output directory exists
    if output_dir is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join('outputs', f'arena_battle_{timestamp}')

    os.makedirs(output_dir, exist_ok=True)

    all_reviews = []
    model_map = {}  # Map of output path to model name

    # Read reviews from each output path
    for path in output_paths:
        reviews_dir = os.path.join(path, 'reviews')
        if not os.path.exists(reviews_dir):
            logger.warning(f'Reviews directory not found: {reviews_dir}')
            continue

        # Read each model's reviews
        for model_dir in os.listdir(reviews_dir):
            model_path = os.path.join(reviews_dir, model_dir)
            if not os.path.isdir(model_path):
                continue

            # Find review files
            review_files = [f for f in os.listdir(model_path) if f.endswith('.jsonl')]
            for review_file in review_files:
                review_path = os.path.join(model_path, review_file)
                reviews = read_reviews(review_path)

                if reviews:
                    all_reviews.extend(reviews)

                    # Store model name
                    if path not in model_map and reviews[0].get('model'):
                        model_map[path] = reviews[0]['model']

    logger.info(f'Loaded {len(all_reviews)} reviews for battle')

    # Generate battles
    battles = []

    if comparison_method == 'llm_judge':
        # Extract model outputs for LLM judging
        outputs = extract_model_outputs(all_reviews)

        # Compare each pair of models
        models = list(set(review.get('model', '') for review in all_reviews if review.get('model')))

        for i, model_a in enumerate(models):
            for model_b in models[i + 1:]:
                logger.info(f'Comparing {model_a} vs {model_b} using LLM judge')
                model_battles = compare_outputs_with_llm_judge(outputs, model_a, model_b, judge_config)
                battles.extend(model_battles)

    elif comparison_method == 'score':
        # Extract model scores for direct comparison
        scores = extract_model_scores(all_reviews)

        # Compare each pair of models
        models = list(set(review.get('model', '') for review in all_reviews if review.get('model')))

        for i, model_a in enumerate(models):
            for model_b in models[i + 1:]:
                logger.info(f'Comparing {model_a} vs {model_b} using scores')
                model_battles = compare_outputs_with_scores(scores, model_a, model_b, metric_name)
                battles.extend(model_battles)

    # Compute ELO ratings
    elo_ratings = compute_elo_ratings(battles, baseline_model)

    # Generate leaderboards
    leaderboards = get_leaderboard(elo_ratings, battles, baseline_model)

    # Generate and save report
    report_path = generate_battle_report(battles, elo_ratings, leaderboards, output_dir, baseline_model)

    # Save all battles to file
    battles_file = os.path.join(output_dir, 'all_battles.json')
    with open(battles_file, 'w', encoding='utf-8') as f:
        json.dump(battles, f, ensure_ascii=False, indent=2)

    logger.info(f'Arena battle complete. Results saved to {output_dir}')

    return {'battles': battles, 'elo_ratings': elo_ratings, 'leaderboards': leaderboards, 'report_path': report_path}
