from typing import Any, Dict, List, Set, Tuple

from evalscope.api.dataset import Sample
from evalscope.api.metric.scorer import AggScore, SampleScore, Score
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger
from evalscope.utils.ner import (
    DEFAULT_TAG_FIX_PATTERNS,
    calculate_bio_metrics,
    clean_prediction,
    create_target_text,
    extract_entities_from_text,
    extract_spans_from_bio,
    xml_to_bio_tags,
)
from .default_data_adapter import DefaultDataAdapter

logger = get_logger()


class NERAdapter(DefaultDataAdapter):
    """
    Base adapter class for Named Entity Recognition (NER) tasks.

    This adapter handles converting between BIO tagging schemes and XML-style entity markup,
    and provides evaluation metrics using seqeval.

    Subclasses should define their entity types and register the benchmark.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Define mapping from BIO tags to user-friendly tag names
        self.entity_type_map = {}
        # Add descriptions for each entity type
        self.entity_descriptions = {}

        # These will be initialized in setup_entity_mappings
        self.reverse_entity_map = {}
        self.entity_list = []
        self.entities_description = ''

        # Define common error patterns to handle
        self.tag_fix_patterns = DEFAULT_TAG_FIX_PATTERNS

        check_import('seqeval', 'seqeval', raise_error=True, feature_name='NER metrics')
        # Note: setup_entity_mappings() should be called by subclasses
        # after they define their entity_type_map and entity_descriptions

    def setup_entity_mappings(self):
        """
        Setup entity mappings and descriptions for prompt formatting.
        This should be called after entity_type_map and entity_descriptions are defined.
        """
        # Reverse mapping for converting back from prediction to evaluation
        self.reverse_entity_map = {v.lower(): k for k, v in self.entity_type_map.items()}

        # Create list of tags for prompt formatting
        self.entity_list = [f'<{ent.lower()}>' for ent in self.entity_type_map.values()]

        # Create description of entities for prompt
        self.entities_description = ', '.join([
            f'{self.entity_type_map[tag]} ({self.entity_descriptions[tag]})' for tag in self.entity_type_map
        ])

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a record with tokens and NER tags into a Sample.
        Creates both the raw text input and annotated text target.
        """
        tokens: List[str] = record['tokens']
        ner_tags: List[str] = record['ner_tags']

        # Create the input text by joining tokens
        input_text = ' '.join(tokens)

        # Process tokens and tags to create annotated target text
        target_text = create_target_text(tokens, ner_tags, self.entity_type_map)

        # Store tokens and tags in metadata for evaluation
        metadata = {'tokens': tokens, 'ner_tags': ner_tags}

        return Sample(input=input_text, target=target_text, metadata=metadata)

    def format_prompt_template(self, sample):
        """
        Format the prompt with entity types, available tags, and text to annotate.
        """
        return self.prompt_template.format(
            entities=self.entities_description, entity_list=', '.join(self.entity_list), text=sample.input
        )

    def format_fewshot_template(self, fewshot, sample):
        """
        Format the few-shot prompt with all required parameters.
        """
        return self.few_shot_prompt_template.format(
            fewshot=fewshot,
            entities=self.entities_description,
            entity_list=', '.join(self.entity_list),
            text=sample.input
        )

    def sample_to_fewshot(self, sample: Sample) -> str:
        """
        Format a sample as a few-shot example showing original and annotated text.
        """
        if not sample.metadata:
            return ''

        # Format few-shot examples to match the expected response format
        return f'Input:\n{sample.input}\n\nOutput:\n{sample.target}'

    def match_score(self, original_prediction, filtered_prediction, reference, task_state) -> Score:
        """
        Evaluate named entity recognition performance using seqeval.
        """
        from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        try:
            # Get the original tokens and tags from the reference metadata
            original_tokens = task_state.metadata['tokens']
            original_tags = task_state.metadata['ner_tags']

            if not original_tokens or len(original_tokens) == 0:
                if hasattr(reference, 'metadata') and reference.metadata:
                    original_tokens = reference.metadata['tokens']
                    original_tags = reference.metadata['ner_tags']

            # Clean and normalize the prediction
            cleaned_prediction = clean_prediction(filtered_prediction, self.tag_fix_patterns)

            # Convert XML-style prediction back to BIO tags aligned with original tokens
            pred_bio_tags = xml_to_bio_tags(cleaned_prediction, original_tokens, self.reverse_entity_map)

            # Use seqeval to calculate metrics
            # Note: seqeval expects lists of lists (one per sequence)
            y_true = [original_tags]
            y_pred = [pred_bio_tags]

            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)

            score.value = {'precision': precision, 'recall': recall, 'f1_score': f1, 'accuracy': accuracy}

            # Store tags for aggregation (proper micro-averaging in aggregate_scores)
            # This way aggregate_scores can compute metrics across all samples at once,
            # which gives you true micro-averaged scores rather than averaged macro scores.
            score.metadata = {'y_true': original_tags, 'y_pred': pred_bio_tags}
        except Exception as e:
            logger.warning(f'Error evaluating NER prediction: {str(e)}')
            score.value = {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'accuracy': 0.0}

        return score

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """
        Aggregate metrics across all samples using seqeval.
        """
        from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

        # Collect all predictions and references
        y_true_all = []
        y_pred_all = []

        for ss in sample_scores:
            # Extract the BIO tags from metadata if available
            # You may need to store these during match_score
            if hasattr(ss.score, 'metadata') and 'y_true' in ss.score.metadata and 'y_pred' in ss.score.metadata:
                y_true_all.append(ss.score.metadata['y_true'])
                y_pred_all.append(ss.score.metadata['y_pred'])

        if not y_true_all:
            # Fallback: calculate averages from individual scores
            num_samples = len(sample_scores)
            avg_precision = sum(ss.score.value.get('precision', 0.0) for ss in sample_scores) / num_samples
            avg_recall = sum(ss.score.value.get('recall', 0.0) for ss in sample_scores) / num_samples
            avg_f1 = sum(ss.score.value.get('f1_score', 0.0) for ss in sample_scores) / num_samples
            avg_accuracy = sum(ss.score.value.get('accuracy', 0.0) for ss in sample_scores) / num_samples
        else:
            # Use seqeval for micro-averaged metrics across all samples
            avg_precision = precision_score(y_true_all, y_pred_all)
            avg_recall = recall_score(y_true_all, y_pred_all)
            avg_f1 = f1_score(y_true_all, y_pred_all)
            avg_accuracy = accuracy_score(y_true_all, y_pred_all)

        num_samples = len(sample_scores)

        agg_scores = [
            AggScore(
                metric_name='precision',
                score=avg_precision,
                num=num_samples,
                metadata={'type': 'seqeval-micro-average'}
            ),
            AggScore(
                metric_name='recall', score=avg_recall, num=num_samples, metadata={'type': 'seqeval-micro-average'}
            ),
            AggScore(metric_name='f1_score', score=avg_f1, num=num_samples, metadata={'type': 'seqeval-micro-average'}),
            AggScore(
                metric_name='accuracy', score=avg_accuracy, num=num_samples, metadata={'type': 'seqeval-accuracy'}
            )
        ]

        return agg_scores
