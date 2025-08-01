import os
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List

from evalscope.api.dataset import Dataset, DatasetDict, RemoteDataLoader, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.model import Model, ModelOutput
from evalscope.api.registry import get_filter, get_metric
from evalscope.filters.ensemble import build_filter_ensemble
from evalscope.report import Report, ReportGenerator
from .benchmark import DataAdapter


class DefaultDataAdapter(DataAdapter):
    """
    Default Data Adapter for the benchmark.
    This class can be extended to implement specific data loading and processing logic.
    """
    """ DATA LOADING HOOKS """

    def load_dataset(self) -> DatasetDict:
        if os.path.exists(self.dataset_id):
            # Load dataset from local path
            pass
        else:
            # Load dataset from remote source
            self.test_dataset = self.load_subsets(self.load_subset)

            if self.few_shot_num > 0 and self.train_split:
                self.fewshot_dataset = self.load_subsets(self.load_fewshot_subset)

        return self.test_dataset

    def load_subsets(self, load_func: Callable[[str], Dataset]) -> DatasetDict:

        if self.reformat_subset:
            subset_data = load_func(self.default_subset)
            dataset_dict = DatasetDict.from_dataset(subset_data)
        else:
            subset_dict = defaultdict()
            for subset in self.subset_list:
                subset_data = load_func(subset)
                subset_dict[subset] = subset_data
            dataset_dict = DatasetDict(subset_dict)
        return dataset_dict

    def load_subset(self, subset: str) -> Dataset:
        """
        Load a specific subset of the dataset.
        """
        split = subset if self.split_as_subset else self.eval_split
        subset_name = self.default_subset if self.split_as_subset else subset

        loader = RemoteDataLoader(
            data_id_or_path=self.dataset_id,
            split=split,
            subset=subset_name,
            sample_fields=partial(self.record_to_sample, few_shot_num=self.few_shot_num),
            limit=self._task_config.limit,
            repeats=self._task_config.repeats,
            data_source=self._task_config.dataset_hub,
        )
        return loader.load()

    def load_fewshot_subset(self, subset: str) -> Dataset:
        """
        Load a few-shot subset of the dataset.
        """
        split = subset if self.split_as_subset else self.train_split
        subset_name = self.default_subset if self.split_as_subset else subset

        loader = RemoteDataLoader(
            data_id_or_path=self.dataset_id,
            split=split,
            subset=subset_name,
            sample_fields=partial(self.record_to_sample, few_shot_num=0),
            limit=self.few_shot_num,
            shuffle=self.few_shot_random,
            data_source=self._task_config.dataset_hub,
        )
        return loader.load()

    def record_to_sample(self, few_shot_num: int, record: Dict[str, Any]) -> Sample:
        raise NotImplementedError('This method should be implemented in subclasses')

    def sample_to_fewshot(self, sample: Sample) -> str:
        raise NotImplementedError('This method should be implemented in subclasses')

    def format_query_template(self, prompt: str) -> str:
        """
        Format the prompt template with the sample data.
        This method can be overridden in subclasses to implement specific formatting logic.
        """
        return self.query_template.format(prompt=prompt)

    def format_fewshot_template(self, fewshot: str, prompt: str) -> str:
        """
        Format the few-shot template with the few-shot examples.
        This method can be overridden in subclasses to implement specific formatting logic.
        """
        return self.fewshot_prompt_template.format(fewshot=fewshot, prompt=prompt)

    """ INFERENCE HOOKS """

    def _on_inference_start(self, model: Model, sample: Sample) -> None:
        """
        Hook method called before inference starts. Typically used to prepare the model or sample.
        """
        pass

    def _on_inference(self, model: Model, sample: Sample) -> ModelOutput:
        """
        Hook method called during inference.
        """

        model_output = model.generate(input=sample.input, tools=sample.tools)
        return model_output

    def _on_inference_end(self, model: Model, sample: Sample, model_output: ModelOutput) -> TaskState:
        """
        Hook method called after inference ends.
        """
        return TaskState(
            model=model.model_id,
            sample=sample,
            messages=[model_output.message],
            output=model_output,
            completed=True,
        )

    def run_inference(self, model: Model, sample: Sample) -> TaskState:
        self._on_inference_start(model, sample)
        model_output = self._on_inference(model, sample)
        task_state = self._on_inference_end(model, sample, model_output)

        return task_state

    """ METRIC CALCULATION HOOKS """

    def filter_prediction(self, prediction: str) -> str:
        """
        Hook method called before calculating metrics. Typically filters or prepares the prediction.
        """
        if self.filters:
            # Apply filters to the result
            filter_ensemble = build_filter_ensemble(filters=self.filters)
            prediction = filter_ensemble(prediction)
        return prediction

    def match_score(self, original_prediction: str, filtered_prediction: str, reference: str,
                    task_state: TaskState) -> Score:
        """
        Hook method called to calculate metrics for the task state.
        This method should be overridden in subclasses to implement specific metric calculations.
        """

        # Initialize the score
        score = Score(
            answer=filtered_prediction,
            prediction=original_prediction,
            explanation=original_prediction,
            metadata=task_state.metadata)

        # Calculate scores for each metric
        for metric in self.metric_list:
            metric_scorer = get_metric(metric)
            metric_score = metric_scorer(
                reference=reference,
                prediction=filtered_prediction,
            )
            score.value[metric] = metric_score

        return score

    def calculate_metrics(self, task_state: TaskState) -> SampleScore:
        """
        Calculate evaluation metrics for the given task state.
        Note: There may be multiple choices in the task state output,
            handle by metrics, typically average over choices or calculate pass@k.
        """
        assert task_state.completed, \
            'TaskState must be completed before calculating metrics.'

        prediction = task_state.output.completion
        # Filter or prepare the prediction
        filtered_prediction = self.filter_prediction(prediction)

        # Calculate the score
        score = self.match_score(
            original_prediction=prediction,
            filtered_prediction=filtered_prediction,
            reference=task_state.target,
            task_state=task_state)

        sample_score = SampleScore(
            score=score,
            sample_id=task_state.sample_id,
            group_id=task_state.group_id,
            sample_metadata=task_state.metadata,
        )

        return sample_score

    """ REPORT GENERATION HOOKS """

    def _on_generate_report_start(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """
        Hook method called before generating the report. Typically used to aggregate scores.
        First, aggregate scores by sample group ID.
        Then, aggregate scores by sample ID.
        """
        pass

    def _on_generate_report_end(self, report: Report) -> None:
        """
        Hook method called after generating the report.
        """
        pass

    def _on_generate_report(self, scores: List[AggScore]) -> Report:
        """
        Hook method called during report generation.
        """
        pass

    def generate_report(self, scores: List[SampleScore]) -> Report:
        agg_scores = self._on_generate_report_start(scores)
        report = self._on_generate_report(agg_scores)
        self._on_generate_report_end(report)
        return report
