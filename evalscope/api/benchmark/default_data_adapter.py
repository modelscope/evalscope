import os
from collections import defaultdict
from overrides import override
from typing import Any, Callable, Dict, List

from evalscope.api.dataset import Dataset, DatasetDict, RemoteDataLoader, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.model import Model, ModelOutput
from evalscope.api.registry import get_aggregation, get_metric
from evalscope.report import Report, ReportGenerator
from .benchmark import DataAdapter


class DefaultDataAdapter(DataAdapter):
    """
    Default Data Adapter for the benchmark.
    This class can be extended to implement specific data loading and processing logic.
    """

    # ####################
    # DATA LOADING METHODS
    # ####################

    @override
    def load_dataset(self) -> DatasetDict:
        if os.path.exists(self.dataset_id):
            # Load dataset from local path
            pass
        else:
            # Load dataset from remote source
            self.test_dataset = self.load_subsets(self.load_subset)

            if self.few_shot_num > 0:
                assert self.train_split is not None, \
                    'Train split must be specified for few-shot prompting.'
                self.fewshot_dataset = self.load_subsets(self.load_fewshot_subset)

        # process the sample input
        for subset in self.test_dataset.keys():
            for sample in self.test_dataset[subset]:
                sample.input = self.process_sample_input(sample, subset=subset)
        return self.test_dataset

    def process_sample_input(self, sample: Sample, subset: str) -> str:
        if self.few_shot_num > 0:
            few_shot_samples = self.fewshot_dataset.get(subset)
            if few_shot_samples is None:
                # Use the first key if subset is not found
                first_key = next(iter(self.fewshot_dataset))
                few_shot_samples = self.fewshot_dataset[first_key]
            # join few-shot samples into a string
            few_shot = '\n\n'.join([self.sample_to_fewshot(sample) for sample in few_shot_samples])
            # Format the input text with few-shot examples
            input_text = self.format_fewshot_template(fewshot=few_shot, prompt=sample.input)
        else:
            # No few-shot examples, use the prompt template directly
            input_text = self.format_query_template(prompt=sample.input)
        return input_text

    def load_subsets(self, load_func: Callable[[str], Dataset]) -> DatasetDict:
        """
        Load subsets of the dataset using the provided load function.
        This method handles the case where the dataset is split into multiple subsets.
        If `reformat_subset` is True, it loads the default subset only.
        Otherwise, it loads all subsets specified in `subset_list`.
        """
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
            sample_fields=self.record_to_sample,
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
            sample_fields=self.record_to_sample,
            limit=self.few_shot_num,
            shuffle=self.few_shot_random,
            data_source=self._task_config.dataset_hub,
        )
        return loader.load()

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a record dictionary to a Sample object.
        This method should be overridden in subclasses to implement specific data processing logic.
        """
        raise NotImplementedError('This method should be implemented in subclasses')

    def sample_to_fewshot(self, sample: Sample) -> str:
        """
        Convert a Sample object to a few-shot example string.
        This method should be overridden in subclasses to implement specific few-shot formatting logic.
        """
        raise NotImplementedError('This method should be implemented in subclasses')

    def format_query_template(self, prompt: str) -> str:
        """
        Format the prompt template with the sample data.
        """
        return self.query_template.format(prompt=prompt)

    def format_fewshot_template(self, fewshot: str, prompt: str) -> str:
        """
        Format the few-shot template with the few-shot examples.
        """
        return self.few_shot_prompt_template.format(fewshot=fewshot, prompt=prompt)

    # #################
    # INFERENCE METHODS
    # #################

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

    @override
    def run_inference(self, model: Model, sample: Sample) -> TaskState:
        self._on_inference_start(model, sample)
        model_output = self._on_inference(model, sample)
        task_state = self._on_inference_end(model, sample, model_output)

        return task_state

    # ##########################
    # METRIC CALCULATION METHODS
    # ##########################

    def filter_prediction(self, prediction: str) -> str:
        """
        Hook method called before calculating metrics. Typically filters or prepares the prediction.
        """
        if self.filter_ensemble is not None:
            # Apply filters to the result
            prediction = self.filter_ensemble(prediction)
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
            metric_func = metric_scorer()
            metric_score = metric_func(
                prediction=filtered_prediction,
                reference=reference,
            )
            score.value[metric] = metric_score

        return score

    @override
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

    # #########################
    # REPORT GENERATION METHODS
    # #########################
    @override
    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """
        Hook method called before generating the report. Typically used to aggregate scores.
        """
        aggregate_cls = get_aggregation(self.aggregation)
        aggregator = aggregate_cls()
        agg_scores = aggregator(sample_scores)

        return agg_scores

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

    @override
    def generate_report(self, scores: List[AggScore]) -> Report:
        report = self._on_generate_report(scores)
        self._on_generate_report_end(report)
        return report
