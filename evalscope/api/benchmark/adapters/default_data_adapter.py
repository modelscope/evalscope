import os
from collections import defaultdict
from functools import partial
from overrides import override
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from evalscope.api.dataset import DataLoader, Dataset, DatasetDict, LocalDataLoader, RemoteDataLoader, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessage, ChatMessageSystem, ChatMessageUser
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.model import Model, ModelOutput
from evalscope.api.registry import get_aggregation, get_metric
from evalscope.constants import HubType, JudgeStrategy
from evalscope.report import Report, ReportGenerator
from evalscope.utils import get_logger
from ..benchmark import DataAdapter

logger = get_logger()


class DefaultDataAdapter(DataAdapter):
    """
    Default Data Adapter for the benchmark evaluation system.

    This class serves as the base implementation for data adapters that handle:
    - Dataset loading and preprocessing
    - Model inference execution
    - Metric calculation and aggregation
    - Report generation

    The adapter follows a pipeline architecture with hooks that can be overridden
    in subclasses to customize behavior for specific benchmarks or evaluation tasks.

    Key responsibilities:
    1. Load datasets with optional few-shot examples
    2. Process samples and format prompts
    3. Execute model inference with proper state management
    4. Calculate evaluation metrics and aggregate results
    5. Generate comprehensive evaluation reports

    This class can be extended to implement specific data loading and processing
    logic for different benchmark datasets and evaluation scenarios.
    """

    # ####################
    # DATA LOADING METHODS
    # ####################

    @override
    def load_dataset(self) -> DatasetDict:
        """
        Load the complete dataset including test data and optional few-shot examples.

        This method handles both local and remote dataset loading, processes samples
        with appropriate prompt formatting, and prepares few-shot examples if needed.

        Returns:
            DatasetDict: A dictionary containing the loaded and processed datasets,
                        organized by subset names.
        """
        # Load the dataset
        self.test_dataset, self.fewshot_dataset = self.load()

        # Process each sample's input by applying prompt templates and few-shot formatting
        self._post_process_samples()

        return self.test_dataset

    def load(self) -> Tuple[DatasetDict, Optional[DatasetDict]]:
        """Load the dataset from disk or remote source.

        Returns:
            Tuple[DatasetDict, Optional[DatasetDict]]: The test dataset and few-shot dataset.
        """
        if os.path.exists(self.dataset_id):
            # Load dataset from local file system path
            with self._temporary_attribute('dataset_hub', HubType.LOCAL):
                return self.load_from_disk()
        else:
            # Load dataset from remote source (e.g., ModelScope, Huggingface)
            return self.load_from_remote()

    def load_from_remote(self):
        """Load dataset from remote source and prepare few-shot examples if needed."""
        test_dataset = None
        fewshot_dataset = None
        # Load dataset from remote source
        test_load_func = partial(self.load_subset, data_loader=RemoteDataLoader)
        test_dataset = self.load_subsets(test_load_func)

        # Load few-shot examples if few-shot prompting is enabled
        if self._should_load_fewshot():
            fewshot_load_func = partial(self.load_fewshot_subset, data_loader=RemoteDataLoader)
            fewshot_dataset = self.load_subsets(fewshot_load_func, is_fewshot=True)
        return test_dataset, fewshot_dataset

    def load_from_disk(self, use_local_loader: bool = False):
        """
        Load dataset from local disk path.

        Args:
            use_local_loader: If True, use local file loading; otherwise use remote loading
                             for local ModelScope datasets.
        """
        test_dataset = None
        fewshot_dataset = None
        if use_local_loader:
            # Use LocalDataLoader for actual local file loading
            test_load_func = partial(self.load_subset, data_loader=LocalDataLoader)
            test_dataset = self.load_subsets(test_load_func)

            # Load few-shot examples if few-shot prompting is enabled
            if self._should_load_fewshot():
                fewshot_load_func = partial(self.load_fewshot_subset, data_loader=LocalDataLoader)
                fewshot_dataset = self.load_subsets(fewshot_load_func, is_fewshot=True)
            return test_dataset, fewshot_dataset
        else:
            # Fallback to remote loading for local ModelScope datasets
            return self.load_from_remote()

    def _should_load_fewshot(self) -> bool:
        """Check if few-shot dataset should be loaded."""
        return self.few_shot_num > 0 and self.train_split is not None

    def _post_process_samples(self):
        """Process all sample inputs with prompt formatting."""
        for subset in self.test_dataset.keys():
            for sample in self.test_dataset[subset]:
                if isinstance(sample.input, str):
                    sample.input = self.process_sample_str_input(sample, subset)
                elif isinstance(sample.input, list):
                    # Handle list[ChatMessage] and add system prompt if needed
                    sample.input = self.process_sample_messages_input(sample, subset)

    def process_sample_str_input(self, sample: Sample, subset: str) -> List[ChatMessage]:
        """
        Convert a sample's input string to a list of ChatMessage objects.

        This method formats the sample input into a structured message format
        suitable for model inference, including system prompts if configured.
        """
        input_text = self.process_sample_input(sample, subset=subset)
        input_messages = [ChatMessageUser(content=input_text)]
        if self.system_prompt:
            input_messages.insert(0, ChatMessageSystem(content=self.system_prompt))
        return input_messages

    def process_sample_messages_input(self, sample: Sample, subset: str) -> List[ChatMessage]:
        """
        Normalize a sample's existing List[ChatMessage] input and ensure system prompt is set once.
        """
        messages = list(sample.input)  # shallow copy to avoid in-place mutations
        if self.system_prompt and not any(isinstance(m, ChatMessageSystem) for m in messages):
            messages = [ChatMessageSystem(content=self.system_prompt)] + messages
        return messages

    def process_sample_input(self, sample: Sample, subset: str) -> str:
        """
        Process a single sample's input by applying prompt templates and few-shot formatting.

        This method handles the complete input preparation pipeline:
        1. Retrieves few-shot examples if enabled
        2. Formats few-shot examples into demonstration text
        3. Applies appropriate prompt template (with or without few-shot context)

        Args:
            sample (Sample): The sample to process
            subset (str): The subset name this sample belongs to

        Returns:
            str: The formatted input text ready for model inference
        """
        if self.few_shot_num > 0:
            if self.fewshot_dataset is not None:
                # Retrieve few-shot examples for the current subset
                few_shot_samples = self.fewshot_dataset.get(subset)
                if few_shot_samples is None:
                    # Fallback: use the first available subset if current subset not found
                    first_key = next(iter(self.fewshot_dataset))
                    few_shot_samples = self.fewshot_dataset[first_key]
                # Select fewshot samples
                assert len(few_shot_samples) >= self.few_shot_num, (
                    f"""The dataset only have ({len(few_shot_samples)}) few-shot samples, but requested ({self.few_shot_num}) fewshot samples, please reduce 'few_shot_num'."""  # noqa: E501
                )
                # Convert few-shot samples to demonstration string
                few_shot = '\n\n'.join([self.sample_to_fewshot(sample) for sample in few_shot_samples])
            else:
                # Build few-shot examples inside the format method
                few_shot = ''
            # Format the input text with few-shot examples and main prompt
            input_text = self.format_fewshot_template(fewshot=few_shot, sample=sample)
        else:
            # No few-shot examples: use the prompt template directly
            input_text = self.format_prompt_template(sample=sample)
        return input_text

    def load_subsets(self, load_func: Callable[[str], Dataset], is_fewshot=False) -> DatasetDict:
        """
        Load multiple subsets of the dataset using the provided loading function.

        This method handles two loading strategies:
        1. Reformat mode: Load only the default subset and reformat it
        2. Multi-subset mode: Load all subsets specified in subset_list

        Args:
            load_func (Callable[[str], Dataset]): Function to load individual subsets

        Returns:
            DatasetDict: Dictionary containing all loaded subsets
        """
        if self.reformat_subset:
            # Load only the default subset
            subset_data = load_func(self.default_subset)
            # Reformat the subset to create multiple subsets based on sample keys
            # NOTE: subset_list and limit is applied here if specified
            limit = self.few_shot_num if is_fewshot else self.limit
            repeats = 1 if is_fewshot else self.repeats
            dataset_dict = DatasetDict.from_dataset(
                dataset=subset_data, subset_list=self.subset_list, limit=limit, repeats=repeats
            )
        else:
            # Load all specified subsets into separate entries
            subset_dict = defaultdict()
            for subset in self.subset_list:
                # Set current subset, since same benchmark need to differentiate
                with self._temporary_attribute('current_subset_name', subset):
                    subset_data = load_func(subset)
                    subset_dict[subset] = subset_data
            dataset_dict = DatasetDict(subset_dict)
        return dataset_dict

    def load_subset(self, subset: str, data_loader: Type[DataLoader]) -> Dataset:
        """
        Load a specific subset of the dataset for evaluation.

        Args:
            subset (str): The subset identifier to load
            data_loader (Type[DataLoader]): The data loader class to use for loading

        Returns:
            Dataset: The loaded dataset subset with processed samples
        """
        # Determine the split and subset names based on configuration
        split = subset if self.split_as_subset else self.eval_split
        subset_name = self.default_subset if self.split_as_subset else subset

        # Create and configure the remote data loader
        loader = data_loader(
            data_id_or_path=self.dataset_id,
            split=split,
            subset=subset_name,
            sample_fields=self.record_to_sample,  # Custom sample conversion function
            filter_func=self.sample_filter,
            limit=self.limit if not self.reformat_subset else None,  # Limit number of samples if specified
            repeats=self.repeats,  # Number of repetitions for each sample
            shuffle=self.shuffle,  # Shuffle dataset if enabled
            shuffle_choices=self.shuffle_choices,  # Shuffle choices if requested
            data_source=self.dataset_hub,  # Data source configuration
            force_redownload=self.force_redownload,  # Force redownload if enabled
            dataset_dir=self.dataset_dir,  # Dataset directory
        )
        dataset = loader.load()
        return dataset

    def load_fewshot_subset(self, subset: str, data_loader: Type[DataLoader]) -> Dataset:
        """
        Load a subset specifically for few-shot examples.

        Args:
            subset (str): The subset identifier to load few-shot examples from
            data_loader (Type[DataLoader]): The data loader class to use for loading

        Returns:
            Dataset: The loaded few-shot dataset with demonstration examples
        """
        # Use training split for few-shot examples
        split = subset if self.split_as_subset else self.train_split
        subset_name = self.default_subset if self.split_as_subset else subset

        # Create loader specifically configured for few-shot sampling
        loader = data_loader(
            data_id_or_path=self.dataset_id,
            split=split,
            subset=subset_name,
            sample_fields=self.record_to_sample,
            filter_func=self.sample_filter,  # Apply sample filtering if defined
            limit=self.few_shot_num
            if not self.reformat_subset else None,  # Limit to specified number of few-shot examples
            shuffle=self.few_shot_random,  # Randomize selection if enabled
            shuffle_choices=self.shuffle_choices,  # Shuffle choices if requested
            data_source=self.dataset_hub,  # Data source configuration
            force_redownload=self.force_redownload,  # Force redownload if enabled
            dataset_dir=self.dataset_dir,  # Dataset directory
        )
        dataset = loader.load()
        return dataset

    def sample_filter(self, sample: Sample) -> bool:
        """
        Apply filtering to a dataset, only samples matching the predicate will be included.

        Args:
            sample (Sample): The sample to filter

        Returns:
            bool: True if the sample passes the filter, False otherwise
        """
        return True  # Default implementation allows all samples

    def record_to_sample(self, record: Dict[str, Any]) -> Union[Sample, List[Sample]]:
        """
        Convert a raw data record to a Sample object.

        This method must be implemented in subclasses to handle dataset-specific
        field mapping and data processing logic.

        Args:
            record (Dict[str, Any]): Raw data record from the dataset

        Returns:
            Union[Sample, List[Sample]]: Processed sample object(s) ready for evaluation
        """
        raise NotImplementedError('This method should be implemented in subclasses')

    def sample_to_fewshot(self, sample: Sample) -> str:
        """
        Convert a Sample object to a formatted few-shot demonstration string.

        This method must be implemented in subclasses to define how samples
        are formatted as examples in few-shot prompts.

        Args:
            sample (Sample): The sample to convert to a few-shot example

        Returns:
            str: Formatted few-shot demonstration string
        """
        raise NotImplementedError('This method should be implemented in subclasses')

    def format_prompt_template(self, sample: Sample) -> str:
        """
        Format the basic prompt template with the sample data.

        This method applies the prompt template to format the input text
        for models when no few-shot examples are used.

        Args:
            sample (Sample): The sample object containing the prompt data

        Returns:
            str: The formatted prompt ready for model input
        """
        return self.prompt_template.format(question=sample.input)

    def format_fewshot_template(self, fewshot: str, sample: Sample) -> str:
        """
        Format the few-shot template with demonstrations and the main prompt.

        This method combines few-shot examples with the main prompt using
        the configured few-shot template.

        Args:
            fewshot (str): The formatted few-shot demonstration examples
            sample (Sample): The sample object containing the prompt data

        Returns:
            str: The complete formatted input with few-shot context
        """
        return self.few_shot_prompt_template.format(fewshot=fewshot, question=sample.input)

    # #################
    # INFERENCE METHODS
    # #################

    def _on_inference_start(self, model: Model, sample: Sample) -> None:
        """
        Hook method called before inference starts.

        This method can be overridden in subclasses to implement custom
        preparation logic before model inference (e.g., model configuration,
        sample preprocessing, state initialization).

        Args:
            model (Model): The model that will perform inference
            sample (Sample): The sample to be processed
        """
        pass

    def _on_inference(self, model: Model, sample: Sample) -> ModelOutput:
        """
        Hook method called during the actual inference process.

        This method executes the model inference and can be overridden
        to implement custom inference logic or model interaction patterns.

        Args:
            model (Model): The model to use for inference
            sample (Sample): The sample to process

        Returns:
            ModelOutput: The raw output from the model
        """
        # Execute model inference with the processed input and any tools
        model_output = model.generate(input=sample.input, tools=sample.tools)
        return model_output

    def _on_inference_end(
        self, model: Model, sample: Sample, model_output: ModelOutput, output_dir: str, **kwargs
    ) -> TaskState:
        """
        Hook method called after inference completes.

        This method processes the model output and creates a TaskState object
        that encapsulates all information about the completed inference task.
        You can save the model output to the specified output directory.

        Args:
            model (Model): The model that performed inference
            sample (Sample): The processed sample
            model_output (ModelOutput): The raw model output
            output_dir (str): The directory where the model output was saved

        Returns:
            TaskState: Complete state object for the inference task
        """
        return TaskState(
            model=model.name,
            sample=sample,
            messages=[model_output.message],
            output=model_output,
            completed=True,
        )

    @override
    def run_inference(self, model: Model, sample: Sample, output_dir: str, **kwargs) -> TaskState:
        """
        Execute the complete inference pipeline for a single sample.

        This method orchestrates the full inference process using the hook methods:
        1. Pre-inference preparation
        2. Model inference execution
        3. Post-inference processing and state creation

        Args:
            model (Model): The model to use for inference
            sample (Sample): The sample to process
            output_dir (str): The directory to store the generated files

        Returns:
            TaskState: Complete state object containing inference results
        """
        self._on_inference_start(model, sample)
        model_output = self._on_inference(model, sample)
        task_state = self._on_inference_end(model, sample, model_output, output_dir, **kwargs)

        return task_state

    # ##########################
    # METRIC CALCULATION METHODS
    # ##########################

    def filter_prediction(self, prediction: str, task_state: TaskState) -> str:
        """
        Filter and prepare the model prediction for metric calculation.

        This method applies configured filters and custom answer extraction
        to clean and prepare the raw model output for evaluation.

        Args:
            prediction (str): The raw model prediction
            task_state (TaskState): The complete task state for context

        Returns:
            str: The filtered and extracted prediction ready for evaluation
        """
        if self.filter_ensemble is not None:
            # Apply configured filters to clean the prediction
            prediction = self.filter_ensemble(prediction)

        # Apply custom answer extraction logic
        extracted_prediction = self.extract_answer(prediction, task_state)
        return extracted_prediction

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """
        Hook method for custom answer extraction from model predictions.

        This method can be overridden in subclasses to implement specific
        logic for extracting the final answer from complex model outputs.

        Args:
            prediction (str): The model prediction to extract from
            task_state (TaskState): The task state for additional context

        Returns:
            str: The extracted answer
        """
        return prediction

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """
        Calculate evaluation scores by comparing prediction with reference.

        This method computes scores using all configured metrics and creates
        a comprehensive Score object with detailed evaluation results.

        Args:
            original_prediction (str): The original, unfiltered model prediction
            filtered_prediction (str): The filtered and processed prediction
            reference (str): The ground truth reference answer
            task_state (TaskState): The complete task state for context

        Returns:
            Score: Object containing all calculated metric scores and metadata
        """
        # Initialize the score object with prediction details
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        # Calculate scores for each configured metric
        for metric in self.metric_list:
            try:
                if isinstance(metric, str):
                    metric_name = metric
                    metric_scorer = get_metric(metric)  # Get metric implementation from registry
                    metric_func = metric_scorer()  # Instantiate the metric scorer
                elif isinstance(metric, dict):
                    metric_name = list(metric.keys())[0]
                    metric_cls = get_metric(metric_name)
                    metric_func = metric_cls(**metric[metric_name])  # Initialize with parameters
                metric_score = metric_func(
                    prediction=filtered_prediction,
                    reference=reference,
                )
                score.value[metric_name] = metric_score
            except Exception as e:
                logger.error(f'Error calculating metric {metric}: {e}')
                score.value[metric_name] = 0
                score.metadata[metric_name] = f'error: {str(e)}'

        return score

    @override
    def calculate_metrics(self, task_state: TaskState) -> SampleScore:
        """
        Calculate comprehensive evaluation metrics for a completed task.

        This method processes the task state to extract predictions, applies
        filtering and answer extraction, calculates all configured metrics,
        and packages the results into a SampleScore object.

        Args:
            task_state (TaskState): The completed task state to evaluate

        Returns:
            SampleScore: Complete scoring results for the sample

        Raises:
            AssertionError: If the task state is not marked as completed
        """
        assert task_state.completed, \
            'TaskState must be completed before calculating metrics.'

        # Extract the raw prediction from the model output
        if task_state.output is None:
            prediction = ''
        else:
            prediction = task_state.output.completion

        # Apply filtering and answer extraction
        filtered_prediction = self.filter_prediction(prediction, task_state)

        if self.judge_strategy == JudgeStrategy.LLM_RECALL:
            # Step 1: Calculate standard metric scores (rule-based)
            rule_based_score = self.match_score(
                original_prediction=prediction,
                filtered_prediction=filtered_prediction,
                reference=task_state.target,
                task_state=task_state
            )

            # Step 2: Apply LLM judge if enabled and get final score
            final_score = self.maybe_llm_match_score(
                original_prediction=prediction,
                filtered_prediction=filtered_prediction,
                reference=task_state.target,
                task_state=task_state,
                rule_based_score=rule_based_score
            )
        else:
            if self.use_llm_judge:
                # Use LLM judge to compute the match score directly
                final_score = self.llm_match_score(
                    original_prediction=prediction,
                    filtered_prediction=filtered_prediction,
                    reference=task_state.target,
                    task_state=task_state
                )
            else:
                # Use standard match score calculation without LLM judge
                final_score = self.match_score(
                    original_prediction=prediction,
                    filtered_prediction=filtered_prediction,
                    reference=task_state.target,
                    task_state=task_state
                )

        # Package the results into a sample score object
        sample_score = SampleScore(
            score=final_score,
            sample_id=task_state.sample_id,
            group_id=task_state.group_id,
            sample_metadata=task_state.metadata,
        )

        return sample_score

    def batch_match_score(
        self, original_predictions: List[str], filtered_predictions: List[str], references: List[str],
        task_states: List[TaskState]
    ) -> Optional[List[Score]]:
        """
        Batch calculate evaluation scores by comparing predictions with references.

        This method computes scores using all configured metrics for a batch of samples
        and creates a list of Score objects with detailed evaluation results.

        Args:
            original_predictions (List[str]): The original, unfiltered model predictions
            filtered_predictions (List[str]): The filtered and processed predictions
            references (List[str]): The ground truth reference answers
            task_states (List[TaskState]): The complete task states for context

        Returns:
            List[Score]: List of objects containing all calculated metric scores and metadata
        """
        return None  # Default implementation does not support batch scoring

    @override
    def batch_calculate_metrics(self, task_states: List[TaskState],
                                sample_scores: List[SampleScore]) -> List[SampleScore]:
        """Batch calculate metrics for a list of task states with tqdm progress and batch processing."""
        total = len(task_states)
        if total == 0:
            return sample_scores

        # Prepare lists for batch processing
        original_predictions: List[str] = []
        filtered_predictions: List[str] = []
        references: List[str] = []

        for ts in task_states:
            pred = ts.output.completion
            original_predictions.append(pred)
            filtered_predictions.append(self.filter_prediction(pred, ts))
            references.append(ts.target)

        batch_scores = self.batch_match_score(
            original_predictions=original_predictions,
            filtered_predictions=filtered_predictions,
            references=references,
            task_states=task_states
        )

        if batch_scores is not None:
            assert len(batch_scores) == len(sample_scores), \
                'Batch scores length must match sample scores length.'
            for batch_score, sample_score in zip(batch_scores, sample_scores):
                sample_score.score.value.update(batch_score.value)

        return sample_scores

    @override
    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """
        Aggregate individual sample scores into summary statistics.

        This method uses the configured aggregation method to compute
        summary statistics (e.g., mean, median, percentiles) across
        all sample scores for comprehensive evaluation results.

        Args:
            sample_scores (List[SampleScore]): Individual scores for all samples

        Returns:
            List[AggScore]: Aggregated scores and statistics
        """
        # Get the configured aggregation implementation
        aggregate_cls = get_aggregation(self.aggregation)
        aggregator = aggregate_cls()

        # Compute aggregated scores
        agg_scores = aggregator(sample_scores)

        return agg_scores

    # #########################
    # REPORT GENERATION METHODS
    # #########################

    def _on_generate_report_end(self, report: Report, output_dir: str, **kwargs) -> None:
        """
        Hook method called after generating the evaluation report.

        This method can be overridden in subclasses to implement custom
        post-processing of the generated report (e.g., additional formatting,
        custom visualizations, external integrations).

        Args:
            report (Report): The generated evaluation report
            output_dir (str): Directory where the report should be saved
        """
        pass

    def _on_generate_report(self, scores: Dict[str, List[AggScore]], model_name: str) -> Report:
        """
        Hook method called during report generation.

        This method creates the evaluation report using the configured
        report generator and can be overridden to implement custom
        report generation logic.

        Args:
            scores (Dict[str, List[AggScore]]): Aggregated scores organized by subset
            model_name (str): Name of the evaluated model

        Returns:
            Report: The generated evaluation report
        """
        return ReportGenerator.generate_report(
            score_dict=scores, model_name=model_name, data_adapter=self, add_aggregation_name=self.add_aggregation_name
        )

    @override
    def generate_report(self, scores: Dict[str, List[AggScore]], model_name: str, output_dir: str, **kwargs) -> Report:
        """
        Generate a comprehensive evaluation report from aggregated scores.

        This method orchestrates the complete report generation process:
        1. Creates the report using configured generators
        2. Applies any post-processing through hook methods

        Args:
            scores (Dict[str, List[AggScore]]): Aggregated scores by subset name
            model_name (str): Name of the model being evaluated

        Returns:
            Report: Complete evaluation report with results and analysis
        """
        report = self._on_generate_report(scores, model_name=model_name)
        self._on_generate_report_end(report, output_dir, **kwargs)
        return report

    def finalize(self, *args, **kwargs):
        # Finalize the evaluation process
        self.sandbox_finalize(*args, **kwargs)
