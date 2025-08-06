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
from evalscope.utils import get_logger
from .benchmark import DataAdapter

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

        Raises:
            AssertionError: If train_split is None when few_shot_num > 0
        """
        if os.path.exists(self.dataset_id):
            # Load dataset from local file system path
            # TODO: Implement local dataset loading logic
            pass
        else:
            # Load dataset from remote source (e.g., HuggingFace Hub)
            self.test_dataset = self.load_subsets(self.load_subset)

            # Load few-shot examples if few-shot prompting is enabled
            if self.few_shot_num > 0:
                assert self.train_split is not None, \
                    'Train split must be specified for few-shot prompting.'
                self.fewshot_dataset = self.load_subsets(self.load_fewshot_subset)

        # Process each sample's input by applying prompt templates and few-shot formatting
        for subset in self.test_dataset.keys():
            for sample in self.test_dataset[subset]:
                sample.input = self.process_sample_input(sample, subset=subset)
        return self.test_dataset

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
            # Retrieve few-shot examples for the current subset
            few_shot_samples = self.fewshot_dataset.get(subset)
            if few_shot_samples is None:
                # Fallback: use the first available subset if current subset not found
                first_key = next(iter(self.fewshot_dataset))
                few_shot_samples = self.fewshot_dataset[first_key]

            # Convert few-shot samples to demonstration string
            few_shot = '\n\n'.join([self.sample_to_fewshot(sample) for sample in few_shot_samples])

            # Format the input text with few-shot examples and main prompt
            input_text = self.format_fewshot_template(fewshot=few_shot, prompt=sample.input)
        else:
            # No few-shot examples: use the prompt template directly
            input_text = self.format_prompt_template(prompt=sample.input)
        return input_text

    def load_subsets(self, load_func: Callable[[str], Dataset]) -> DatasetDict:
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
            # Load only the default subset and create a single-entry DatasetDict
            subset_data = load_func(self.default_subset)
            dataset_dict = DatasetDict.from_dataset(subset_data)
        else:
            # Load all specified subsets into separate entries
            subset_dict = defaultdict()
            for subset in self.subset_list:
                subset_data = load_func(subset)
                subset_dict[subset] = subset_data
            dataset_dict = DatasetDict(subset_dict)
        return dataset_dict

    def load_subset(self, subset: str) -> Dataset:
        """
        Load a specific subset of the dataset for evaluation.

        This method configures and executes the data loading for a single subset,
        handling both split-as-subset and traditional subset configurations.

        Args:
            subset (str): The subset identifier to load

        Returns:
            Dataset: The loaded dataset subset with processed samples
        """
        # Determine the split and subset names based on configuration
        split = subset if self.split_as_subset else self.eval_split
        subset_name = self.default_subset if self.split_as_subset else subset

        # Create and configure the remote data loader
        loader = RemoteDataLoader(
            data_id_or_path=self.dataset_id,
            split=split,
            subset=subset_name,
            sample_fields=self.record_to_sample,  # Custom sample conversion function
            limit=self._task_config.limit,  # Limit number of samples if specified
            repeats=self._task_config.repeats,  # Number of repetitions for each sample
            data_source=self._task_config.dataset_hub,  # Data source configuration
        )
        return loader.load()

    def load_fewshot_subset(self, subset: str) -> Dataset:
        """
        Load a subset specifically for few-shot examples.

        This method loads training data to be used as demonstrations in few-shot prompting.
        It typically loads from the training split with limited samples and optional shuffling.

        Args:
            subset (str): The subset identifier to load few-shot examples from

        Returns:
            Dataset: The loaded few-shot dataset with demonstration examples
        """
        # Use training split for few-shot examples
        split = subset if self.split_as_subset else self.train_split
        subset_name = self.default_subset if self.split_as_subset else subset

        # Create loader specifically configured for few-shot sampling
        loader = RemoteDataLoader(
            data_id_or_path=self.dataset_id,
            split=split,
            subset=subset_name,
            sample_fields=self.record_to_sample,
            limit=self.few_shot_num,  # Limit to specified number of few-shot examples
            shuffle=self.few_shot_random,  # Randomize selection if enabled
            data_source=self._task_config.dataset_hub,
        )
        return loader.load()

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a raw data record to a Sample object.

        This method must be implemented in subclasses to handle dataset-specific
        field mapping and data processing logic.

        Args:
            record (Dict[str, Any]): Raw data record from the dataset

        Returns:
            Sample: Processed sample object ready for evaluation

        Raises:
            NotImplementedError: This method must be implemented in subclasses
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

        Raises:
            NotImplementedError: This method must be implemented in subclasses
        """
        raise NotImplementedError('This method should be implemented in subclasses')

    def format_prompt_template(self, prompt: str) -> str:
        """
        Format the basic prompt template with the sample data.

        This method applies the prompt template to format the input text
        for models when no few-shot examples are used.

        Args:
            prompt (str): The raw prompt text to format

        Returns:
            str: The formatted prompt ready for model input
        """
        return self.prompt_template.format(prompt=prompt)

    def format_fewshot_template(self, fewshot: str, prompt: str) -> str:
        """
        Format the few-shot template with demonstrations and the main prompt.

        This method combines few-shot examples with the main prompt using
        the configured few-shot template.

        Args:
            fewshot (str): The formatted few-shot demonstration examples
            prompt (str): The main prompt for the current sample

        Returns:
            str: The complete formatted input with few-shot context
        """
        return self.few_shot_prompt_template.format(fewshot=fewshot, prompt=prompt)

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

    def _on_inference_end(self, model: Model, sample: Sample, model_output: ModelOutput, output_dir: str,
                          **kwargs) -> TaskState:
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

    def match_score(self, original_prediction: str, filtered_prediction: str, reference: str,
                    task_state: TaskState) -> Score:
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
                metric_scorer = get_metric(metric)  # Get metric implementation from registry
                metric_func = metric_scorer()  # Instantiate the metric scorer
                metric_score = metric_func(
                    prediction=filtered_prediction,
                    reference=reference,
                )
                score.value[metric] = metric_score
            except Exception as e:
                logger.error(f'Error calculating metric {metric}: {e}')
                return None

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
        prediction = task_state.output.completion

        # Apply filtering and answer extraction
        filtered_prediction = self.filter_prediction(prediction, task_state)

        # Step 1: Calculate standard metric scores (rule-based)
        rule_based_score = self.match_score(
            original_prediction=prediction,
            filtered_prediction=filtered_prediction,
            reference=task_state.target,
            task_state=task_state)

        # Step 2: Apply LLM judge if enabled and get final score
        final_score = self.maybe_llm_match_score(
            original_prediction=prediction,
            filtered_prediction=filtered_prediction,
            reference=task_state.target,
            task_state=task_state,
            rule_based_score=rule_based_score)

        # Package the results into a sample score object
        sample_score = SampleScore(
            score=final_score,
            sample_id=task_state.sample_id,
            group_id=task_state.group_id,
            sample_metadata=task_state.metadata,
        )

        return sample_score

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
        return ReportGenerator.generate_report(score_dict=scores, model_name=model_name, data_adapter=self)

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
