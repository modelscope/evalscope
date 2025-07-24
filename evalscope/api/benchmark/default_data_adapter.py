from .base import DataAdapter


class DefaultDataAdapter(DataAdapter):
    """
    Default Data Adapter for the benchmark.
    This class can be extended to implement specific data loading and processing logic.
    """

    def load_dataset(self):
        # Implement dataset loading logic here
        pass

    def generate_prompts(self):
        # Implement prompt generation logic here
        pass

    def run_inference(self):
        # Implement inference logic here
        pass

    def evaluate(self):
        # Implement evaluation logic here
        pass

    def generate_report(self):
        # Implement report generation logic here
        pass
