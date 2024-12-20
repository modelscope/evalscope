class EvaluatorCollection:

    def __init__(self, schema):
        self.schema = schema
        self.evaluators = {}

    def add_evaluator(self, dataset_name):
        evaluator = self.schema.get_evaluator(dataset_name)
        if evaluator:
            self.evaluators[dataset_name] = evaluator

    def evaluate(self, dataset_name, sample):
        evaluator = self.evaluators.get(dataset_name)
        if evaluator:
            return evaluator.evaluate(sample)
