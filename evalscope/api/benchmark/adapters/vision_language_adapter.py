from .default_data_adapter import DefaultDataAdapter


class VisionLanguageAdapter(DefaultDataAdapter):
    """Adapter for vision-language benchmarks. e.g., image captioning, visual question answering, etc."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
