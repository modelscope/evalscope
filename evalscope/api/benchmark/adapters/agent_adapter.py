from .default_data_adapter import DefaultDataAdapter


class AgentAdapter(DefaultDataAdapter):
    """Adapter for agent benchmarks. e.g., function calling, etc."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
