from transformers import GenerationConfig


def fix_do_sample_warning(generation_config: GenerationConfig) -> None:
    # Use the default values of temperature/top_p/top_k in generation_config.
    if generation_config.temperature == 0:
        generation_config.do_sample = False
    if generation_config.do_sample is False:
        generation_config.temperature = 1.
        generation_config.top_p = 1.
        generation_config.top_k = 50
