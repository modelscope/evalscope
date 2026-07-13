from typing import Literal

BodyComposeMode = Literal['override', 'fill', 'passthrough']

_VALID_COMPOSE_MODES = {'override', 'fill', 'passthrough'}


class AnnotatedBody(dict):
    """Dict subclass carrying a ``compose_mode`` attribute.

    Flows transparently through any code path that accepts ``dict``.
    """

    def __init__(self, *args, compose_mode: BodyComposeMode = 'override', **kwargs):
        if compose_mode not in _VALID_COMPOSE_MODES:
            raise ValueError(
                f'Invalid compose_mode: {compose_mode!r}. '
                f'Must be one of: {", ".join(sorted(_VALID_COMPOSE_MODES))}'
            )
        super().__init__(*args, **kwargs)
        self.compose_mode = compose_mode
