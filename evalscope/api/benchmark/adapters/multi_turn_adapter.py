# Copyright (c) Alibaba, Inc. and its affiliates.
"""Template-method base class for multi-turn conversational benchmarks.

Multi-turn benchmarks (e.g. Multi-IF) all share the same inference skeleton:

    for turn in range(max_turns):
        prompt = build_turn_prompt(...)
        if prompt is None: break
        history.append(prompt)
        output = model.generate(history, tools=sample.tools)
        history.append(output.message)
        on_turn_complete(...)

Without a common base, each benchmark re-implements this loop and its
per-turn bookkeeping, leading to divergent behavior (e.g. different ways of
stashing per-turn records into sample.metadata).

``MultiTurnAdapter`` factories this loop once. Subclasses only need to:

* ``build_turn_prompt(sample, history, turn_index)`` — produce the user
  prompt for turn ``turn_index`` given the conversation so far. Return
  ``None`` to signal end-of-conversation.
* ``on_turn_complete(sample, turn_index, model_output, history)`` (optional)
  — inspect the just-produced assistant message, stash per-turn info into
  ``sample.metadata``, etc.

Since ``perf_metrics`` now lives on ``ChatMessage`` (populated by ModelAPI
implementations), the base loop does not need any explicit perf-accumulation
hook: each assistant message in ``history`` automatically carries its own
``perf_metrics``, and ``PerfCollector`` reads them straight from
``task_state.messages`` in :meth:`DefaultEvaluator._persist_result`.
"""
from typing import Any, Optional, Union

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessage, ChatMessageUser
from evalscope.api.model import Model, ModelOutput
from evalscope.utils.logger import get_logger
from .default_data_adapter import DefaultDataAdapter

logger = get_logger()


class MultiTurnAdapter(DefaultDataAdapter):
    """Base adapter for benchmarks that require multi-turn model inference.

    Subclasses MUST set ``self.max_turns`` (typically from ``extra_params``)
    in ``__init__`` and override :meth:`build_turn_prompt`. They MAY override
    :meth:`on_turn_complete` to capture per-turn information.
    """

    #: Default upper bound on turns. Subclasses usually override via
    #: ``self.max_turns = self.extra_params.get('max_turns', N)``.
    max_turns: int = 1

    # ------------------------------------------------------------------ #
    # Hooks to be implemented / overridden by subclasses                   #
    # ------------------------------------------------------------------ #

    def build_turn_prompt(
        self,
        sample: Sample,
        history: list,
        turn_index: int,
    ) -> Optional[Union[str, ChatMessage]]:
        """Return the user prompt for ``turn_index``, or ``None`` to stop.

        Args:
            sample: The sample being evaluated.
            history: Mutable list of chat messages so far (do not append;
                the base loop handles that).
            turn_index: 0-based turn index (0 = first user turn).

        Returns:
            * ``None`` — signal end-of-conversation (loop exits cleanly).
            * ``str`` — convenience, will be wrapped in ``ChatMessageUser``.
            * ``ChatMessage`` — used as-is (useful for multi-modal content).
        """
        raise NotImplementedError

    def on_turn_complete(
        self,
        sample: Sample,
        turn_index: int,
        model_output: ModelOutput,
        history: list,
    ) -> None:
        """Optional hook invoked after each turn's response is appended.

        Override to record per-turn bookkeeping (e.g. scoring inputs) into
        ``sample.metadata``. Default: no-op.
        """
        return None

    def get_max_turns(self, sample: Sample) -> int:
        """Return the max number of turns for this sample.

        Override when the turn budget is per-sample (e.g. determined by the
        length of a list in ``sample.metadata``). Default: ``self.max_turns``.
        """
        return self.max_turns

    def initialize_history(self, sample: Sample) -> list:
        """Return the initial conversation history before the first turn.

        Override to pre-populate the history with a system message, fixed
        context, or any other seed messages. Default: empty list.
        """
        return []

    def build_final_output(
        self,
        model: Model,
        sample: Sample,
        history: list,
        last_output: Optional[ModelOutput],
    ) -> ModelOutput:
        """Return the ``ModelOutput`` used for ``TaskState.output``.

        Default: the last turn's ``ModelOutput``, or an empty ``ModelOutput``
        if no turns executed. Override to, e.g., synthesize an output from
        the full conversation for display purposes.
        """
        if last_output is not None:
            return last_output
        logger.warning(
            f'MultiTurnAdapter produced no turns for sample {sample.id}; '
            'returning an empty ModelOutput.'
        )
        return ModelOutput(model=model.name)

    # ------------------------------------------------------------------ #
    # Orchestration                                                         #
    # ------------------------------------------------------------------ #

    def run_inference(self, model: Model, sample: Sample, output_dir: str, **kwargs: Any) -> TaskState:
        """Drive a multi-turn conversation with ``model``.

        Builds the conversation one turn at a time, letting ``build_turn_prompt``
        decide when to stop. The final ``TaskState`` carries the complete
        message history (each assistant message already has its own
        ``perf_metrics`` when ``collect_perf`` is enabled) plus a
        ``ModelOutput`` produced by :meth:`build_final_output`.

        Subclasses typically customize behavior via hooks rather than
        overriding this method:

        * Simple multi-turn: implement :meth:`build_turn_prompt` only.
        * Per-turn bookkeeping (multi_if-style): also override
          :meth:`on_turn_complete`.
        * Per-sample turn budget / seed system prompt / custom final
          output (scicode-style): override :meth:`get_max_turns`,
          :meth:`initialize_history`, and/or :meth:`build_final_output`.
        """
        history: list = self.initialize_history(sample)
        max_turns = self.get_max_turns(sample)
        last_output: Optional[ModelOutput] = None

        for turn_index in range(max_turns):
            prompt = self.build_turn_prompt(sample, history, turn_index)
            if prompt is None:
                break
            if isinstance(prompt, str):
                prompt = ChatMessageUser(content=prompt)
            history.append(prompt)

            last_output = model.generate(input=history, tools=sample.tools)
            # Assistant message already carries perf_metrics when collect_perf=True.
            history.append(last_output.message)

            self.on_turn_complete(sample, turn_index, last_output, history)

        return TaskState(
            model=model.name,
            sample=sample,
            messages=history,
            output=self.build_final_output(model, sample, history, last_output),
            completed=True,
        )
