import asyncio
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from evalscope.perf.arguments import Arguments
from evalscope.perf.core.strategies.base import BenchmarkStrategy
from evalscope.perf.plugin.datasets.base import Conversation, Message, Turn
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.perf.core.http_client import AioHttpClient
    from evalscope.perf.plugin.api.base import ApiPluginBase

logger = get_logger()


class MultiTurnStrategy(BenchmarkStrategy):
    """Multi-turn conversation benchmark strategy.

    Each worker owns one active conversation at a time and progresses through
    its turns sequentially.  Workers cycle through ``all_conversations`` until
    ``args.number`` conversations have been started (attempted).  A conversation
    that is abandoned mid-way due to a failed turn still counts toward this
    budget; only degenerate empty conversations are excluded.

    Open-loop mode is intentionally **not** supported for multi-turn
    conversations.  The fundamental reason is that open-loop semantics require
    each request to be fired independently of in-flight requests, but multi-turn
    conversations have an inherent sequential dependency: turn N cannot begin
    until the assistant response for turn N-1 has been received (the response
    must be appended to the context before the next request can be built).
    Decoupling dispatch from completion would break the conversation context
    and produce meaningless results.
    """

    def __init__(
        self,
        args: Arguments,
        api_plugin: 'ApiPluginBase',
        client: 'AioHttpClient',
        queue: asyncio.Queue,
        all_conversations: List[Conversation],
    ) -> None:
        super().__init__(args, api_plugin, client, queue)
        self._all_conversations = all_conversations
        # Conversation cycling index – safe without a lock because asyncio is
        # single-threaded/cooperative.  Warmup and benchmark still pull disjoint
        # conversations from the dataset; the hand-off fix changes only when
        # workers move from warmup work to measured work, not which conversations
        # are measured.
        self._conv_index = 0
        self._warmup_count = self.args.warmup_count

        # Shared dispatch state.  ``_phase_counter`` claims work items from one
        # ordered stream: first ``_phase_warmup_budget`` warmup conversations,
        # then measured conversations up to ``_phase_budget``.  ``_phase_duration``
        # is armed lazily when the first measured conversation is claimed so
        # warmup does not consume the timed budget.
        self._phase_counter = 0
        self._phase_budget = 0
        self._phase_warmup_budget = 0
        self._phase_is_warmup = False
        self._phase_deadline: Optional[float] = None
        self._phase_duration: Optional[float] = None

        # Trace identity:  monotonic across phases; each claimed conversation
        # gets a unique ``trace_id`` string for trace-level metric aggregation.
        self._next_trace_seq = 0

        if self._warmup_count > 0:
            logger.info(f'Warmup enabled: {self._warmup_count} warmup conversations (benchmark: {self.args.number})')

    def _next_conversation(self) -> Conversation:
        """Return the next conversation from the cycled pool."""
        conv = self._all_conversations[self._conv_index % len(self._all_conversations)]
        self._conv_index += 1
        return conv

    def _next_trace_id(self, is_warmup: bool) -> str:
        """Allocate a unique trace_id string for one claimed conversation."""
        seq = self._next_trace_seq
        self._next_trace_seq += 1
        return f'{"warmup" if is_warmup else "bench"}-{seq}'

    async def _worker(self, worker_id: int) -> None:
        """Process conversations until the current work stream is exhausted."""
        while True:
            # Atomically claim a conversation slot before awaiting to prevent
            # other workers from overshooting the stream budget.
            if self._phase_counter >= self._phase_budget:
                return
            work_index = self._phase_counter
            is_warmup = work_index < self._phase_warmup_budget

            # Trace-level soft exit: warmup is exempt.  The measured deadline is
            # armed by the first measured conversation claim, which keeps all
            # warmup service time outside the measured window.
            if not is_warmup:
                if self._phase_deadline is not None and time.perf_counter() >= self._phase_deadline:
                    return
                if self._phase_deadline is None and self._phase_duration is not None:
                    self._phase_deadline = self._compute_deadline(self._phase_duration)

            self._phase_counter += 1
            conversation = self._next_conversation()
            trace_id = self._next_trace_id(is_warmup)

            if not conversation:
                # Degenerate conversation with no turns – skip without counting.
                self._phase_counter -= 1
                continue

            # Accumulated context sent with each turn.  Real assistant responses
            # are appended after each successful turn so the next turn sees the
            # growing history.
            context: List[Message] = []
            prev_prompt_tokens: int = 0
            prev_completion_tokens: int = 0
            total_turns = len(conversation)

            for turn_idx, turn in enumerate(conversation):
                # turn: Turn – delta messages plus optional per-turn overrides.

                # Respect per-conversation max_turns.
                if self.args.max_turns is not None and turn_idx >= self.args.max_turns:
                    # Mark the last successfully enqueued turn as conversation-final.
                    # The turn at turn_idx was never sent, so turn_idx-1 was the last.
                    # Nothing to mark here; the previous iteration already set is_last_turn
                    # via the look-ahead below if it was the effective last turn.
                    break

                # Trace-replay datasets may specify a per-turn pre-send wait to
                # simulate tool-call latency.  No-op for legacy plugins.
                if turn.tool_call_latency:
                    await asyncio.sleep(turn.tool_call_latency)

                # Append this turn's delta to the growing context.
                context.extend([m.copy() for m in turn.messages])

                # Rate limiting: apply a Poisson inter-request sleep so
                # multi-turn runs honour the configured arrival rate.
                # Relative-time pacing is appropriate here because the dominant
                # delay between turns is the API response latency (seconds),
                # not event-loop jitter (microseconds).  The sleep is simply
                # an additional throttle after each response is received.
                if self.args.rate != -1:
                    interval = np.random.exponential(1.0 / self.args.rate)
                    await asyncio.sleep(interval)

                # Send the turn.  Per-turn ``max_tokens`` (from trace replay)
                # overrides the global ``--max-tokens`` when set.
                request = self.api_plugin.build_request(list(context))
                if request is None:
                    logger.error(
                        f'worker={worker_id} turn={turn_idx}: build_request returned None; abandoning conversation.'
                    )
                    break
                if turn.max_tokens is not None:
                    request['max_tokens'] = turn.max_tokens
                benchmark_data = await self.client.post(request)

                # Inject multi-turn specific metadata.
                benchmark_data.is_warmup = is_warmup
                benchmark_data.input_num_turns = turn_idx + 1
                benchmark_data.trace_id = trace_id
                benchmark_data.is_first_turn = turn_idx == 0

                # Ensure token counts are available before computing cache ratio.
                # Some OpenAI-compatible servers omit ``usage`` in the stream, so
                # prompt_tokens / completion_tokens remain None until finalize() is
                # called.  finalize() is idempotent.
                if benchmark_data.success:
                    benchmark_data.finalize(self.api_plugin)

                # Compute KV-cache hit count (absolute tokens, not a percentage).
                #
                # Priority:
                #   1. real_cached_tokens – server-reported cached token count
                #      (from usage.prompt_tokens_details.cached_tokens).
                #   2. Estimation heuristic – prev_prompt_tokens + prev_completion_tokens,
                #      i.e. the full context that was already in the KV cache after turn N-1.
                #
                # Turn 1 always yields cached_tokens = 0 because there is no prior
                # context.  The 0 is stored explicitly so the aggregator can include
                # this turn's prompt_tokens in the denominator, producing an unbiased
                # global ratio: total_cached_tokens / total_prompt_tokens.
                if benchmark_data.prompt_tokens is not None and benchmark_data.prompt_tokens > 0:
                    if benchmark_data.real_cached_tokens is not None:
                        benchmark_data.cached_tokens = benchmark_data.real_cached_tokens
                    elif prev_prompt_tokens > 0:
                        cacheable_tokens = prev_prompt_tokens + prev_completion_tokens
                        benchmark_data.cached_tokens = cacheable_tokens
                    else:
                        # Turn 1: no prior context, cached_tokens = 0.
                        benchmark_data.cached_tokens = 0
                if benchmark_data.prompt_tokens:
                    prev_prompt_tokens = benchmark_data.prompt_tokens
                if benchmark_data.completion_tokens:
                    prev_completion_tokens = benchmark_data.completion_tokens

                # Determine whether this is the last turn of the conversation:
                # • normal completion: final index in the dataset
                # • max_turns cap: next iteration would be skipped
                # • request failure: conversation is abandoned after this turn
                effective_last = (
                    turn_idx == total_turns - 1
                    or (self.args.max_turns is not None and turn_idx + 1 >= self.args.max_turns)
                    or not benchmark_data.success
                )

                # Enqueue for metrics collection.
                benchmark_data.is_last_turn = effective_last
                await self.queue.put(benchmark_data)

                if not benchmark_data.success:
                    logger.debug(
                        f'worker={worker_id} turn={turn_idx} failed ({benchmark_data.error}), abandoning conversation.'
                    )
                    break

                # Append real response to context for next turn.
                context.append(
                    {
                        'role': 'assistant',
                        'content': benchmark_data.generated_text,
                    }
                )

    async def run(self) -> None:
        self._log_warmup_handoff()
        await self._run_phase(
            budget=self._warmup_count + self.args.number,
            is_warmup=False,
            warmup_budget=self._warmup_count,
            duration=self.args.duration,
        )

    def _log_warmup_handoff(self) -> None:
        """Warn when warmup cannot cover the opening multi-turn cohort."""
        if self.args.parallel <= 1:
            return
        if self._warmup_count <= 0:
            logger.warning(
                'Multi-turn warmup is disabled; the first measured cohort may start against an idle server. '
                f'Use --warmup-num {self.args.parallel} or higher to cover every concurrency slot.'
            )
            return
        if self._warmup_count < self.args.parallel:
            uncovered = self.args.parallel - self._warmup_count
            logger.warning(
                f'Multi-turn warmup covers only {self._warmup_count} of the {self.args.parallel} '
                f'concurrency slots; {uncovered} measured conversation(s) may still be released '
                'in the opening cohort.'
            )

    async def _run_phase(
        self,
        budget: int,
        is_warmup: bool,
        deadline: Optional[float] = None,
        warmup_budget: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> None:
        """Spawn ``args.parallel`` workers and drain one ordered work stream.

        By default this keeps the historical single-phase behaviour used by
        lifecycle tests.  ``warmup_budget`` turns the phase into a hand-off
        stream: the first N claimed conversations are warmup, and following
        claims are measured without stopping the workers in between.
        """
        self._phase_counter = 0
        self._phase_budget = budget
        if warmup_budget is None and is_warmup:
            self._phase_warmup_budget = budget
        else:
            self._phase_warmup_budget = warmup_budget or 0
        self._phase_is_warmup = is_warmup
        self._phase_deadline = deadline
        self._phase_duration = duration
        workers = [asyncio.create_task(self._worker(worker_id=i)) for i in range(self.args.parallel)]
        try:
            await asyncio.gather(*workers)
        finally:
            for worker in workers:
                if not worker.done():
                    worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            self._phase_duration = None
            self._phase_warmup_budget = 0
