import asyncio
import numpy as np
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

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
        # single-threaded/cooperative.  Continuous across phases so warmup
        # consumes the first ``warmup_count`` conversations and benchmark
        # picks up from there (preserves dataset uniqueness contract).
        self._conv_index = 0
        self._warmup_count = self.args.warmup_count

        # Per-phase dispatch state.  Reset at the start of every phase by
        # ``_run_phase``.  ``_phase_counter`` tracks how many conversations
        # this phase has already claimed; ``_phase_budget`` caps that count;
        # ``_phase_is_warmup`` tags every enqueued ``BenchmarkData``;
        # ``_phase_deadline`` is the trace-soft-exit cutoff (None disables it).
        self._phase_counter = 0
        self._phase_budget = 0
        self._phase_is_warmup = False
        self._phase_deadline: Optional[float] = None

        # Trace identity:  monotonic across phases; each claimed conversation
        # gets a unique ``trace_id`` string for trace-level metric aggregation.
        self._next_trace_seq = 0

        if self._warmup_count > 0:
            logger.info(
                f'Warmup enabled: {self._warmup_count} warmup conversations '
                f'(benchmark: {self.args.number})'
            )

    def _next_conversation(self) -> Conversation:
        """Return the next conversation from the cycled pool."""
        conv = self._all_conversations[self._conv_index % len(self._all_conversations)]
        self._conv_index += 1
        return conv

    def _next_trace_id(self, is_warmup: bool) -> str:
        """Allocate a unique trace_id string for one claimed conversation."""
        seq = self._next_trace_seq
        self._next_trace_seq += 1
        return f"{'warmup' if is_warmup else 'bench'}-{seq}"

    async def _worker(self, worker_id: int) -> None:
        """Process conversations until the current phase budget is reached."""
        while True:
            # Trace-level soft exit: stop claiming new conversations once the
            # duration deadline has elapsed.  An already-claimed conversation
            # below will still run all its remaining turns to completion so
            # trace-level metrics stay coherent (matches trie's semantics).
            if self._phase_deadline is not None and time.perf_counter() >= self._phase_deadline:
                return
            # Atomically claim a conversation slot before awaiting to prevent
            # other workers from overshooting the phase budget.
            if self._phase_counter >= self._phase_budget:
                return
            self._phase_counter += 1
            is_warmup = self._phase_is_warmup
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

                # Rate limiting (mirrors standard benchmark behaviour).
                # When --rate is set, apply a Poisson inter-request sleep so
                # multi-turn runs honour the configured arrival rate.
                if self.args.rate != -1:
                    interval = np.random.exponential(1.0 / self.args.rate)
                    await asyncio.sleep(interval)

                # Send the turn.  Per-turn ``max_tokens`` (from trace replay)
                # overrides the global ``--max-tokens`` when set.
                request = self.api_plugin.build_request(list(context))
                if request is None:
                    logger.error(
                        f'worker={worker_id} turn={turn_idx}: build_request returned None; '
                        'abandoning conversation.'
                    )
                    break
                if turn.max_tokens is not None:
                    request['max_tokens'] = turn.max_tokens
                benchmark_data = await self.client.post(request)

                # Inject multi-turn specific metadata.
                benchmark_data.is_warmup = is_warmup
                benchmark_data.input_num_turns = turn_idx + 1
                benchmark_data.trace_id = trace_id
                benchmark_data.is_first_turn = (turn_idx == 0)

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
                        f'worker={worker_id} turn={turn_idx} '
                        f'failed ({benchmark_data.error}), abandoning conversation.'
                    )
                    break

                # Append real response to context for next turn.
                context.append({
                    'role': 'assistant',
                    'content': benchmark_data.generated_text,
                })

    async def run(self) -> None:
        # Two-phase dispatch: warmup conversations complete in full before any
        # benchmark conversation starts.  ``_conv_index`` persists across
        # phases so warmup and benchmark pull disjoint conversations from the
        # dataset (first ``warmup_count`` vs. the rest).
        if self._warmup_count > 0:
            # Warmup ignores --duration (warmup must finish in full before
            # the timed benchmark window begins, mirroring trie's model).
            await self._run_phase(budget=self._warmup_count, is_warmup=True, deadline=None)
        await self._run_phase(
            budget=self.args.number,
            is_warmup=False,
            deadline=self._compute_deadline(self.args.duration),
        )

    async def _run_phase(self, budget: int, is_warmup: bool, deadline: Optional[float] = None) -> None:
        """Spawn ``args.parallel`` workers and drain them within one phase.

        When ``deadline`` is set, workers stop claiming new conversations once
        wall-clock crosses it, but any conversation already in progress is
        allowed to finish all its turns (trace-level soft exit, matches trie).
        ``budget`` is still honoured as an upper bound on the number of
        conversations claimed; whichever limit (count or wall-clock) is hit
        first ends the phase.
        """
        self._phase_counter = 0
        self._phase_budget = budget
        self._phase_is_warmup = is_warmup
        self._phase_deadline = deadline
        workers = [asyncio.create_task(self._worker(worker_id=i)) for i in range(self.args.parallel)]
        await asyncio.gather(*workers, return_exceptions=True)
