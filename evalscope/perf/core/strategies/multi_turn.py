import asyncio
import numpy as np
from typing import TYPE_CHECKING, Any, Dict, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.core.strategies.base import BenchmarkStrategy
from evalscope.perf.plugin.datasets.base import Message, Messages
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.perf.core.http_client import AioHttpClient
    from evalscope.perf.plugin.api.base import ApiPluginBase

logger = get_logger()


class MultiTurnStrategy(BenchmarkStrategy):
    """Multi-turn conversation benchmark strategy.

    Each worker owns one active conversation at a time and progresses through
    its turns sequentially.  Workers cycle through ``all_conversations`` until
    ``args.number`` conversations have been completed.

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
        all_conversations: List[List[Messages]],
    ) -> None:
        super().__init__(args, api_plugin, client, queue)
        self._all_conversations = all_conversations
        # Conversation cycling index – safe without a lock because asyncio is
        # single-threaded/cooperative.
        self._conv_index = 0
        self._conv_counter = 0

    def _next_conversation(self) -> List[Dict]:
        """Return the next conversation from the cycled pool."""
        conv = self._all_conversations[self._conv_index % len(self._all_conversations)]
        self._conv_index += 1
        return conv

    async def _worker(self, worker_id: int) -> None:
        """Process conversations until the global conversation budget is reached."""
        while True:
            # Atomically claim a conversation slot before awaiting to prevent
            # other workers from overshooting args.number.
            if self._conv_counter >= self.args.number:
                return
            self._conv_counter += 1
            conversation = self._next_conversation()

            if not conversation:
                # Degenerate conversation with no turns – skip without counting.
                self._conv_counter -= 1
                continue

            # Accumulated context sent with each turn.  Real assistant responses
            # are appended after each successful turn so the next turn sees the
            # growing history.
            context: List[Message] = []
            prev_prompt_tokens: int = 0
            prev_completion_tokens: int = 0

            for turn_idx, turn_delta in enumerate(conversation):
                # turn_delta: Messages – the delta to append for this turn

                # Respect per-conversation max_turns.
                if self.args.max_turns is not None and turn_idx >= self.args.max_turns:
                    break

                # Append this turn's delta to the growing context.
                context.extend([m.copy() for m in turn_delta])

                # Rate limiting (mirrors standard benchmark behaviour).
                # When --rate is set, apply a Poisson inter-request sleep so
                # multi-turn runs honour the configured arrival rate.
                if self.args.rate != -1:
                    interval = np.random.exponential(1.0 / self.args.rate)
                    await asyncio.sleep(interval)

                # Send the turn.
                request = self.api_plugin.build_request(list(context))
                benchmark_data = await self.client.post(request)

                # Inject multi-turn specific metadata.
                benchmark_data.input_num_turns = turn_idx + 1

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

                # Enqueue for metrics collection.
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
        workers = [asyncio.create_task(self._worker(worker_id=i)) for i in range(self.args.parallel)]
        await asyncio.gather(*workers, return_exceptions=True)
