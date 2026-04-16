"""Multi-turn conversation benchmark for evalscope perf.

Execution model
---------------
``parallel`` asyncio tasks (workers) run concurrently.  Each worker owns one
active conversation at a time and progresses through its turns sequentially:

::

    Conversation Pool: [conv_0, conv_1, ..., conv_M]  (cycled until turn budget)
             |
       +-----+------+
     Worker_0    Worker_1  ...  Worker_(parallel-1)
      |                |
     conv_i turn_0   conv_j turn_0
      | (await)       | (await)          <- parallel turns in-flight simultaneously
     conv_i turn_1   conv_j turn_1
      |               |
     ...             ...

Parameter semantics (same as normal benchmark mode)
----------------------------------------------------
* ``--number``   - total number of turns (API requests) to send.
* ``--parallel`` - number of concurrently active turn-level HTTP requests.

Multi-turn specific parameters
-------------------------------
* ``--min-turns`` - minimum user turns per conversation (random_multi_turn).
* ``--max-turns`` - maximum user turns per conversation.
"""

import asyncio
from typing import TYPE_CHECKING, Dict, List, Tuple

from evalscope.perf.arguments import Arguments
from evalscope.perf.benchmark import connect_test, data_process_completed_event, statistic_benchmark_metric
from evalscope.perf.http_client import AioHttpClient
from evalscope.perf.plugin import ApiRegistry, DatasetRegistry
from evalscope.perf.utils.db_util import summary_result
from evalscope.perf.utils.handler import exception_handler
from evalscope.utils.logger import get_logger
from evalscope.utils.tqdm_utils import TqdmLogging as tqdm

if TYPE_CHECKING:
    from evalscope.perf.plugin.api.base import ApiPluginBase

logger = get_logger()


def _extract_user_turns(conversation: List[Dict]) -> List[Dict]:
    """Return only the user-role messages from a conversation in order."""
    return [m for m in conversation if m.get('role') == 'user']


@exception_handler
async def multi_turn_benchmark(args: Arguments) -> Tuple[Dict, Dict]:
    """Run a multi-turn conversation benchmark.

    Args:
        args: Benchmark configuration.  ``args.number`` is the total turn
              budget and ``args.parallel`` controls concurrency.

    Returns:
        Tuple of ``(metrics_result, percentile_result)`` dicts, identical in
        structure to the output of the standard ``benchmark()`` function.
    """
    api_plugin_class = ApiRegistry.get_class(args.api)
    api_plugin: 'ApiPluginBase' = api_plugin_class(args)

    # ------------------------------------------------------------------
    # 1. Load all conversations from the dataset
    # ------------------------------------------------------------------
    logger.info(f'Loading conversations from dataset: {args.dataset}')
    dataset_plugin = DatasetRegistry.get_class(args.dataset)(args)

    with tqdm(desc='Loading[conversations]', logger=logger) as pbar:
        all_conversations: List[List[Dict]] = []
        for conv in dataset_plugin.build_messages():
            all_conversations.append(conv)
            pbar.update(1)

    if not all_conversations:
        raise ValueError(f'Dataset "{args.dataset}" produced no conversations!')

    logger.info(f'Loaded {len(all_conversations)} conversations')

    # ------------------------------------------------------------------
    # 2. Setup shared state
    # ------------------------------------------------------------------
    benchmark_data_queue: asyncio.Queue = asyncio.Queue(maxsize=max(1, args.parallel * args.queue_size_multiplier))
    data_process_completed_event.clear()

    # Conversation cycling: each worker independently advances this counter.
    # asyncio is single-threaded / cooperative, so plain int is safe.
    _conv_index = 0
    _turn_counter = 0

    def _next_conversation() -> List[Dict]:
        """Return the next conversation from the cycled pool."""
        nonlocal _conv_index
        conv = all_conversations[_conv_index % len(all_conversations)]
        _conv_index += 1
        return conv

    # ------------------------------------------------------------------
    # 3. Test connection (reuse connect_test from benchmark.py)
    # ------------------------------------------------------------------
    await connect_test(args, api_plugin)

    # ------------------------------------------------------------------
    # 4. Shared HTTP client
    # ------------------------------------------------------------------
    client = AioHttpClient(args, api_plugin)

    async with client:
        # ----------------------------------------------------------------
        # 5. Start the metrics consumer task (reused from benchmark.py)
        # ----------------------------------------------------------------
        statistic_task = asyncio.create_task(statistic_benchmark_metric(benchmark_data_queue, args, api_plugin))

        # ----------------------------------------------------------------
        # 6. Worker coroutine: one per parallel slot
        # ----------------------------------------------------------------
        async def conversation_worker(worker_id: int) -> None:
            """Process conversations until the global turn budget is reached."""
            nonlocal _turn_counter

            while _turn_counter < args.number:
                # Grab the next conversation (cycled)
                conversation = _next_conversation()
                user_msgs = _extract_user_turns(conversation)

                if not user_msgs:
                    # Degenerate conversation with no user messages, skip
                    continue

                # Accumulated context sent with each turn.
                # Real assistant responses are appended after each successful
                # turn so the next turn sees the growing history.
                context: List[Dict] = []
                prev_prompt_tokens: int = 0  # used for approx_cached_percent
                prev_completion_tokens: int = 0  # needed to account for cached asst tokens

                for user_turn_idx, user_msg in enumerate(user_msgs):
                    # ---- Check global turn budget ----
                    if _turn_counter >= args.number:
                        return

                    # ---- Respect per-conversation max_turns ----
                    if args.max_turns is not None and user_turn_idx >= args.max_turns:
                        break

                    # ---- Build current context (append user message) ----
                    context.append(user_msg.copy())

                    # ---- Reserve this turn slot BEFORE awaiting ----
                    # Incrementing here (before the await) ensures no other
                    # worker can claim the same slot, preventing overshoot
                    # beyond args.number.
                    _turn_counter += 1

                    # ---- Send the turn ----
                    request = api_plugin.build_request(list(context))
                    benchmark_data = await client.post(request)

                    # ---- Inject multi-turn specific metadata ----
                    benchmark_data.input_num_turns = user_turn_idx + 1

                    # Estimate KV-cache hit rate.
                    # cacheable = prev_prompt_tokens + prev_completion_tokens
                    # because after turn N-1, the server KV cache holds:
                    #   [user_0, ..., user_{N-1}]  (= prev_prompt_tokens)
                    #   [asst_{N-1}]               (= prev_completion_tokens)
                    # both of which appear as prefix in the current request.
                    # This matches vLLM's formula:
                    #   (input_tokens - current_user_tokens) / input_tokens
                    if (
                        benchmark_data.prompt_tokens is not None and benchmark_data.prompt_tokens > 0
                        and prev_prompt_tokens > 0
                    ):
                        cacheable_tokens = prev_prompt_tokens + prev_completion_tokens
                        benchmark_data.approx_cached_percent = (100.0 * cacheable_tokens / benchmark_data.prompt_tokens)
                    if benchmark_data.prompt_tokens:
                        prev_prompt_tokens = benchmark_data.prompt_tokens
                    if benchmark_data.completion_tokens:
                        prev_completion_tokens = benchmark_data.completion_tokens

                    # ---- Enqueue for metrics ----
                    await benchmark_data_queue.put(benchmark_data)

                    if not benchmark_data.success:
                        logger.debug(
                            f'worker={worker_id} turn={user_turn_idx} '
                            f'failed ({benchmark_data.error}), abandoning conversation.'
                        )
                        break

                    # ---- Append real response to context for next turn ----
                    context.append({
                        'role': 'assistant',
                        'content': benchmark_data.generated_text,
                    })

        # ----------------------------------------------------------------
        # 7. Launch workers and wait for all to finish
        # ----------------------------------------------------------------
        workers = [asyncio.create_task(conversation_worker(worker_id=i)) for i in range(args.parallel)]
        await asyncio.gather(*workers, return_exceptions=True)

        # ----------------------------------------------------------------
        # 8. Drain the metrics queue and signal the consumer to stop
        # ----------------------------------------------------------------
        await benchmark_data_queue.join()
        data_process_completed_event.set()

        metrics, result_db_path = await statistic_task

    # ------------------------------------------------------------------
    # 9. Summarise and return
    # ------------------------------------------------------------------
    metrics_result, percentile_result = summary_result(args, metrics, result_db_path)
    return metrics_result, percentile_result
