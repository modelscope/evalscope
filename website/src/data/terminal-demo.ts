export type TerminalTone = 'command' | 'status' | 'result' | 'path' | 'muted';

export interface TerminalReplayLine {
  text: string;
  tone: TerminalTone;
  block?: 'table';
  progress?: string[];
}

export interface TerminalReplayScene {
  id: 'eval' | 'perf' | 'service';
  icon: string;
  label: string;
  command: string;
  lines: TerminalReplayLine[];
  summary: string;
  recordedAt: string;
  outputPath: string;
}

const perfSummary = `╭──────────────────────────────────────────────────────────────────────────────╮
│ Performance Test Summary Report                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

Basic Information:
┌───────────────────────┬──────────────────────────────────────────────────────┐
│ Model                 │ qwen-plus                                            │
│ Test Dataset          │ random                                               │
│ API Type              │ openai                                               │
│ Total Generated       │ 7,200.0 tokens                                       │
│ Avg Output Rate       │ 73.3 tok/s                                           │
│ Total Test Time       │ 98.23 s                                              │
│ Output Path           │ outputs/website-terminal-demo/20260911_103241/qwen-… │
└───────────────────────┴──────────────────────────────────────────────────────┘


             Performance Overview
┏━━━━━━┳━━━━━━┳━━━━━┳━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃Conc. ┃ Rate ┃ Num ┃  RPS ┃  Gen/s ┃ Success┃
┡━━━━━━╇━━━━━━╇━━━━━╇━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│    1 │    - │  24 │ 0.47 │     47 │    100%│
│    2 │    - │  24 │ 0.79 │  79.41 │    100%│
│    4 │    - │  24 │ 1.42 │ 141.66 │    100%│
└──────┴──────┴─────┴──────┴────────┴────────┘


                         Per-Request Metrics
┏━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃Conc. ┃ Rate ┃ Metric            ┃    avg ┃    p50 ┃    p99 ┃    max┃
┡━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│    1 │    - │ Latency (s)       │  2.128 │   2.02 │   2.92 │   2.92│
│      │      │ TTFT (ms)         │ 433.25 │ 413.21 │ 721.08 │ 721.08│
│      │      │ TPOT (ms)         │  17.11 │  15.82 │  23.46 │  23.46│
│      │      │ Input Tokens      │   1003 │   1003 │   1003 │   1003│
│      │      │ Output Tokens     │    100 │    100 │    100 │    100│
│      │      │ Cache Hit (%)     │     0% │      - │      - │      -│
│      │      │ Decode toks/s     │  58.45 │      - │      - │      -│
│      │      │ Decoded Tok/Iter  │   5.02 │      - │      - │      -│
│      │      │ Spec. Accept Rate │  80.1% │      - │      - │      -│
├──────┼──────┼───────────────────┼────────┼────────┼────────┼───────┤
│    2 │    - │ Latency (s)       │  2.419 │   2.38 │   3.45 │   3.45│
│      │      │ TTFT (ms)         │ 446.91 │ 423.58 │  779.5 │  779.5│
│      │      │ TPOT (ms)         │  19.91 │  19.44 │     29 │     29│
│      │      │ Input Tokens      │   1003 │   1003 │   1003 │   1003│
│      │      │ Output Tokens     │    100 │    100 │    100 │    100│
│      │      │ Cache Hit (%)     │     0% │      - │      - │      -│
│      │      │ Decode toks/s     │  50.23 │      - │      - │      -│
│      │      │ Decoded Tok/Iter  │   4.76 │      - │      - │      -│
│      │      │ Spec. Accept Rate │    79% │      - │      - │      -│
├──────┼──────┼───────────────────┼────────┼────────┼────────┼───────┤
│    4 │    - │ Latency (s)       │  2.357 │   2.48 │   3.01 │   3.01│
│      │      │ TTFT (ms)         │ 405.09 │ 352.01 │ 845.67 │ 845.67│
│      │      │ TPOT (ms)         │  19.71 │  21.83 │  25.76 │  25.76│
│      │      │ Input Tokens      │   1003 │   1003 │   1003 │   1003│
│      │      │ Output Tokens     │    100 │    100 │    100 │    100│
│      │      │ Cache Hit (%)     │     0% │      - │      - │      -│
│      │      │ Decode toks/s     │  50.74 │      - │      - │      -│
│      │      │ Decoded Tok/Iter  │   4.71 │      - │      - │      -│
│      │      │ Spec. Accept Rate │  78.8% │      - │      - │      -│
└──────┴──────┴───────────────────┴────────┴────────┴────────┴───────┘


                             Workload Throughput
┏━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃Conc. ┃ Rate ┃ Metric (tok/s)      ┃ Overall ┃ Last 30s ┃ Steady (drop 20%)┃
┡━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│    1 │    - │ Total Prompt tok/s  │  471.37 │   478.68 │            464.66│
│      │      │ New Prompt tok/s    │  471.37 │   478.68 │            464.66│
│      │      │ Cached Prompt tok/s │       0 │        0 │                 0│
│      │      │ Completion tok/s    │      47 │    47.73 │             46.33│
├──────┼──────┼─────────────────────┼─────────┼──────────┼──────────────────┤
│    2 │    - │ Total Prompt tok/s  │  796.49 │   844.58 │            847.26│
│      │      │ New Prompt tok/s    │  796.49 │   844.58 │            847.26│
│      │      │ Cached Prompt tok/s │       0 │        0 │                 0│
│      │      │ Completion tok/s    │   79.41 │    84.21 │             84.47│
├──────┼──────┼─────────────────────┼─────────┼──────────┼──────────────────┤
│    4 │    - │ Total Prompt tok/s  │ 1420.83 │  1420.83 │           1649.88│
│      │      │ New Prompt tok/s    │ 1420.83 │  1420.83 │           1649.88│
│      │      │ Cached Prompt tok/s │       0 │        0 │                 0│
│      │      │ Completion tok/s    │  141.66 │   141.66 │            164.49│
└──────┴──────┴─────────────────────┴─────────┴──────────┴──────────────────┘`;

export const terminalReplayScenes: TerminalReplayScene[] = [
  {
    id: 'eval',
    icon: 'tabler:scale',
    label: 'eval',
    command: `evalscope eval \\
  --model qwen-plus \\
  --eval-type openai_api \\
  --api-url https://dashscope.aliyuncs.com/compatible-mode/v1 \\
  --api-key "$DASHSCOPE_API_KEY" \\
  --datasets gsm8k arc --limit 5 --seed 42 \\
  --generation-config stream=True \\
  --work-dir outputs/website-terminal-demo`,
    lines: [
      { text: '$ evalscope eval \\', tone: 'command' },
      { text: '  --model qwen-plus \\', tone: 'command' },
      { text: '  --eval-type openai_api \\', tone: 'command' },
      { text: '  --api-url https://dashscope.aliyuncs.com/compatible-mode/v1 \\', tone: 'command' },
      { text: '  --api-key "$DASHSCOPE_API_KEY" \\', tone: 'command' },
      { text: '  --datasets gsm8k arc --limit 5 --seed 42 \\', tone: 'command' },
      { text: '  --generation-config stream=True \\', tone: 'command' },
      { text: '  --work-dir outputs/website-terminal-demo', tone: 'command' },
      {
        text: '2026-09-11 10:32:18 - evalscope - INFO: Args: Task config is provided with CommandLine type.',
        tone: 'status',
      },
      { text: '2026-09-11 10:32:19 - evalscope - INFO: Running with native backend', tone: 'status' },
      {
        text: '2026-09-11 10:32:19 - evalscope - INFO: Dump task config to outputs/website-terminal-demo/20260911_103219/configs/task_config.yaml',
        tone: 'path',
      },
      { text: '    "generation_config": {"batch_size": 8, "stream": true}', tone: 'muted' },
      { text: '2026-09-11 10:32:19 - evalscope - INFO: Start loading benchmark dataset: gsm8k', tone: 'status' },
      {
        text: '2026-09-11 10:32:19 - evalscope - WARNING: gsm8k: 5 samples to evaluate (1 subset, --limit=5 per subset).',
        tone: 'muted',
      },
      { text: "2026-09-11 10:32:19 - evalscope - INFO: Subsets of gsm8k: ['main']", tone: 'status' },
      { text: '2026-09-11 10:32:19 - evalscope - INFO: Loading model for prediction...', tone: 'status' },
      {
        text: 'Running[eval]:   0%|          | 0/2 [00:00<?, ?benchmark/s]',
        tone: 'muted',
        progress: [
          'Running[eval]:   0%|          | 0/2 [00:00<?, ?benchmark/s]',
          'Running[eval]:  50%|█████     | 1/2 [00:07<00:06,  6.10s/benchmark]',
          'Running[eval]: 100%|██████████| 2/2 [00:07<00:00,  3.27s/benchmark]',
        ],
      },
      {
        text: '2026-09-11 10:32:25 - evalscope - INFO: Evaluating[gsm8k] 100%| 5/5 [Elapsed: 00:06 < Remaining: 00:00]',
        tone: 'status',
      },
      { text: '2026-09-11 10:32:25 - evalscope - INFO: gsm8k report table:', tone: 'status' },
      {
        text: `┌───────────┬───────────┬────────────┬──────────┬───────┬─────────┐
│ Model     │ Dataset   │ Metric     │ Subset   │   Num │ Score   │
├───────────┼───────────┼────────────┼──────────┼───────┼─────────┤
│ qwen-plus │ GSM8K     │ Accuracy ↑ │ main     │     5 │ 100%    │
└───────────┴───────────┴────────────┴──────────┴───────┴─────────┘`,
        tone: 'muted',
        block: 'table',
      },
      { text: '2026-09-11 10:32:25 - evalscope - INFO: gsm8k perf table:', tone: 'status' },
      {
        text: `Model      Dataset      Num  Avg Lat    Avg TTFT    Avg TPOT    Avg Thpt       Avg In    Avg Out
---------  ---------  -----  ---------  ----------  ----------  -----------  --------  ---------
qwen-plus  GSM8K          5  3.768 s    587.2 ms    17.9 ms     49.62 tok/s       659        187`,
        tone: 'muted',
        block: 'table',
      },
      { text: '2026-09-11 10:32:25 - evalscope - INFO: Start loading benchmark dataset: arc', tone: 'status' },
      {
        text: '2026-09-11 10:32:25 - evalscope - WARNING: arc: 10 samples to evaluate (2 subsets, --limit=5 per subset).',
        tone: 'muted',
      },
      { text: "2026-09-11 10:32:25 - evalscope - INFO: Subsets of arc: ['ARC-Easy', 'ARC-Challenge']", tone: 'status' },
      {
        text: '2026-09-11 10:32:26 - evalscope - INFO: Evaluating[arc] 100%| 10/10 [Elapsed: 00:01 < Remaining: 00:00]',
        tone: 'status',
      },
      { text: '2026-09-11 10:32:26 - evalscope - INFO: arc report table:', tone: 'status' },
      {
        text: `┌───────────┬───────────┬────────────┬───────────────┬───────┬─────────┐
│ Model     │ Dataset   │ Metric     │ Subset        │   Num │ Score   │
├───────────┼───────────┼────────────┼───────────────┼───────┼─────────┤
│ qwen-plus │ ARC       │ Accuracy ↑ │ ARC-Easy      │     5 │ 100%    │
├───────────┼───────────┼────────────┼───────────────┼───────┼─────────┤
│ qwen-plus │ ARC       │ Accuracy ↑ │ ARC-Challenge │     5 │ 80%     │
├───────────┼───────────┼────────────┼───────────────┼───────┼─────────┤
│ qwen-plus │ ARC       │ Accuracy ↑ │ OVERALL       │    10 │ 90%     │
└───────────┴───────────┴────────────┴───────────────┴───────┴─────────┘`,
        tone: 'muted',
        block: 'table',
      },
      { text: '2026-09-11 10:32:26 - evalscope - INFO: arc perf table:', tone: 'status' },
      {
        text: `Model      Dataset      Num  Avg Lat    Avg TTFT    Avg TPOT    Avg Thpt      Avg In    Avg Out
---------  ---------  -----  ---------  ----------  ----------  ----------  --------  ---------
qwen-plus  ARC           10  0.496 s    400.8 ms    31.8 ms     8.06 tok/s       116          4`,
        tone: 'muted',
        block: 'table',
      },
      {
        text: '2026-09-11 10:32:27 - evalscope - INFO: HTML report generated: outputs/website-terminal-demo/20260911_103219/reports/report.html',
        tone: 'path',
      },
      {
        text: "2026-09-11 10:32:27 - evalscope - INFO: Finished evaluation for qwen-plus on ['gsm8k', 'arc']",
        tone: 'status',
      },
      {
        text: '2026-09-11 10:32:27 - evalscope - INFO: Output directory: outputs/website-terminal-demo/20260911_103219',
        tone: 'path',
      },
    ],
    summary: 'GSM8K + ARC · 15 samples · streaming enabled',
    recordedAt: '2026-09-11 10:32:19 +08:00',
    outputPath: 'outputs/website-terminal-demo/20260911_103219',
  },
  {
    id: 'perf',
    icon: 'tabler:gauge',
    label: 'perf',
    command: `evalscope perf \\
  --model qwen-plus --api openai \\
  --url https://dashscope.aliyuncs.com/compatible-mode/v1 \\
  --api-key "$DASHSCOPE_API_KEY" \\
  --tokenizer-path /path/to/Qwen2.5-0.5B-Instruct \\
  --dataset random --min-prompt-length 1024 --max-prompt-length 1024 \\
  --max-tokens 100 --temperature 0 --stream \\
  --number 24 24 24 --parallel 1 2 4 --warmup-num 3 \\
  --outputs-dir outputs/website-terminal-demo`,
    lines: [
      { text: '$ evalscope perf \\', tone: 'command' },
      { text: '  --model qwen-plus --api openai \\', tone: 'command' },
      { text: '  --url https://dashscope.aliyuncs.com/compatible-mode/v1 \\', tone: 'command' },
      { text: '  --api-key "$DASHSCOPE_API_KEY" \\', tone: 'command' },
      { text: '  --tokenizer-path /path/to/Qwen2.5-0.5B-Instruct \\', tone: 'command' },
      { text: '  --dataset random --min-prompt-length 1024 --max-prompt-length 1024 \\', tone: 'command' },
      { text: '  --max-tokens 100 --temperature 0 --stream \\', tone: 'command' },
      { text: '  --number 24 24 24 --parallel 1 2 4 --warmup-num 3 \\', tone: 'command' },
      { text: '  --outputs-dir outputs/website-terminal-demo', tone: 'command' },
      {
        text: '2026-09-11 10:32:41 - evalscope - WARNING: URL "https://dashscope.aliyuncs.com/compatible-mode/v1" has no endpoint path, auto-appended "/chat/completions".',
        tone: 'muted',
      },
      {
        text: '2026-09-11 10:32:41 - evalscope - INFO: Save the result to: outputs/website-terminal-demo/20260911_103241/qwen-plus',
        tone: 'path',
      },
      { text: '2026-09-11 10:32:45 - evalscope - INFO: Test connection successful.', tone: 'status' },
      {
        text: '2026-09-11 10:32:45 - evalscope - INFO: Warmup enabled: 3 warmup requests (total: 27, benchmark: 24)',
        tone: 'status',
      },
      {
        text: 'Running[perf]:   0%|          | 0/3 [00:00<?, ?it/s]',
        tone: 'muted',
        progress: [
          'Running[perf]:   0%|          | 0/3 [00:00<?, ?it/s]',
          'Running[perf]:  33%|███▎      | 1/3 [01:36<02:03, 61.68s/it]',
          'Running[perf]:  67%|██████▋   | 2/3 [01:36<00:45, 45.59s/it]',
          'Running[perf]: 100%|██████████| 3/3 [01:54<00:00, 33.18s/it]',
        ],
      },
      { text: perfSummary, tone: 'muted', block: 'table' },
      {
        text: '2026-09-11 10:34:36 - evalscope - INFO: Performance summary saved to: outputs/website-terminal-demo/20260911_103241/qwen-plus/performance_summary.txt',
        tone: 'path',
      },
      {
        text: '2026-09-11 10:34:36 - evalscope - INFO: HTML report generated: outputs/website-terminal-demo/20260911_103241/qwen-plus/perf_report.html',
        tone: 'path',
      },
    ],
    summary: 'Random workload · 3 concurrency tiers · 72 streamed requests',
    recordedAt: '2026-09-11 10:32:41 +08:00',
    outputPath: 'outputs/website-terminal-demo/20260911_103241/qwen-plus',
  },
  {
    id: 'service',
    icon: 'tabler:server',
    label: 'service',
    command: `evalscope service \\
  --host 127.0.0.1 --port 9000 \\
  --outputs outputs/website-terminal-demo`,
    lines: [
      { text: '$ evalscope service \\', tone: 'command' },
      { text: '  --host 127.0.0.1 --port 9000 \\', tone: 'command' },
      { text: '  --outputs outputs/website-terminal-demo', tone: 'command' },
      { text: '2026-09-11 10:12:02 - evalscope - INFO: Starting EvalScope service on 127.0.0.1:9000', tone: 'status' },
      { text: '2026-09-11 10:12:02 - evalscope - INFO: Available endpoints:', tone: 'status' },
      {
        text: '2026-09-11 10:12:02 - evalscope - INFO:   GET  /health                         - Health check',
        tone: 'status',
      },
      {
        text: '2026-09-11 10:12:02 - evalscope - INFO:   POST /api/v1/eval/invoke             - Run model evaluation task (blocking)',
        tone: 'status',
      },
      {
        text: '2026-09-11 10:12:02 - evalscope - INFO:   GET  /api/v1/eval/report             - Get HTML evaluation report',
        tone: 'status',
      },
      {
        text: '2026-09-11 10:12:02 - evalscope - INFO:   POST /api/v1/perf/invoke             - Run performance benchmark task (blocking)',
        tone: 'status',
      },
      {
        text: '2026-09-11 10:12:02 - evalscope - INFO:   GET  /api/v1/perf/report             - Get HTML performance benchmark report',
        tone: 'status',
      },
      { text: '2026-09-11 10:12:02 - evalscope - INFO: Dashboard: http://127.0.0.1:9000/dashboard', tone: 'path' },
      { text: '🌐 EvalScope Dashboard: http://127.0.0.1:9000/dashboard', tone: 'path' },
      { text: " * Serving Flask app 'evalscope.service.app'", tone: 'status' },
      { text: ' * Debug mode: off', tone: 'status' },
      { text: ' * Running on http://127.0.0.1:9000', tone: 'status' },
      { text: '2026-09-11 10:12:02,087 - werkzeug - INFO: Press CTRL+C to quit', tone: 'muted' },
      { text: '$ curl --fail --silent http://127.0.0.1:9000/health', tone: 'command' },
      { text: '{"service":"evalscope","status":"ok","timestamp":"2026-09-11T10:12:13.572514"}', tone: 'result' },
      {
        text: '2026-09-11 10:12:13,572 - werkzeug - INFO: 127.0.0.1 - - [11/Sep/2026 10:12:13] "GET /health HTTP/1.1" 200 -',
        tone: 'status',
      },
    ],
    summary: 'Dashboard available · /health returned 200 OK',
    recordedAt: '2026-09-11 10:12:02 +08:00',
    outputPath: 'outputs/website-terminal-demo',
  },
];
