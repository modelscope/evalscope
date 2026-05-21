# External Agent Bridge — layering & extension guide

This package proxies inbound provider-native HTTP requests (Anthropic
Messages, future: OpenAI Chat / Responses) from an external CLI agent
into EvalScope's `Model.generate_async`, then renders the `ModelOutput`
back into the agent's expected response format.

This file documents **how to add another wire protocol** without
turning this directory (or `evalscope/models/utils/`) into a tangle.

## Layering

```
        ┌────────────────────────────────────────────┐
        │ External CLI agent (Claude Code, etc.)     │
        └────────────────────────────────────────────┘
                          │  HTTP (provider-native)
                          ▼
        ┌────────────────────────────────────────────┐
        │  server.py        HTTP routing + sessions  │  transport
        ├────────────────────────────────────────────┤
        │  translate_<protocol>.py                   │  translation
        │    dict ↔ ChatMessage ↔ dict (both ways)   │
        ├────────────────────────────────────────────┤
        │  sse_<protocol>.py    (only if streaming)  │  stream replay
        └────────────────────────────────────────────┘
                          │  ChatMessage[] / ModelOutput
                          ▼
        ┌────────────────────────────────────────────┐
        │  EvalScope Model.generate_async            │
        └────────────────────────────────────────────┘
```

`ChatMessage` is the **only** IR. Cross-provider routing (e.g. an
Anthropic-format request served by an OpenAI backend) falls out
automatically: parse to `ChatMessage`, hand to `Model`, render the
`ModelOutput` in the requested wire format. No direct
Anthropic↔OpenAI translation exists or should be added.

## Rules for adding a new protocol

1. **One file per wire protocol**, both directions in it.
   `translate_openai_chat.py` owns `dict → ChatMessage` *and*
   `ModelOutput → dict` for `/v1/chat/completions`. Do **not** split
   by direction (`*_in.py` / `*_out.py`).

2. **Do not merge translation into `evalscope/models/utils/<provider>.py`.**
   `models/utils/` converts `ChatMessage` to/from the provider's
   *typed SDK objects* for outbound EvalScope→provider calls. The
   bridge consumes raw HTTP dicts on the inbound side. Different
   data shapes, different callers — keep them apart.

3. **Reuse `models/utils/` helpers at block granularity** when the
   logic is identical (e.g. assistant content-block parsing). Import
   the small helper rather than copy-pasting. Don't pull in the
   whole `MessageParam` machinery — the dict path doesn't need it.

4. **Stop-reason maps stay per-protocol.** The Anthropic↔EvalScope
   tables are not symmetric (`refusal`, `stop_sequence`, etc. don't
   round-trip cleanly), and adding OpenAI will introduce its own
   asymmetries. Each `translate_*.py` keeps its own table.

5. **Register the route in `server.py`.** Each route is a thin
   handler: read body → call the `translate_*` parser → run the
   model → call the `translate_*` renderer → write response.

## Streaming

Streaming is intentionally **not** supported in P0 (matches inspect-ai
`agent_bridge` behaviour, which never added it). When/if added, follow
the `inspect_sandbox_tools/_agent_bridge/proxy.py` pattern: the model
layer still returns a complete `ModelOutput`; the server replays it as
a `sse_<protocol>.py`-formatted event stream. Do not push streaming
into the translation or model layers.

## Why not just monkey-patch the SDK?

`inspect_ai/agent/_bridge/` monkey-patches Anthropic / OpenAI / Google
SDKs at the transport layer. That only works when the agent runs
in-process with EvalScope. Our target (external CLI agents like Claude
Code) is a separate subprocess, so we must terminate real HTTP — hence
this server. If you ever need in-process bridging too, build it as a
sibling package; do not blur the two transports together.

## File map

| File | Purpose |
| --- | --- |
| `server.py` | `ModelProxyServer`, `TrialSession`, HTTP routing |
| `translate_anthropic.py` | `/v1/messages` dict ↔ `ChatMessage` ↔ dict |
| `sse_anthropic.py` | Anthropic SSE event construction (when streaming) |
| `trace_recorder.py` | Per-trial trace capture for `AgentTrace` |
