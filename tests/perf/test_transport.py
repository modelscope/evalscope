from evalscope.perf.transport.sse import SSEDecoder


def test_sse_decoder_handles_split_utf8_and_multiple_events() -> None:
    decoder = SSEDecoder()
    raw = 'data: {"text":"你好"}\n\ndata: [DONE]\n\n'.encode()
    split = raw.index('好'.encode()) + 1
    assert decoder.feed(raw[:split]) == []
    assert decoder.feed(raw[split:]) == ['data: {"text":"你好"}', 'data: [DONE]']
    assert decoder.finish() == []
