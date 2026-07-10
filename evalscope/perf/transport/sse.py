import codecs
from typing import List


class SSEDecoder:
    """Incrementally decode UTF-8 Server-Sent Events."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder('utf-8')()
        self._buffer = ''

    def feed(self, chunk: bytes) -> List[str]:
        self._buffer += self._decoder.decode(chunk, final=False).replace('\r\n', '\n')
        events = []
        while '\n\n' in self._buffer:
            message, self._buffer = self._buffer.split('\n\n', 1)
            message = message.strip()
            if message:
                events.append(message)
        return events

    def finish(self) -> List[str]:
        self._buffer += self._decoder.decode(b'', final=True)
        message = self._buffer.strip()
        self._buffer = ''
        return [message] if message else []
