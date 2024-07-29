from dataclasses import dataclass

@dataclass
class ServerSentEvent(object):
    def __init__(self, data='', event=None, id=None, retry=None):
        self.data = data
        self.event = event
        self.id = id 
        self.retry = retry

    @classmethod
    def decode(cls, line):
        """ Decode line to ServerSentEvent


        Args:
            line (str): The line.

        Return:
            ServerSentEvent (obj:`ServerSentEvent`): The ServerSentEvent object.

        """
        if not line:
            return None
        sse_msg = cls()
        # format data:xxx
        field_type, _, field_value = line.partition(":")
        if field_value.startswith(" "): # compatible with openai api
            field_value = field_value[1:]
        if field_type == "event":
            sse_msg.event = field_value
        elif field_type == "data":
            field_value = field_value.rstrip()
            sse_msg.data = field_value
        elif field_type == "id":
            sse_msg.id = field_value
        elif field_type == "retry":
            sse_msg.retry = field_value
        else:
            pass

        return sse_msg

