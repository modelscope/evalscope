from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PromptData:
    data: List[str]
    index: Optional[int] = 0
    system_prompt: Optional[str] = None
    multiple_choices: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        if self.multiple_choices is None:
            return {
                'data': self.data,
                'index': self.index,
                'system_prompt': self.system_prompt,
            }
        else:
            return {
                'data': self.data,
                'index': self.index,
                'system_prompt': self.system_prompt,
                'multiple_choices': self.multiple_choices,
            }
