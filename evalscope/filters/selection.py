from collections import Counter
from typing import List

from evalscope.api.filter import Filter
from evalscope.api.registry import register_filter


@register_filter('take_first')
class TakeFirstFilter(Filter):

    def __init__(self) -> None:
        """
        Can define custom behavior here, if an individual instantiation of a Filter class should have state.
        """

    def apply(self, instance: List[str]) -> List[str]:
        """
        Take only the first response from the instance list.
        """
        return [instance[0]] if instance else []


@register_filter('take_first_k')
class TakeKFilter(Filter):

    def __init__(self, **kwargs) -> None:
        self.k = kwargs.pop('k')
        super().__init__(**kwargs)

    def apply(self, instance: List[str]) -> List[str]:
        """
        Take the first k responses from the instance list.
        """
        assert len(instance) >= self.k, (
            f'Need at least {self.k} responses to take first {self.k}, but got {len(instance)} only!'
        )
        return instance[:self.k]


@register_filter('majority_vote')
class MajorityVoteFilter(Filter):

    def __init__(self) -> None:
        """
        Can define custom behavior here, if an individual instantiation of a Filter class should have state.
        """

    def apply(self, instance: List[str]) -> List[str]:
        """
        Select the response that occurs most frequently in the instance list.
        """
        if not instance:
            return []

        counts = Counter(instance)
        vote = counts.most_common(1)[0][0]
        return [vote]
