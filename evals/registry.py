# # Copyright (c) Alibaba, Inc. and its affiliates.
#
# DEFAULT_GROUP = 'default'
#
#
# class Registry:
#     # todo: ??? group_key, moldules, organizations, ...  --> wrapper: build_eval_alibaba, build_eval_openai, ...
#
#     def __init__(self, name: str):
#         self._name = name
#         self._modules = {DEFAULT_GROUP: {}}
#
#     def __repr__(self):
#         format_str = self.__class__.__name__ + f' ({self._name})\n'
#         for group_name, group in self._modules.items():
#             format_str += f'group_name={group_name}, '\
#                 f'modules={list(group.keys())}\n'
#
#         return format_str
#
#     # todo: ...