.. image:: _static/images/evalscope_logo.png

.. raw:: html

    <br>

欢迎来到 EvalScope 中文教程！
==========================================

上手路线
-------------------------------

为了用户能够快速上手，我们推荐以下流程：

- 对于想要使用 EvalScope 的用户，我们推荐先阅读 快速开始_ 部分来设置环境，并启动一个迷你实验熟悉流程。

- 对于一些基础使用，我们建议用户阅读 教程_ ，包括 如何使用 EvalScope 进行离线评测、如何使用 竞技场模式 进行评测、如何使用其他评测后端、如何使用模型服务压测工具。

- 若您想进行更多模块的自定义，例如增加数据集和模型，我们提供了 进阶教程_ 。

- 此外，我们提供了 第三方工具_ 来帮助用户快速评测模型，例如使用 ToolBench 进行评测。

- 最后，我们提供了 最佳实践_ 来帮助用户进行评测，例如如何使用 Swift 进行评测。

我们始终非常欢迎用户的 PRs 和 Issues 来完善 EvalScope

目录
-------------
.. _快速开始:
.. toctree::
   :maxdepth: 1
   :caption: 快速开始
   
   get_started/introduction.md
   get_started/installation.md
   get_started/basic_usage.md
   get_started/supported_dataset.md

.. _教程:
.. toctree::
   :maxdepth: 1
   :caption: 教程
   
   user_guides/offline_evaluation.md
   user_guides/arena.md
   user_guides/opencompass_backend.md
   user_guides/vlmevalkit_backend.md
   user_guides/stress_test.md

.. _进阶教程:
.. toctree::
   :maxdepth: 1
   :caption: 进阶教程
   
   advanced_guides/custom_dataset.md
   advanced_guides/custom_model.md

.. _第三方工具:
.. toctree::
   :maxdepth: 1
   :caption: 第三方工具
   
   third_party/toolbench.md
   third_party/longwriter.md

.. _最佳实践:
.. toctree::
   :maxdepth: 1
   :caption: 最佳实践
   
   best_practice/swift_integration.md
   best_practice/llm_full_stack.md
   best_practice/experiments.md
