.. image:: _static/images/evalscope_logo.png

.. _pypi_downloads: https://pypi.org/project/evalscope
.. _github_pr: https://github.com/modelscope/evalscope/pulls

.. raw:: html

   <p align="center">
   <a href="https://badge.fury.io/py/evalscope"><img src="https://badge.fury.io/py/evalscope.svg" alt="PyPI version" height="18"></a>
   <a href="https://pypi.org/project/evalscope"><img alt="PyPI - Downloads" src="https://static.pepy.tech/badge/evalscope">
   </a>
   <a href='https://evalscope.readthedocs.io/en/latest/?badge=latest'>
      <img src='https://readthedocs.org/projects/evalscope-en/badge/?version=latest' alt='Documentation Status' />
   </a>
   </p>


Welcome to the EvalScope Tutorial!
==========================================
Getting Started with EvalScope
-------------------------------
To help users quickly get started, we recommend the following flow:

- For those who want to use EvalScope, we recommend first reading the QuickStart_ section to set up the environment and initiate a mini-experiment to familiarize yourself with the process.

- For some basic usages, we suggest users read the Tutorials_, which include how to perform offline evaluations with EvalScope, how to use Arena models for evaluations, how to utilize other evaluation backends, and how to use the model service stress testing tool.

- If you wish to customize more modules, such as adding datasets and models, we provide the AdvancedTutorials_.

- Additionally, we offer Third-PartyTools_ to help users quickly evaluate models, such as using ToolBench for evaluations.

- Finally, we provide BestPractices_ to assist users with evaluations, such as how to use Swift for evaluations.

We always welcome users' PRs and Issues to improve EvalScope.

.. _QuickStart:
.. toctree::
   :maxdepth: 1
   :caption: Quick Start
   
   get_started/introduction.md
   get_started/installation.md
   get_started/basic_usage.md

.. _Tutorials:
.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   
   user_guides/offline_evaluation.md
   user_guides/arena.md
   user_guides/opencompass_backend.md
   user_guides/vlmevalkit_backend.md
   user_guides/stress_test.md

.. _AdvancedTutorials:
.. toctree::
   :maxdepth: 1
   :caption: Advanced Tutorials
   
   advanced_guides/custom_dataset.md
   advanced_guides/custom_model.md

.. _Third-PartyTools:
.. toctree::
   :maxdepth: 1
   :caption: Third-Party Tools
   
   third_party/toolbench.md

.. _BestPractices:
.. toctree::
   :maxdepth: 1
   :caption: Best Practices
   
   best_practice/swift_integration.md
   best_practice/llm_full_stack.md
   best_practice/experiments.md

Index and Tables
==================
* :ref:`genindex`
* :ref:`search`