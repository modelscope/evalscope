# Official MultiPL-E Java regression fixtures

These two complete, unchanged dataset rows come from
[nuprl/MultiPL-E](https://huggingface.co/datasets/nuprl/MultiPL-E/tree/28441b6024e71d4a1c1c0f6bf171c935cd5a43f2),
revision `28441b6024e71d4a1c1c0f6bf171c935cd5a43f2`, retrieved on 2026-11-02:

- `humaneval_53_add.json`: `humaneval-java`, test split, `HumanEval_53_add`.
- `mbpp_3_is_not_prime.json`: `mbpp-java`, test split, `mbpp_3_is_not_prime`.

The prompt, tests, stop tokens, and source metadata are retained without edits.
In particular, each `tests` suffix starts with the closure of an unfinished target
method, followed by an assertion-based `main` and the enclosing class closure.
The tests supply complete, self-contained arithmetic solutions and exercise every
original assertion in these rows. No fixture is downloaded while tests run.

The [MultiPL-E dataset card](https://huggingface.co/datasets/nuprl/MultiPL-E)
identifies the dataset license as MIT. HumanEval's original source is
[OpenAI HumanEval](https://github.com/openai/human-eval), copyright OpenAI under
the MIT license retained in `LICENSE.humaneval`. MBPP's original source is
[Google Research MBPP](https://github.com/google-research/google-research/tree/master/mbpp),
released with Austin et al., *Program Synthesis with Large Language Models* (2021).
Its [dataset card](https://huggingface.co/datasets/google-research-datasets/mbpp)
identifies [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
MultiPL-E translated the original Python tasks to Java; these translated rows
have not been changed here.

The separate [MultiPL-E project license](https://github.com/nuprl/MultiPL-E/blob/3025a531af7450e7df8b96fe0440e9804480bbad/LICENSE)
is retained in `LICENSE.multipl-e` as an upstream notice. Its BSD terms and machine
learning restriction concern the project repository and should not be confused
with the MIT declaration on the Hugging Face dataset card.

The optional local Java runner creates a temporary `org.javatuples` package marker
to resolve the shared assembler's unused wildcard import. It provides no tuple
classes or APIs, tests no external JAR, and is never added to production output.
Real `javac` compilation and `java -ea` execution validate the generated program;
those integration checks explicitly skip without a local JDK. Deterministic
assembly checks always run.
