# Official MultiPL-E Java regression fixtures

These two complete, unchanged dataset rows come from
[nuprl/MultiPL-E](https://huggingface.co/datasets/nuprl/MultiPL-E/tree/28441b6024e71d4a1c1c0f6bf171c935cd5a43f2),
revision `28441b6024e71d4a1c1c0f6bf171c935cd5a43f2`, retrieved on 2026-11-02:

- `humaneval_53_add` in `cases.json`: `humaneval-java`, test split, `HumanEval_53_add`.
- `mbpp_3_is_not_prime` in `cases.json`: `mbpp-java`, test split, `mbpp_3_is_not_prime`.

The prompt, tests, stop tokens, and source metadata are retained without edits.
In particular, each `tests` suffix starts with the closure of an unfinished target
method, followed by an assertion-based `main` and the enclosing class closure.
The tests supply complete, self-contained arithmetic solutions and exercise every
original assertion in these rows. No fixture is downloaded while tests run.

The [MultiPL-E dataset card](https://huggingface.co/datasets/nuprl/MultiPL-E)
identifies the dataset license as MIT. HumanEval's original source is
[OpenAI HumanEval](https://github.com/openai/human-eval), copyright OpenAI under
the MIT license retained below. MBPP's original source is
[Google Research MBPP](https://github.com/google-research/google-research/tree/master/mbpp),
released with Austin et al., *Program Synthesis with Large Language Models* (2021).
Its [dataset card](https://huggingface.co/datasets/google-research-datasets/mbpp)
identifies [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
MultiPL-E translated the original Python tasks to Java; these translated rows
have not been changed here.

The separate [MultiPL-E project license](https://github.com/nuprl/MultiPL-E/blob/3025a531af7450e7df8b96fe0440e9804480bbad/LICENSE)
is retained below as an upstream notice. Its BSD terms and machine
learning restriction concern the project repository and should not be confused
with the MIT declaration on the Hugging Face dataset card.

## HumanEval license

The MIT License

Copyright (c) OpenAI (https://openai.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

## MultiPL-E project license

BSD 3-Clause License with Machine Learning Restriction

Copyright (c) 2022, Northeastern University, Oberlin College, Roblox Inc,
Stevens Institute of Technology, University of Massachusetts Amherst, and
Wellesley College.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

4.  The contents of this repository may not be used as training data for any
    machine learning model, including but not limited to neural networks.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
