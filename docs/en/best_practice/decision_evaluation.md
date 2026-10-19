# Let AI Make the Call: Evaluating Decision Models with EvalScope

Open the comments on any product and the first screen often isn't real users: "earn 500 a day from home", "DM me for the insider price", "like for cash back". Whether you are just browsing or building a product that shows comments to others, you want the same thing: quickly tell what's worth reading from what should be collapsed.

Hand this to a general-purpose LLM and it will likely write three paragraphs of analysis before politely reaching a verdict. All you wanted was two things: an answer, and how sure it is of that answer.

This is where decision models fit. They don't write essays; they take a position: return a category, a score, or a plain "yes / no", with a probability for every option. [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) and [Clef-Flash](https://developers.cloudflare.com/changelog/post/2026-10-01-clef-workers-ai/) are such models. Anything that needs explanation or discussion can still go to a chat model afterwards.

This guide uses [**EvalScope**](https://github.com/modelscope/evalscope) to pose the questions, collect the answers and grade them, checking three things along the way: whether the answers are right, whether the probabilities are usable, and whether responses are fast enough. We start with a public question set to walk through the flow; the extension sections then cover your own data and self-hosting.

## 1. Three kinds of questions a decision model answers

Comment filtering is just one use; message classification, content scoring, and judging whether two pieces of text are related can all be handed to it. These jobs come down to three common question kinds.

Here is a real comment-classification question. The comment (originally in Chinese) reads:

> Earn 500+ a day from your phone at home: cash back for paid reviews, open to stay-at-home parents and students, limited slots, first come first served!

The reference answer stays with EvalScope for grading and is never sent to the model. Below is the question as written (excerpted from the public [OpenDecisions](https://modelscope.cn/datasets/evalscope/OpenDecisions) set):

![A comment-classification question with options A, B, C and D](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/decision_evaluation/public-question.png)

Pick one out of promotional spam, normal review, abuse and chit-chat. This "choose one out of many" kind is called **Choice**. The model returns a probability per option, which suits message classification and request routing.

Two more kinds are just as common: **Noul** is a yes/no judgment: it asks whether a condition holds and returns the probability that it does, such as "should this comment be collapsed"; **Score** is a graded rating over ordered levels, suited to judgments with degrees, like completeness or the severity of disruption.

The key to writing a good question is stating the subject, the rules and the allowed answers clearly: use Choice for categories, Noul for whether a condition holds, and Score for degree-based ratings. Rules should be concrete: "comments discussing price may stay; comments where merchants solicit customers should be collapsed" is far clearer than "filter out bad content".

## 2. One command to evaluate on public questions

Our test run uses `decision-model-preview`, served through DashScope's System One endpoint (the common request format for decision models): questions go in, judgments and probabilities come back. Any other service compatible with this interface works with the same commands: just swap in your own model name, API URL and key.

Run the following commands in order. The public questions come from [OpenDecisions](https://modelscope.cn/datasets/evalscope/OpenDecisions), which covers over two thousand questions across all three kinds; EvalScope downloads the data, poses the questions and grades the answers itself:

```bash
# Install dependencies (including the perf and Web UI components)
pip install 'evalscope[perf,service]'

# Connection settings: replace with your service's model, URL and key
export DECISION_MODEL='decision-model-preview'
export DECISION_API_URL='https://trial.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'
export DECISION_API_KEY='replace-with-your-service-key'

# Evaluate on the public questions
evalscope eval \
  --model "$DECISION_MODEL" \
  --eval-type systemone_api \
  --api-url "$DECISION_API_URL" \
  --api-key "$DECISION_API_KEY" \
  --datasets open_decisions
```

When the run finishes, the terminal prints the per-kind scores and composite metrics. We ran our tests on 59 content-moderation questions picked from the full set, and the highlights from that report are below; the command above evaluates all 2352 questions by default.

| Question kind | Samples | Accuracy ↑ | MAE ↓ |
| --- | ---: | ---: | ---: |
| Choice | 22 | 63.6% | — |
| Noul | 19 | 89.5% | — |
| Score (3 levels, 0–2) | 2 | 100% | 0.015 |
| Score (4 levels, 0–3) | 16 | 56.3% | 0.6462 |

For Choice, the option with the highest probability wins; Noul treats 0.5 as the default yes/no cutoff, and accuracy is simply the share of correct answers. Score accuracy checks whether the top-probability level is the right one, while MAE measures how far the probability-weighted score drifts from the reference level on average, so lower is better. Score questions with different scales are listed separately.

The composite scores for the same 59 questions are below; higher is better on all three:

| Composite metric | Score ↑ |
| --- | ---: |
| Intelligence (judgment ability) | 39.1% |
| Calibration (probability reliability) | 53.9% |
| Capability (overall performance) | 46.5% |

Don't read these three composite metrics as accuracy. **Intelligence** measures how much better the model does than a baseline: random guessing for Choice and Noul, and usually the middle level for Score. For example, random guessing on yes/no questions gives 50% accuracy, so reaching 80% converts to an ability score of `(80% - 50%) / (100% - 50%) = 60%`. Questions are weighted by difficulty, and the per-kind ability scores average into this run's **39.1%**.

**Calibration** checks whether the stated probabilities match actual outcomes. For example, across the predictions where the model assigned an option an 80% probability, that option should actually hold about 80% of the time. The computation groups predictions by probability and measures this deviation; the smaller the deviation, the higher the score. Choice uses hard-difficulty questions, and Score also factors in how far the whole probability distribution sits from the reference level. The per-kind scores average out to **53.9%** this run.

**Capability** is simply the mean of the two, about `(39.1% + 53.9%) / 2 = 46.5%` here. It is an overall performance score, not the share of questions answered correctly.

For a quick first run, add `--limit 50` to take only the first 50 questions. Evaluation results land in `./outputs` by default, where the Web UI below reads them directly.

## 3. Read the results in the Web UI

When the evaluation finishes, start the Web UI and open the model report on the Evaluations page at `http://127.0.0.1:9000`:

```bash
evalscope service
```

Start with the overall numbers on the report page: accuracy for Choice and Noul questions, mean absolute error (MAE) for Score questions. Once the totals look right, drill into individual questions.

Open a question in Predictions to see the model's pick next to the reference answer, along with the probability of each option. Below is the real result for the Chinese comment question from earlier: the model chose promotional spam, and the page marks the matching reference answer.

![The prediction, reference answer and per-option probabilities for the public comment question in the Web UI](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/decision_evaluation/prediction-probabilities.jpg)

In the figure, `spam_or_scam_ad` corresponds to promotional spam; the other three options are normal review, abuse and chit-chat. Beyond whether the pick is right, the probabilities also show whether the model hesitated between options. To inspect mistakes, filter for wrong answers and read them side by side with the questions and probabilities.

## 4. Response speed matters too

Automated classification and filtering call the model repeatedly, so waiting time shapes the experience too. The load test needs no hand-prepared data either; it reuses the same public question set: pass `open_decisions` as the dataset and the questions load automatically and are sent one by one:

```bash
evalscope perf --api systemone --model "$DECISION_MODEL" \
  --url "$DECISION_API_URL" --api-key "$DECISION_API_KEY" \
  --dataset open_decisions \
  --parallel 1 4 --number 24 24 --warmup-num 2 --no-stream
```

The command sends 24 requests at each of the two concurrency levels, 1 and 4, with 2 warmup requests before the measured run. The figure below comes from a short load test with Chinese comment requests; running the command above replays the same flow on the public questions. In that test, raising concurrency from 1 to 4 pushed throughput to over three times, while p95 latency rose only slightly. The left side is the actual report; the right compares the two concurrency levels. p95 is the latency that 95% of requests stay within.

![The performance report for this run, with successful throughput and p95 latency at both concurrency levels](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/decision_evaluation/performance-comparison.png)

When reading a load-test report, check the success rate first, because speed with a high failure rate doesn't count; then look at latency and throughput. The full records open directly on the Web UI's Performance page, with no path switch needed: perf results also land in the default `./outputs` directory. Before wiring this into your own application, rerun with a larger request volume to see sustained behavior; when switching models, keep the same questions and test setup so runs stay comparable.

## 5. Going further: use your own data

Public questions are a shared reference; whether the model fits your use case depends on your own data. The data doesn't have to be large: a few records for each typical situation you run into are already worth evaluating. Organize the content to judge into the [custom data format](../advanced_guides/custom_dataset/decision.md), save it locally, and point `general_decision` at it:

```bash
evalscope eval \
  --model "$DECISION_MODEL" \
  --eval-type systemone_api \
  --api-url "$DECISION_API_URL" \
  --api-key "$DECISION_API_KEY" \
  --datasets general_decision \
  --dataset-args '{"general_decision":{"local_path":"./decision-data"}}'
```

One record per line of JSON, with the material, the question and the expected answer all in one line:

```json
{"id": "fold-1", "state": "the comment text", "question": {"type": "noul", "instructions": "Should this comment be collapsed?"}, "expected": "no"}
```

Once it finishes, open the Web UI as before and browse question by question in Predictions; among your own records, the wrong answers are the ones worth pausing on. Take this comment (originally in Chinese): the rules say it should stay, yet the model gave collapsing an 82% probability, the exact opposite of the reference answer.

> A commenter left the merchant's phone number here; I've reported it, and this comment does not repeat those contact details.

![The prediction, reference answer and yes/no probabilities for a misjudged Chinese comment in the Web UI](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/decision_evaluation/decision-error-probabilities.jpg)

High probability can still be wrong. Raising the collapse threshold usually reduces friendly fire, but may let more promotions slip through; the trade-off has to be measured on your own data. If mistakes cluster in one kind of content, add such samples, adjust the rules, or [train your own decision model](https://swift.readthedocs.io/en/latest/BestPractices/Decision.html) with ms-swift, then rerun the same questions to see the change. See the [decision model evaluation guide](../user_guides/systemone.md) for more probability metrics.

## 6. Going further: deploy on your own machine

If you'd rather not use a hosted service, you can run the open-source [Cloudflare/clef-flash](https://modelscope.cn/models/Cloudflare/clef-flash) on your own machine (about 9B parameters, based on Qwen3.5-9B, supports all three question kinds). You need a vLLM build with System One decision-head support, which is new enough that you install the nightly build; once installed, start the server, switch the connection settings and evaluate, all in this one block:

```bash
pip install -U modelscope
pip install -U --pre vllm --extra-index-url https://wheels.vllm.ai/nightly
VLLM_USE_MODELSCOPE=true vllm serve Cloudflare/clef-flash \
  --served-model-name clef-flash \
  --max-model-len 16384 --port 8000

# Once the server is up, switch the connection settings in another terminal and evaluate
export DECISION_MODEL='clef-flash'
export DECISION_API_URL='http://127.0.0.1:8000/v1'
export DECISION_API_KEY='EMPTY'
evalscope eval --model "$DECISION_MODEL" --eval-type systemone_api --api-url "$DECISION_API_URL" --api-key "$DECISION_API_KEY" --datasets open_decisions
```

From here on, the flow is exactly the one from sections 2 to 4: read the report, browse the predictions, load-test, just with the model now running on your own machine.

## Wrapping up

This guide followed one thread all the way through: put a decision model through EvalScope, check whether the answers are right on public questions, look through the probabilities question by question in the Web UI, load-test the response speed, then switch to your own data, or to a model hosted on your own machine, and run the same flow once more.

Whether it fits comes down to three things: whether the rules are written clearly, whether the mistakes are acceptable, and whether it is fast enough. Public questions are only a reference; the real answer sits in your own data. Write down the answers you agree with, run the evaluation, and see where the model judges differently: a report this close to your own use case answers "is this model right for me" far better than any public leaderboard. Want to try other question kinds? The [dataset list](../get_started/supported_dataset/llm.md) has more ready-made entries; to plug in another System One service, the [decision model API configuration](../user_guides/systemone.md) has you covered. Pick whichever entry fits and get started.
