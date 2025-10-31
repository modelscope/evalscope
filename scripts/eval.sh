evalscope eval \
 --model Qwen3-14B \
 --api-url http://127.0.0.1:13133/v1 \
 --api-key EMPTY \
 --datasets chembench \
 --eval-type openai_api \
 --generation-config '{"max_tokens": 7000}' \
 --limit 10
# --eval-batch-size 8 \
#  --datasets gsm8k mmlu ceval gpqa_diamond  # \
# --datasets internal_mof_information_extraction \