from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import HubType, Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate, answer_character

# Canonical PolyAI/banking77 ClassLabel order, also used by the mteb data-only mirror.
INTENT_LABELS = [
    'activate_my_card',
    'age_limit',
    'apple_pay_or_google_pay',
    'atm_support',
    'automatic_top_up',
    'balance_not_updated_after_bank_transfer',
    'balance_not_updated_after_cheque_or_cash_deposit',
    'beneficiary_not_allowed',
    'cancel_transfer',
    'card_about_to_expire',
    'card_acceptance',
    'card_arrival',
    'card_delivery_estimate',
    'card_linking',
    'card_not_working',
    'card_payment_fee_charged',
    'card_payment_not_recognised',
    'card_payment_wrong_exchange_rate',
    'card_swallowed',
    'cash_withdrawal_charge',
    'cash_withdrawal_not_recognised',
    'change_pin',
    'compromised_card',
    'contactless_not_working',
    'country_support',
    'declined_card_payment',
    'declined_cash_withdrawal',
    'declined_transfer',
    'direct_debit_payment_not_recognised',
    'disposable_card_limits',
    'edit_personal_details',
    'exchange_charge',
    'exchange_rate',
    'exchange_via_app',
    'extra_charge_on_statement',
    'failed_transfer',
    'fiat_currency_support',
    'get_disposable_virtual_card',
    'get_physical_card',
    'getting_spare_card',
    'getting_virtual_card',
    'lost_or_stolen_card',
    'lost_or_stolen_phone',
    'order_physical_card',
    'passcode_forgotten',
    'pending_card_payment',
    'pending_cash_withdrawal',
    'pending_top_up',
    'pending_transfer',
    'pin_blocked',
    'receiving_money',
    'Refund_not_showing_up',
    'request_refund',
    'reverted_card_payment?',
    'supported_cards_and_currencies',
    'terminate_account',
    'top_up_by_bank_transfer_charge',
    'top_up_by_card_charge',
    'top_up_by_cash_or_cheque',
    'top_up_failed',
    'top_up_limits',
    'top_up_reverted',
    'topping_up_by_card',
    'transaction_charged_twice',
    'transfer_fee_charged',
    'transfer_into_account',
    'transfer_not_received_by_recipient',
    'transfer_timing',
    'unable_to_verify_identity',
    'verify_my_identity',
    'verify_source_of_funds',
    'verify_top_up',
    'virtual_card_not_working',
    'visa_or_mastercard',
    'why_verify_identity',
    'wrong_amount_of_cash_received',
    'wrong_exchange_rate_for_cash_withdrawal',
]


@register_benchmark(
    BenchmarkMeta(
        name='banking77',
        pretty_name='BANKING77',
        dataset_id='mteb/banking77',
        dataset_hub=HubType.HUGGINGFACE,
        default_subset='default',
        subset_list=['default'],
        eval_split='test',
        train_split='train',
        evaluation_version='v1.0',
        supports_choice=True,
        choice_instructions='Which banking support intent is expressed by the message in `question`?',
        tags=[Tags.MULTIPLE_CHOICE],
        metric_list=['acc'],
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
        description="""
## Overview

Fine-grained banking support intent classification with the complete 77-class taxonomy.

## Task Description

- **Task Type**: Single-choice classification
- **Input**: Banking support message
- **Output**: One of 77 banking intents
- **Domain**: Customer support

## Key Features

- Public dataset: `mteb/banking77` on Hugging Face
- Preserves the source labels and complete task context
- Supports chat generation and text System One Choice models

## Evaluation Notes

- Reports accuracy, not macro-F1. All 77 intents remain available for every sample; no candidate pruning or added out-of-scope intent.
- Defaults to 0-shot; training examples can be configured where a training split is available
- System prompts become Choice task instructions, without a native chat-role hierarchy
- Evaluation semantics version: v1.0
""",
    )
)
class Banking77Adapter(MultiChoiceAdapter):
    """Classify support messages over the complete canonical intent inventory."""

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        label = int(record['label'])
        if not 0 <= label < len(INTENT_LABELS):
            raise ValueError('BANKING77 label is outside the 77-class inventory.')
        label_text = record.get('label_text')
        if label_text is not None and label_text != INTENT_LABELS[label]:
            raise ValueError('BANKING77 numeric and text labels disagree with the canonical inventory.')
        return Sample(
            input=record['text'],
            choices=[label.replace('_', ' ') for label in INTENT_LABELS],
            target=answer_character(label),
        )
