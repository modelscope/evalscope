import json
import time
from typing import Any, Dict, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.base import ApiPluginBase
from evalscope.perf.plugin.registry import register_api
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_api('enhanced_custom')
class EnhancedCustomPlugin(ApiPluginBase):
    """
    Example of an enhanced custom API plugin that demonstrates how to use
    the new HTTP client extension points for custom request/response processing.
    """

    def __init__(self, model_path: str):
        """Initialize the enhanced custom plugin

        Args:
            model_path (str): The model path for tokenizer initialization
        """
        super().__init__(model_path=model_path)
        if model_path is not None:
            try:
                from modelscope import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            except Exception as e:
                logger.warning(f'Failed to load tokenizer: {e}')
                self.tokenizer = None
        else:
            self.tokenizer = None

        # Initialize custom tracking variables
        self.request_count = 0
        self.response_times = []

    def build_request(self, messages: List[Dict], param: Arguments) -> Dict:
        """Build the request based on messages and parameters"""
        try:
            if param.query_template:
                query = json.loads(param.query_template)
                # Replace placeholders with actual values
                ApiPluginBase.replace_values(query, param.model, messages[0]['content'])
            else:
                # Default request format
                query = {
                    'messages': messages,
                    'model': param.model,
                    'max_tokens': param.max_tokens or 100,
                    'stream': param.stream
                }
            return query
        except Exception as e:
            logger.exception(f'Failed to build request: {e}')
            return None

    def parse_responses(self, responses: List, request: Any = None, **kwargs) -> tuple[int, int]:
        """Parse responses and return token counts"""
        if not responses:
            return 0, 0

        try:
            # Try to get usage from the last response
            last_response = json.loads(responses[-1])
            if 'usage' in last_response:
                return (last_response['usage'].get('prompt_tokens',
                                                   0), last_response['usage'].get('completion_tokens', 0))
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

        # Fallback to tokenizer-based calculation
        if self.tokenizer and request:
            try:
                input_text = request.get('messages', [{}])[0].get('content', '')
                output_text = ''.join([
                    json.loads(resp).get('choices', [{}])[0].get('message', {}).get('content', '') for resp in responses
                ])

                input_tokens = len(self.tokenizer.encode(input_text))
                output_tokens = len(self.tokenizer.encode(output_text))
                return input_tokens, output_tokens
            except Exception as e:
                logger.warning(f'Tokenizer-based parsing failed: {e}')

        return 0, 0

    # Enhanced HTTP client processing methods

    def preprocess_request(self, request_body: Dict) -> Dict:
        """
        Custom request preprocessing - add request tracking and custom headers
        """
        self.request_count += 1

        # Add custom tracking ID to request
        processed_request = request_body.copy()
        processed_request['_tracking_id'] = f'req_{self.request_count}_{int(time.time())}'

        # Add custom request formatting if needed
        if 'custom_format' in processed_request:
            # Example: transform request to custom API format
            custom_data = processed_request.pop('custom_format')
            processed_request.update(custom_data)

        logger.debug(f"Preprocessed request {self.request_count}: {processed_request.get('_tracking_id')}")
        return processed_request

    def process_response_content(self, content: str, is_streaming: bool = False) -> str:
        """
        Custom response content processing
        """
        try:
            # Parse and enhance response content
            if isinstance(content, str) and content.strip():
                response_data = json.loads(content)

                # Add processing timestamp
                response_data['_processed_at'] = time.time()

                # Custom streaming response handling
                if is_streaming:
                    response_data['_stream_chunk'] = True
                    logger.debug(f"Processing streaming chunk: {response_data.get('id', 'unknown')}")

                # Add custom response validation
                if 'choices' in response_data:
                    for choice in response_data['choices']:
                        if 'message' in choice and 'content' in choice['message']:
                            # Custom content filtering/transformation
                            choice['message']['content'] = self._filter_content(choice['message']['content'])

                return json.dumps(response_data, ensure_ascii=False)

        except json.JSONDecodeError:
            # Handle non-JSON content
            logger.debug(f'Processing non-JSON content: {content[:100]}...')
        except Exception as e:
            logger.warning(f'Response content processing failed: {e}')

        return content

    def process_error_response(self, error_data: Any) -> str:
        """
        Custom error response processing
        """
        try:
            if isinstance(error_data, dict):
                # Enhance error data with custom information
                enhanced_error = error_data.copy()
                enhanced_error['_error_processed_at'] = time.time()
                enhanced_error['_request_count'] = self.request_count

                # Custom error categorization
                error_type = self._categorize_error(error_data)
                enhanced_error['_error_category'] = error_type

                logger.debug(f'Processing error (category: {error_type}): {enhanced_error}')
                return json.dumps(enhanced_error, ensure_ascii=False)

        except Exception as e:
            logger.warning(f'Error response processing failed: {e}')

        # Fallback to string representation
        return json.dumps({'error': str(error_data), '_processed_at': time.time()}, ensure_ascii=False)

    def postprocess_responses(self, responses: List[str], original_request: Dict):
        """
        Custom post-processing after all responses are collected
        """
        try:
            # Calculate response timing
            if responses:
                response_time = time.time()
                self.response_times.append(response_time)

                # Log response statistics
                logger.debug(f"Request {original_request.get('_tracking_id', 'unknown')} "
                             f'generated {len(responses)} responses')

                # Custom response aggregation or validation
                self._validate_response_sequence(responses)

                # Custom metrics collection
                self._collect_custom_metrics(responses, original_request)

        except Exception as e:
            logger.warning(f'Response post-processing failed: {e}')

    def _filter_content(self, content: str) -> str:
        """
        Custom content filtering - example implementation
        """
        # Example: remove sensitive information, format content, etc.
        filtered = content.strip()

        # Add custom filtering logic here
        # e.g., remove PII, apply content policies, etc.

        return filtered

    def _categorize_error(self, error_data: Dict) -> str:
        """
        Categorize errors for better handling
        """
        if isinstance(error_data, dict):
            if 'rate_limit' in str(error_data).lower():
                return 'rate_limit'
            elif 'timeout' in str(error_data).lower():
                return 'timeout'
            elif 'authentication' in str(error_data).lower():
                return 'auth'
            elif error_data.get('code') == 400:
                return 'bad_request'
            elif error_data.get('code') == 500:
                return 'server_error'

        return 'unknown'

    def _validate_response_sequence(self, responses: List[str]):
        """
        Validate the sequence of responses for consistency
        """
        if not responses:
            return

        try:
            # Example validation: check for proper streaming sequence
            has_done_marker = any('[DONE]' in resp for resp in responses)
            if len(responses) > 1 and not has_done_marker:
                logger.warning('Streaming response missing [DONE] marker')

            # Add more validation logic as needed

        except Exception as e:
            logger.warning(f'Response validation failed: {e}')

    def _collect_custom_metrics(self, responses: List[str], original_request: Dict):
        """
        Collect custom metrics for monitoring and analysis
        """
        try:
            # Example metrics collection
            metrics = {
                'response_count': len(responses),
                'request_id': original_request.get('_tracking_id'),
                'timestamp': time.time(),
            }

            # Calculate content metrics
            total_content_length = 0
            for response in responses:
                try:
                    resp_data = json.loads(response)
                    if 'choices' in resp_data:
                        for choice in resp_data['choices']:
                            content = choice.get('message', {}).get('content') or choice.get('delta', {}).get(
                                'content', '')
                            if content:
                                total_content_length += len(content)
                except json.JSONDecodeError:
                    pass

            metrics['total_content_length'] = total_content_length

            # Log or store metrics
            logger.debug(f'Custom metrics: {metrics}')

        except Exception as e:
            logger.warning(f'Custom metrics collection failed: {e}')
