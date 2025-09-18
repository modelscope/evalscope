import json
from typing import TYPE_CHECKING

from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.api.mixin.sandbox_mixin import SandboxMixin

logger = get_logger()


def evaluate_in_sandbox(
    adapter: 'SandboxMixin', code: str, evaluation_sample: str, timeout: int = 6, debug: bool = False
) -> bool:
    """
    Evaluate code in sandbox environment for Live Code Bench.

    Args:
        adapter: The adapter instance with sandbox capabilities
        code: The code to evaluate
        evaluation_sample: JSON string containing input/output test cases
        timeout: Timeout for execution
        debug: Whether to enable debug logging

    Returns:
        bool: True if all test cases pass, False otherwise
    """
    try:
        # Parse the evaluation sample
        test_data = json.loads(evaluation_sample)
        inputs = test_data.get('inputs', [])
        outputs = test_data.get('outputs', [])
        fn_name = test_data.get('fn_name')

        if debug:
            logger.info(f'Evaluating code with {len(inputs)} test cases')
            logger.info(f'Function name: {fn_name}')

        # Determine if this is call-based or stdio-based
        if fn_name:
            # Call-based evaluation
            return _evaluate_call_based_in_sandbox(adapter, code, inputs, outputs, fn_name, timeout, debug)
        else:
            # Standard input/output evaluation
            return _evaluate_stdio_in_sandbox(adapter, code, inputs, outputs, timeout, debug)

    except Exception as e:
        if debug:
            logger.error(f'Sandbox evaluation error: {str(e)}')
        return False


def _evaluate_call_based_in_sandbox(
    adapter: 'SandboxMixin', code: str, inputs: list, outputs: list, fn_name: str, timeout: int, debug: bool
) -> bool:
    """Evaluate call-based problems in sandbox."""
    try:
        # Prepare the test code
        test_code = f"""
import json
import sys

# User's code
{code}

# Test execution
try:
    inputs = {inputs}
    expected_outputs = {outputs}

    if 'class Solution' in '''{code}''':
        # LeetCode style
        solution = Solution()
        method = getattr(solution, '{fn_name}')
    else:
        # Function is directly available
        method = {fn_name}

    all_passed = True
    for i, (test_input, expected_output) in enumerate(zip(inputs, expected_outputs)):
        # Parse input if it's JSON string
        if isinstance(test_input, str):
            test_input = json.loads(test_input)
        if isinstance(test_input, list):
            result = method(*test_input)
        else:
            result = method(test_input)

        # Parse expected output if it's JSON string
        if isinstance(expected_output, str):
            expected_output = json.loads(expected_output)

        # Convert tuple to list for comparison
        if isinstance(result, tuple):
            result = list(result)

        if result != expected_output:
            print(f"Test case {{i}} failed: expected {{expected_output}}, got {{result}}")
            all_passed = False
            break

    if all_passed:
        print("ALL_TESTS_PASSED")
    else:
        print("SOME_TESTS_FAILED")

except Exception as e:
    print(f"EXECUTION_ERROR: {{str(e)}}")
    import traceback
    traceback.print_exc()
"""

        # Execute in sandbox
        result = adapter.execute_code_in_sandbox(code=test_code, timeout=timeout, language='python')

        if debug:
            logger.info(f'Sandbox execution result: {result}')

        # Check if execution was successful and tests passed
        if result.get('status') == 'success':
            output = result.get('output', '')
            return 'ALL_TESTS_PASSED' in output
        else:
            if debug:
                logger.error(f'Sandbox execution failed: {result}')
            return False

    except Exception as e:
        if debug:
            logger.error(f'Call-based evaluation error: {str(e)}')
        return False


def _evaluate_stdio_in_sandbox(
    adapter: 'SandboxMixin', code: str, inputs: list, outputs: list, timeout: int, debug: bool
) -> bool:
    """Evaluate stdio-based problems in sandbox."""
    try:
        # For stdio problems, we need to test each input/output pair
        all_passed = True

        for i, (test_input, expected_output) in enumerate(zip(inputs, outputs)):
            test_code = f"""
import sys
from io import StringIO

# Redirect stdin
sys.stdin = StringIO('''{test_input}''')

# User's code
{code}
"""

            # Execute in sandbox
            result = adapter.execute_code_in_sandbox(code=test_code, timeout=timeout, language='python')

            if result.get('status') != 'success':
                if debug:
                    logger.error(f'Test case {i} execution failed: {result}')
                all_passed = False
                break

            # Compare output
            actual_output = result.get('output', '').strip()
            expected_output = expected_output.strip()

            if actual_output != expected_output:
                if debug:
                    logger.info(f"Test case {i} failed: expected '{expected_output}', got '{actual_output}'")
                all_passed = False
                break

        return all_passed

    except Exception as e:
        if debug:
            logger.error(f'Stdio evaluation error: {str(e)}')
        return False
