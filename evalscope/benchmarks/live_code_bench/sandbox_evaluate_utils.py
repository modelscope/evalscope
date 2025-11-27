import json
from typing import TYPE_CHECKING, Dict, List, Tuple

from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.api.mixin.sandbox_mixin import SandboxMixin

logger = get_logger()


def evaluate_in_sandbox(
    adapter: 'SandboxMixin',
    code: str,
    evaluation_sample: str,
    timeout: int = 6,
    debug: bool = False
) -> Tuple[bool, Dict]:
    """
    Evaluate code in sandbox environment for Live Code Bench.

    Args:
        adapter: The adapter instance with sandbox capabilities
        code: The code to evaluate
        evaluation_sample: JSON string containing input/output test cases
        timeout: Timeout for execution
        debug: Whether to enable debug logging

    Returns:
        Tuple[bool, Dict]: (overall_pass, detailed_results)
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
        return False, {'error': str(e), 'total_tests': 0, 'passed_tests': 0}


def _evaluate_call_based_in_sandbox(
    adapter: 'SandboxMixin', code: str, inputs: list, outputs: list, fn_name: str, timeout: int, debug: bool
) -> Tuple[bool, Dict]:
    """Evaluate call-based problems in sandbox."""
    try:
        all_passed = True
        passed_count = 0
        failed_cases = []

        for i, (test_input, expected_output) in enumerate(zip(inputs, outputs)):
            # Prepare individual test code for each test case
            test_code = f"""
import json
import sys
from typing import TYPE_CHECKING, Dict, List, Tuple, Set, Sequence, Mapping
import ast

#Convert multi-type string to list with original data type
def parse_mixed_data(data_string):
    lines = data_string.strip().split('\\n')
    result = []

    for line in lines:
        if line.strip():  # skip empty line
            try:
                parsed_value = ast.literal_eval(line.strip())
                result.append(parsed_value)
            except (ValueError, SyntaxError):
                result.append(line.strip()) # Keep as string if parse failed

    return result

# User's code
{code}

# Test execution for single test case
try:
    test_input = {repr(test_input)}
    expected_output = {repr(expected_output)}

    if 'class Solution' in '''{code}''':
        # LeetCode style
        solution = Solution()
        method = getattr(solution, '{fn_name}')
    else:
        # Function is directly available
        method = {fn_name}

    # Parse input if it's JSON string
    parse_multi_type = False
    if isinstance(test_input, str):
        try:
            if test_input.find("\\n") > -1:
                test_input = parse_mixed_data(test_input)
                parse_multi_type = True
            else:
                test_input = json.loads(test_input)
        except:
            pass  # Keep as string if not valid JSON

    # Call the method
    if parse_multi_type:
        result = method(*test_input)
    else:
        result = method(test_input)

    # Parse expected output if it's JSON string
    if isinstance(expected_output, str):
        try:
            expected_output = json.loads(expected_output)
        except:
            pass  # Keep as string if not valid JSON

    # Convert tuple to list for comparison
    if isinstance(result, tuple):
        result = list(result)

    if result == expected_output:
        print("TEST_PASSED")
    else:
        print(f"TEST_FAILED: expected {{expected_output}}, got {{result}}")

except Exception as e:
    print(f"EXECUTION_ERROR: {{str(e)}}")
    import traceback
    traceback.print_exc()
"""

            # Execute in sandbox
            result = adapter.execute_code_in_sandbox(code=test_code, timeout=timeout, language='python')

            if debug:
                logger.info(f'Test case {i} execution result: {result}')

            # Check if execution was successful and test passed
            if result.get('status') == 'success':
                output = result.get('output', '')
                if 'TEST_PASSED' in output:
                    passed_count += 1
                elif 'TEST_FAILED:' in output:
                    # Extract failure details from output
                    for line in output.split('\n'):
                        if line.startswith('TEST_FAILED:'):
                            failed_cases.append(f"Test {i}: {line.replace('TEST_FAILED: ', '')}")
                            break
                    all_passed = False
                    break
                elif 'EXECUTION_ERROR:' in output:
                    # Extract error details
                    for line in output.split('\n'):
                        if line.startswith('EXECUTION_ERROR:'):
                            failed_cases.append(f'Test {i}: {line}')
                            break
                    all_passed = False
                    break
                else:
                    failed_cases.append(f'Test {i}: Unknown error in output. Result: {result}')
                    all_passed = False
                    break
            else:
                failed_cases.append(f'Test {i}: Sandbox execution failed - Result: {result}')
                all_passed = False
                break

        detailed_results = {'total_tests': len(inputs), 'passed_tests': passed_count, 'failed_cases': failed_cases}

        return all_passed, detailed_results

    except Exception as e:
        if debug:
            logger.error(f'Call-based evaluation error: {str(e)}')
        return False, {'error': str(e), 'total_tests': len(inputs), 'passed_tests': 0}


def _evaluate_stdio_in_sandbox(
    adapter: 'SandboxMixin', code: str, inputs: list, outputs: list, timeout: int, debug: bool
) -> Tuple[bool, Dict]:
    """Evaluate stdio-based problems in sandbox."""
    try:
        all_passed = True
        passed_count = 0
        failed_cases = []

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
                failed_cases.append(f'Test {i}: Execution error - Result: {result}')
                all_passed = False
                break

            # Compare output
            actual_output = result.get('output', '').strip()
            expected_output = expected_output.strip()

            if actual_output == expected_output:
                passed_count += 1
            else:
                if debug:
                    logger.info(f"Test case {i} failed: expected '{expected_output}', got '{actual_output}'")
                failed_cases.append(f"Test {i}: Expected '{expected_output}', got '{actual_output}'")
                all_passed = False
                break

        detailed_results = {'total_tests': len(inputs), 'passed_tests': passed_count, 'failed_cases': failed_cases}

        return all_passed, detailed_results

    except Exception as e:
        if debug:
            logger.error(f'Stdio evaluation error: {str(e)}')
        return False, {'error': str(e), 'total_tests': len(inputs), 'passed_tests': 0}
