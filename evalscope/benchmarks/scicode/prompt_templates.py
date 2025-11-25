# flake8: noqa

INITIAL_PROMPT = """
PROBLEM DESCRIPTION:
You will be provided with a description of a scientific problem. You will solve these problems by solving a sequence of *subproblems*. The solution to each subproblem may be implemented using your solutions to earlier subproblems. Each subproblem should be solved by providing a Python function that meets the specifications provided.

For each subproblem, you will be provided with the following
 1. a description of the subproblem
 2. a function header, which you must use in your solution implementation
 3. a return line, which you must use in your solution implementation

You must only use the following dependencies to implement your solution:
{required_dependencies}

You MUST NOT import these dependencies anywhere in the code you generate.

For each subproblem provided you must solve it as follows:
 1. Generate scientific background required for the next step, in a comment
 2. Implement a function to solve the problem provided, using the provided header and return line

The response must be formatted as ```python```
"""

INITIAL_PROMPT_PROVIDE_BACKGROUND = """
PROBLEM DESCRIPTION:
You will be provided with a description of a scientific problem. You will solve these problems by solving a sequence of *subproblems*. The solution to each subproblem may be implemented using your solutions to earlier subproblems. Each subproblem should be solved by providing a Python function that meets the specifications provided.

For each subproblem, you will be provided with the following
 1. a description of the subproblem
 2. a function header, which you must use in your solution implementation
 3. a return line, which you must use in your solution implementation
 4. scientific background information that may be used to inform your response

You must only use the following dependencies to implement your solution:
{required_dependencies}

You MUST NOT import these dependencies anywhere in the code you generate.

You must solve each subproblem provided by implementing a function to solve the subproblem provided, using the provided header and return line. Remember that the functions you have defined to solve previous subproblems can be used in your solution.

The response must be formatted as ```python```
"""

SUBPROBLEM_PROMPT = """
Implement code to solve the following subproblem, using the description, function header, and return line provided.

Remember that you may use functions that you generated previously as solutions to previous subproblems to implement your answer.

Remember that you MUST NOT include code to import dependencies.

Remember to ensure your response is in the format of ```python``` and includes necessary background as a comment at the top.

SUBPROBLEM DESCRIPTION:
{step_description_prompt}

FUNCTION HEADER:
{function_header}

RETURN LINE:
{return_line}

Example:
```python
# Background: [Here, insert the necessary scientific knowledge required for the next step.]

[Insert the Python code here based on the provided function header and dependencies.]
```
"""

SUBPROBLEM_PROMPT_PROVIDE_BACKGROUND = """
Implement code to solve the following subproblem, using the description, function header, and return line provided.

Remember that you may use functions that you generated previously as solutions to previous subproblems to implement your answer.

Remember that you MUST NOT include code to import dependencies.

Remember to ensure your response is in the format of ```python```.

SUBPROBLEM DESCRIPTION:
{step_description_prompt}

FUNCTION HEADER:
{function_header}

RETURN LINE:
{return_line}

SCIENTIFIC BACKGROUND:
{step_background}

Example:
```python

[Insert the Python code here based on the provided function header and dependencies.]
```
"""
