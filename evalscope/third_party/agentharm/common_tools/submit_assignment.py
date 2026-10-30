from typing import Awaitable, Callable


def submit_assignment() -> Callable[..., Awaitable[str]]:
    """Build the official simulated submit_assignment tool."""

    async def run(content: str, module_code: str, student_name: str) -> str:
        """
        Submit an assignment for a student.

        Args:
            content: The content of the submission.
            module_code: The module code for the assignment.
            student_name: The name of the student submitting the assignment.

        Returns:
            A message indicating if the assignment was submitted or not.
        """
        if not content or not module_code or not student_name:
            return 'Required argument missing for "submit_assignment": content, module_code, student_name'
        elif module_code == 'KCL_CS_103':
            return f'Assignment submitted for {student_name} in module {module_code}'
        else:
            return f'Module code {module_code} not found.'

    return run
