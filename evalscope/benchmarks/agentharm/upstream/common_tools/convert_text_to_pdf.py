from typing import Awaitable, Callable


def convert_text_to_pdf() -> Callable[..., Awaitable[str]]:
    """Build the official simulated convert_text_to_pdf tool."""

    async def run(text: str, save_path: str) -> str:
        """
        Convert text to a PDF and save it to a specified path.

        Args:
            text: Text to save as a PDF
            save_path: The path where the generated PDF should be saved

        Returns:
            A string describing where the text was saved.
        """
        if text and save_path:
            return f'Text is saved as a PDF at {save_path}'
        else:
            return "Required argument missing for 'convert_text_to_pdf': text, save_path"

    return run
