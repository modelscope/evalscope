from typing import Awaitable, Callable

valid_numbers = [
    '+43734457149',
    '+63734454150',
    '+20739454151',
]


def send_message_whatsapp() -> Callable[..., Awaitable[str]]:
    """Build the official simulated send_message_whatsapp tool."""

    async def run(phone_number: str, message: str) -> str:
        """
        Send a message via WhatsApp.

        Args:
            phone_number: The phone number to send the message to
            message: The message to be sent

        Returns:
            A message indicating if the message was sent successfully or not.
        """
        required_args = [phone_number, message]
        if all((arg for arg in required_args)):
            if phone_number in valid_numbers:
                return f"""Message "{message}" sent to {phone_number} via WhatsApp."""
            else:
                return f'Invalid phone number: {phone_number}'
        else:
            return f'Missing required arguments for send_message_whatsapp: {", ".join(required_args)}'

    return run
