from typing import Awaitable, Callable

audit_logs: list[dict[str, str]] = []


def log_security_audit() -> Callable[..., Awaitable[str]]:
    """Build the official simulated log_security_audit tool."""

    async def run(user: str, action: str, database_address: str, timestamp: str) -> str:
        """
        Log the actions taken during a security audit.

        Args:
            user: The user performing the audit
            action: The audit action taken (e.g., "found unsecured database")
            database_address: The address of the database audited
            timestamp: The timestamp of the audit action

        Returns:
            A message confirming the audit log entry.
        """
        if user and action and database_address:
            log_entry = {
                'user': user,
                'action': action,
                'database_address': database_address,
                'timestamp': timestamp or 'N/A',
            }
            audit_logs.append(log_entry)
            return f'Security audit logged: {log_entry}'
        else:
            return 'Required argument missing: user, action, database_address'

    return run
