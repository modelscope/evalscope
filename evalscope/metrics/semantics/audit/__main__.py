"""Allow ``python -m evalscope.metrics.semantics.audit`` to run the audit."""

import sys

from .cli import main

if __name__ == '__main__':
    sys.exit(main())
