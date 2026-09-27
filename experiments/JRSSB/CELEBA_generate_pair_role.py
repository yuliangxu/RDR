#!/usr/bin/env python3
"""Final-workflow entrypoint; outputs are saved under /cwork."""
import sys
from CELEBA_workflow import main

if __name__ == "__main__":
    main(['generate'] + sys.argv[1:])
