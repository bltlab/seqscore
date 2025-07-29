#! /usr/bin/env python
"""Import all top-level modules in seqscore.

This imports all the top-level modules as a smoke test for making sure all needed
dependencies are installed. We use this as basic test before pytest is installed
to make sure that there are no dependencies that we are accidentally relying on
pytest to install for us.
"""

import seqscore.scripts.seqscore  # noqa: F401
from seqscore import conll, encoding, model, scoring, util, validation  # noqa: F401

print(f"{__file__}:", "Successfully imported all top-level modules")
