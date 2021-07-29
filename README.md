# SeqScore

SeqScore: Scoring for named entity recognition and other sequence labeling tasks


# Installation

Normal installation: `pip install seqscore`

If you need to install from a copy of this repository, use: `pip install
-e .`


# Usage

## Overview

For a list of commands, run `seqscore --help`.

Some examples:
```
# Score like conlleval
seqscore score --labels BIO --repair-method conlleval --reference <reference_conll_file> <prediction_conll_file>
# Score discarding invalid chunks, which sometimes produces higher scores
seqscore score --labels BIO --repair-method discard --reference <reference_conll_file> <prediction_conll_file>
seqscore validate --labels BIO <input_conll_file>
seqscore dump --labels BIO <input_conll_file> <output_delim_file>
```

Scoring only supports BIO chunk encoding. Validation can be done for IO, BIO, and BIOES.
At the moment, `dump` only supports BIO, but support will be added for IO and BIOES.


# Features coming soon!

* More documentation
* More error analysis tools


# Setting up for development

1. Create environment: `conda create -y -n seqscore python=3.8`
2. Activate the environment: `conda activate seqscore`
3. Install dependencies: `pip install -r requirements.txt`
4. Install seqscore: `pip install -e .`
