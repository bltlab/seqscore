# SeqScore

![Build Status](https://github.com/bltlab/seqscore/actions/workflows/main.yml/badge.svg)

SeqScore provides scoring for named entity recognition and other
chunking tasks evaluated over sequence labels.

SeqScore is maintained by the BLT Lab at Brandeis University. Please
open an issue if you find incorrect behavior or features you would like
to see added. Due to the risk of introducing regressions or incorrect
scoring behavior, *we generally do not accept pull requests*. Please do not
open a pull request unless you are asked to do so by a maintainer in an
issue.

## Installation

To install the latest official release of SeqScore, run: `pip install seqscore`.
This will install the package and add the command `seqscore` in your Python
environment.

SeqScore requires Python 3.7 or higher. It is tested on Python 3.7, 3.8, 3.9,
3.10, and 3.11.

## License

SeqScore is distributed under the MIT License.

## Citation

If you use SeqScore, please cite
[SeqScore: Addressing Barriers to Reproducible Named Entity Recognition Evaluation](https://aclanthology.org/2021.eval4nlp-1.5/).

BibTeX:

```
@inproceedings{palen-michel-etal-2021-seqscore,
    title = "{S}eq{S}core: Addressing Barriers to Reproducible Named Entity Recognition Evaluation",
    author = "Palen-Michel, Chester  and
      Holley, Nolan  and
      Lignos, Constantine",
    booktitle = "Proceedings of the 2nd Workshop on Evaluation and Comparison of NLP Systems",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.eval4nlp-1.5",
    pages = "40--50",
}
```

# Usage

## Overview

For a list of commands, run `seqscore --help`:

```
$ seqscore --help
Usage: seqscore [OPTIONS] COMMAND [ARGS]...

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  convert
  count
  repair
  score
  summarize
  validate
```

## Scoring

The most common application of SeqScore is scoring CoNLL-format NER
predictions. Let's assume you have two files, one containing the
correct labels (annotation) and the other containing the predictions
(system output).

The correct labels are in the file [samples/reference.bio](samples/reference.bio):

```
This O
is O
a O
sentence O
. O

University B-ORG
of I-ORG
Pennsylvania I-ORG
is O
in O
West B-LOC
Philadelphia I-LOC
, O
Pennsylvania B-LOC
. O

```

The predictions are in the file [samples/predicted.bio](samples/predicted.bio):

```
This O
is O
a O
sentence O
. O

University B-ORG
of I-ORG
Pennsylvania I-ORG
is O
in O
West B-LOC
Philadelphia B-LOC
, O
Pennsylvania B-LOC
. O

```

To score the predictions, run:
`seqscore score --labels BIO --reference samples/reference.bio samples/predicted.bio`

```
| Type   |   Precision |   Recall |     F1 |   Reference |   Predicted |   Correct |
|--------|-------------|----------|--------|-------------|-------------|-----------|
| ALL    |       50.00 |    66.67 |  57.14 |           3 |           4 |         2 |
| LOC    |       33.33 |    50.00 |  40.00 |           2 |           3 |         1 |
| ORG    |      100.00 |   100.00 | 100.00 |           1 |           1 |         1 |
```

A few things to note:

* The reference file must be specified with the `--reference` flag.
* The chunk encoding (BIO, BIOES, etc.) must be specified using the
  `--labels` flag.
* Both files need to use the same chunk encoding. If you have
  files that use different chunk encodings, use the `convert` command.
* You can get output in different formats using the `--score-format`
  flag. Using `--score-format delim` will produce tab-delimited
  output. In the delimited format, you can specify the `--full-precision`
  flag to output higher numerical precision.
* In the default (pretty) output format, numbers are rounded "half up"
  at two decimal places. In other words, 57.124 will round to 57.12,
  and 57.125 will round to 57.13. This is different than the "half even"
  rounding used by `conlleval` and other libraries that rely on `printf`
  behavior for rounding. Half up rounding is used as it is more likely to
  match the rounding a user would perform if shown three decimal places.
  If you request `conlleval` output format, the same rounding used by
  `conlleval` will be used.

The above scoring command will work for files that do not have any
invalid transitions, that is, those that perfectly follow what the
encoding allows. However, consider this BIO-encoded file,
[samples/invalid.bio](samples/invalid.bio):

```
This O
is O
a O
sentence O
. O

University I-ORG
of I-ORG
Pennsylvania I-ORG
is O
in O
West B-LOC
Philadelphia I-LOC
, O
Pennsylvania B-LOC
. O

```

Note that the token `University` has the label `I-ORG`, but there is
no preceding `B-ORG`. If we score it as before with
`seqscore score --labels BIO --reference samples/reference.bio samples/invalid.bio`,
scoring will fail:

```
seqscore.encoding.EncodingError: Stopping due to validation errors in invalid.bio:
Invalid transition 'O' -> 'I-ORG' for token 'University' on line 7
```

To score output with invalid transitions, we need to specify a repair
method which can correct them. We can tell SeqScore to use the same
approach that conlleval uses (which we refer to as "begin" repair in our
paper):
`seqscore score --labels BIO --repair-method conlleval --reference samples/reference.bio samples/invalid.bio`:

```
Validation errors in sequence at line 7 of invalid.bio:
Invalid transition 'O' -> 'I-ORG' for token 'University' on line 7
Used method conlleval to repair:
Old: ('I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'B-LOC', 'O')
New: ('B-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'B-LOC', 'O')
| Type   |   Precision |   Recall |     F1 |   Reference |   Predicted |   Correct |
|--------|-------------|----------|--------|-------------|-------------|-----------|
| ALL    |      100.00 |   100.00 | 100.00 |           3 |           3 |         3 |
| LOC    |      100.00 |   100.00 | 100.00 |           2 |           2 |         2 |
| ORG    |      100.00 |   100.00 | 100.00 |           1 |           1 |         1 |
```

You can use the `-q` flag to suppress the logging of all of the repairs
applied. For example, running the command
`seqscore score -q --labels BIO --repair-method conlleval --reference samples/reference.bio samples/invalid.bio`
will hide the repairs:

```
| Type   |   Precision |   Recall |     F1 |   Reference |   Predicted |   Correct |
|--------|-------------|----------|--------|-------------|-------------|-----------|
| ALL    |      100.00 |   100.00 | 100.00 |           3 |           3 |         3 |
| LOC    |      100.00 |   100.00 | 100.00 |           2 |           2 |         2 |
| ORG    |      100.00 |   100.00 | 100.00 |           1 |           1 |         1 |
```

You may want to also explore the `discard` repair, which can
produce higher scores for output from models without a CRF/constrained
decoding as they are more likely to produce invalid transitions.

SeqScore can also display all errors (false positives and false negatives)
encountered in scoring using the `--error-counts` flag. For example, running the
command
`seqscore score --labels BIO --error-counts --reference samples/reference.bio samples/predicted.bio`
will produce the following output:

```
|   Count | Error   | Type   | Tokens            |
|---------|---------|--------|-------------------|
|       1 | FP      | LOC    | Philadelphia      |
|       1 | FP      | LOC    | West              |
|       1 | FN      | LOC    | West Philadelphia |
```

The output shows that the system produced two false positives and missed one
mention in the reference (false negative). The most frequent errors appear at
the top. The `--error-counts` flag can be combined with `--score-format delim`
to write a delimited table that can be read as a spreadsheet.

## Validation

To check if a file has any invalid transitions, we can run
`seqscore validate --labels BIO samples/reference.bio`:

```
No errors found in 0 tokens, 2 sequences, and 1 documents in reference.bio
```

For the example of the [samples/invalid.bio](samples/invalid.bio), we can run
`seqscore validate --labels BIO samples/invalid.bio`:

 ```
Encountered 1 errors in 1 tokens, 2 sequences, and 1 documents in invalid.bio
Invalid transition 'O' -> 'I-ORG' for token 'University' on line 7
```

## Convert

We can convert a file from one chunk encoding to another. For example,
`seqscore convert --input-labels BIO --output-labels BIOES samples/reference.bio samples/reference.bioes`
will read [samples/reference.bio](samples/reference.bio) in BIO
encoding and write the BIOES-converted file to [samples/reference.bioes](samples/reference.bioes):

```
This O
is O
a O
sentence O
. O

University B-ORG
of I-ORG
Pennsylvania E-ORG
is O
in O
West B-LOC
Philadelphia E-LOC
, O
Pennsylvania S-LOC
. O

```

We can get a list of available chunk encodings by running `seqscore convert --help`:

```
Usage: seqscore convert [OPTIONS] FILE OUTPUT_FILE

Options:
  --file-encoding TEXT            [default: UTF-8]
  --ignore-comment-lines
  --ignore-document-boundaries / --use-document-boundaries
  --output-delim TEXT             [default: space]
  --input-labels [BIO|BIOES|BILOU|BMES|BMEOW|IO|IOB]
                                  [required]
  --output-labels [BIO|BIOES|BILOU|BMES|BMEOW|IO|IOB]
                                  [required]
  --help                          Show this message and exit.
```

## Repair

We can also apply repair methods to a file, creating an output file
with only valid transitions. For example, we can run
`seqscore repair --labels BIO --repair-method conlleval samples/invalid.bio samples/invalid_repair_conlleval.bio`,
which will apply the conlleval repair method to the
[samples/invalid.bio](samples/invalid.bio) and write the repaired
labels to
[samples/invalid_repair_conlleval.bio](samples/invalid_repair_conlleval.bio):

```
This O
is O
a O
sentence O
. O

University B-ORG
of I-ORG
Pennsylvania I-ORG
is O
in O
West B-LOC
Philadelphia I-LOC
, O
Pennsylvania B-LOC
. O

```

If we want to apply the discard repair method, we can run
`seqscore repair --labels BIO --repair-method discard samples/invalid.bio samples/invalid_repair_discard.bio`
and the output will be written to [samples/invalid_repair_discard.bio](samples/invalid_repair_discard.bio):

```
This O
is O
a O
sentence O
. O

University O
of O
Pennsylvania O
is O
in O
West B-LOC
Philadelphia I-LOC
, O
Pennsylvania B-LOC
. O

```

Repairing the file before performing other operations is available in the
`count` and `summarize` subcommands.

## Summarize

The `summarize` subcommand can produce counts of the types of chunks
in the input file. For example, if we run
`seqscore summarize --labels BIO samples/reference.bio`
we get the following output:

```
File 'samples/reference.bio' contains 1 document(s) with the following mentions:
| Entity Type   |   Count |
|---------------|---------|
| LOC           |       2 |
| ORG           |       1 |
```

If the quiet (`-q`) flag is provided, the first line giving the filename
and document count is not printed.

## Count

The `count` subcommand can produce the counts of chunks in the input
file. Unlike `summarize`, it counts chunk-type pairs, not just types.
For example, if we run
`seqscore count --labels BIO samples/reference.bio counts.csv`,
tab-delimited counts would be written to `counts.csv` as follows:

```
1	ORG	University of Pennsylvania
1	LOC	West Philadelphia
1	LOC	Pennsylvania
```

## Process

The `process` subcommand can remove entity types from a file or map them to
other types. Removing types can be performed by specifying one of `--keep-types`
or `--remove-types`.

For example, if we only wanted to keep the ORG type, we could run:
`seqscore process --labels BIO --keep-types ORG samples/reference.bio samples/keep_ORG.bio`,
and the following output will be written to [samples/keep_ORG.bio](samples/keep_ORG.bio):

```
This O
is O
a O
sentence O
. O

University B-ORG
of I-ORG
Pennsylvania I-ORG
is O
in O
West O
Philadelphia O
, O
Pennsylvania O
. O
```

You can also keep multiple types by specifying a comma-separated list of types:
`--keep-types LOC,ORG`.

Instead of specifying which types to keep, we can also specify which types to
remove using `--remove-types`. For example, if we wanted to remove only the
ORG type, we could run:
`seqscore process --labels BIO --remove-types ORG samples/reference.bio samples/remove_ORG.bio`,
and the following output will be written to [samples/remove_ORG.bio](samples/remove_ORG.bio):

```
This O
is O
a O
sentence O
. O

University O
of O
Pennsylvania O
is O
in O
West B-LOC
Philadelphia I-LOC
, O
Pennsylvania B-LOC
. O
```

As with keep, you can specify multiple tags to remove, for example
`--remove-types LOC,ORG`.

The `--type-map` argument allows you to specify a JSON file that specifies a
mapping between types and other types. Suppose you want to collapse several
types into a more generic NAME type. In that case, the type map would be
specified as follows:

```
{
  "NAME": ["LOC", "ORG"]
}
```

The type map must be a JSON dictionary. The keys are the types to be mapped to,
while the value for each key is a list of types to be mapped from. Note that
the value must always be a list, even if it would only contain one element.

We can apply the above type map to a file using the following command:
`seqscore process --labels BIO --type-map samples/type_map_NAME.json samples/reference.bio samples/all_NAME.bio`,
resulting in this output:

```
This O
is O
a O
sentence O
. O

University B-NAME
of I-NAME
Pennsylvania I-NAME
is O
in O
West B-NAME
Philadelphia I-NAME
, O
Pennsylvania B-NAME
. O
```

# FAQ

## Why can't I score output files that are in the format `conlleval` expects?

At this time, SeqScore intentionally does not support the "merged"
format used by `conlleval` where each line contains a token, correct
tag, and predicted tag:

```
University B-ORG B-ORG
of I-ORG I-ORG
Pennsylvania I-ORG I-ORG
is O O
in O O
West B-LOC B-LOC
Philadelphia I-LOC B-LOC
, O O
Pennsylvania B-LOC B-LOC
. O O
```

We do not support this format because we have found that creating
predictions in this format is a common source of errors in scoring
pipelines.

# Development

The following instructions are for the project maintainers only.

For development, check out the `dev` branch (latest, but less tested
than `main`).

To install from a clone of this repository, use:
`pip install -e .`

## Setting up an environment for development

1. Create an environment: `conda create -y -n seqscore python=3.7`
2. Activate the environment: `conda activate seqscore`
3. Install seqscore: `pip install -e .`
4. Install development dependencies: `pip install -r requirements.txt`

# Contributors

SeqScore was developed by the BLT Lab at Brandeis University under the
direction of PI and and lead developer Constantine Lignos. Chester Palen-Michel
and Nolan Holley contributed to its development. Gordon Dou, Maya Kruse, and
Andrew Rueda gave feedback on its features and assisted in README writing.
