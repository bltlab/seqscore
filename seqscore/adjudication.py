import os
import re
from collections import Counter, defaultdict
from math import log
from typing import (
    AbstractSet,
    DefaultDict,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from attr import attrib, attrs

from seqscore.model import LabeledSequence, Mention, SequenceProvenance, Span

ENTITY = "entity"
NON_ENTITY = "non_entity"


@attrs(frozen=True, slots=True, auto_attribs=True)
class AnnotatorMention:
    mention: Mention
    annotator: int


@attrs(frozen=True, slots=True)
class AnnotatorLabeledSequence:
    tokens: Tuple[str, ...] = attrib()
    mentions_by_span: Dict[Span, List[AnnotatorMention]] = attrib(factory=dict)
    labels_by_annotator: Dict[int, Tuple[str, ...]] = attrib(factory=dict)
    provenance: Optional[SequenceProvenance] = attrib(
        default=None, eq=False, kw_only=True
    )

    def __len__(self):
        return len(self.tokens)

    def span_tokens(self, span: Span) -> Tuple[str, ...]:
        return self.tokens[span.start : span.end]


def sus_sentence_ending(labelsequence: LabeledSequence) -> bool:
    if len(labelsequence) < 4:
        return True
    elif re.fullmatch(r"[A-Z]\d+", labelsequence.tokens[-2]):
        # Don't mark N380 as a bad sentence candidate since it's currency
        return False
    # XXX: Are there other punctuations we want to account for? Ge'ez etc?
    elif (
        labelsequence.tokens[-2][0].isupper()
        and labelsequence.tokens[-1] == "."
        and len(labelsequence.tokens[-2]) < 5
    ):
        # check if second to last token has upper case first character and is followed by a . and is short
        # Checking for Dr. /Dkt .
        return True
    else:
        return False


@attrs
class AdjudicationSequence:
    labelsequence: LabeledSequence = attrib()
    conflict_labels_by_annotator: Dict[int, Tuple[str, ...]] = attrib()
    orig_labels_by_annotator: Dict[int, Tuple[str, ...]] = attrib()
    comment_labels: List[Optional[str]] = attrib()

    @classmethod
    def create_sequence(
        cls,
        label_sequence: AnnotatorLabeledSequence,
        mentions: Sequence[Mention],
        conflict_mentions: Sequence[AnnotatorMention],
        comment_labels: List[Optional[str]],
    ) -> "AdjudicationSequence":
        # if conflict_mentions:
        #     unsure_mentions_list = ", ".join(str(m) for m in conflict_mentions)
        #     conflict_string = f"# CONFLICT: {unsure_mentions_list}"
        # else:
        #     conflict_string = None
        conflict_mentions_by_annotator = defaultdict(list)
        for m in conflict_mentions:
            conflict_mentions_by_annotator[m.annotator].append(m.mention)
        conflict_labels_by_annotator = {
            annotator: new_labels_from_mentions(
                conflict_mentions_by_annotator[annotator], len(label_sequence)
            )
            for annotator in conflict_mentions_by_annotator
        }
        return AdjudicationSequence(
            LabeledSequence(
                label_sequence.tokens,
                new_labels_from_mentions(mentions, len(label_sequence)),
                tuple(mentions),
                provenance=label_sequence.provenance,
            ),
            conflict_labels_by_annotator,
            label_sequence.labels_by_annotator,
            comment_labels,
        )

    def output_lines(self, delim: str = "\t") -> Generator[str, None, None]:
        annotators = sorted(self.orig_labels_by_annotator)
        conflicting_indices = set(
            [
                idx
                for annotator in self.conflict_labels_by_annotator
                for idx, label in enumerate(self.conflict_labels_by_annotator[annotator])
                if label != "O"
            ]
        )
        sus_sent = sus_sentence_ending(self.labelsequence)
        prev_label = None
        for i, (token, label) in enumerate(self.labelsequence.tokens_with_labels()):
            orig_labels = [
                self.orig_labels_by_annotator[annotator][i] for annotator in annotators
            ]
            comment = self.comment_labels[i] if self.comment_labels[i] else ""
            if i in conflicting_indices:
                # TODO Leave more useful comment about what conflict is
                # TODO: use entropy stuff to leave hints
                # TODO: Add comment labels to the AdjudicationSequence,
                #  so can zip them in when they exist
                if prev_label != "ADJ":
                    conflict_prefix = f"conflict-todo:{comment}"
                else:
                    conflict_prefix = f"conflict:{comment}"
                prev_label = "ADJ"
                ret = [token, "ADJ"] + orig_labels + [conflict_prefix]
            else:
                prev_label = label
                ret = [token, label] + orig_labels + [f"auto-adjudicated:{comment}"]
            if sus_sent and i == len(self.labelsequence) - 1:
                ret[-1] = ret[-1] + "check-sentence-break"
            ret.extend(["", ""])
            yield delim.join(ret)


@attrs
class MentionCounts:
    _counts_dict: DefaultDict[Tuple[str, ...], Counter] = attrib(
        factory=lambda: defaultdict(Counter)
    )
    _total_mention_count: DefaultDict[Tuple[str, ...], int] = attrib(
        factory=lambda: defaultdict(int)
    )

    def put(self, mention: Tuple[str, ...], label: str):
        self._counts_dict[mention][label] += 1
        self._total_mention_count[mention] += 1

    def put_value(self, mention: Tuple[str, ...], label: str, value: int):
        self._counts_dict[mention][label] = value
        self._total_mention_count[mention] += value

    def mention_tokens(self) -> List[Tuple[str, ...]]:
        return list(self._counts_dict.keys())

    def mention_count(self, mention: Tuple[str, ...], mention_type: str):
        if mention_type in self._counts_dict[mention]:
            return self._counts_dict[mention][mention_type]
        else:
            raise ValueError(
                f"Mention type {mention_type} doesn't exist in this mention count"
            )

    def mention_types(self, mention: Tuple[str, ...]) -> List[str]:
        return list(self._counts_dict[mention].keys())

    def entropy(self, mention: Tuple[str, ...]) -> float:
        ps = [
            (
                self._counts_dict[mention][mention_type]
                / self._total_mention_count[mention]
            )
            for mention_type in self.mention_types(mention)
        ]
        return -1 * sum(p * log(p) for p in ps if p != 0)

    def most_common_types(
        self, mention: Tuple[str, ...], n: Optional[int] = None
    ) -> List[Tuple[str, int]]:
        return self._counts_dict[mention].most_common(n=n)

    def highest_entropy_mentions(
        self, n: Optional[int] = None
    ) -> List[Tuple[Tuple[str, ...], float]]:
        ret = sorted(
            [(mention, self.entropy(mention)) for mention in self.mention_tokens()],
            key=lambda x: x[1],
            reverse=True,
        )
        if n is not None:
            return ret[:n]
        return ret

    def lowest_entropy_mentions(
        self, n: Optional[int] = None
    ) -> List[Tuple[Tuple[str, ...], float]]:
        ret = sorted(
            [(mention, self.entropy(mention)) for mention in self.mention_tokens()],
            key=lambda x: x[1],
            reverse=False,
        )
        if n is not None:
            return ret[:n]
        return ret

    def mentions_to_flag(self, threshold: float = 0.3) -> Dict[Tuple[str, ...], str]:
        """Mentions with entropy under a certain threshold mapped to most common type"""
        mentions = [
            mention
            for mention, entropy in self.lowest_entropy_mentions()
            if threshold >= entropy > 0.0
        ]
        return {
            mention: self.most_common_types(mention, n=1)[0][0] for mention in mentions
        }


@attrs
class DisagreementResult:
    disagreements = attrib()
    disaggrements_by_type = attrib()
    disagreements_by_span = attrib()
    fleiss_kappa: float = attrib()
    majority_vote_sequences: Tuple[AdjudicationSequence, ...] = attrib()


@attrs(frozen=True)
class AdjudicationResult:
    overall_entity_counts: MentionCounts = attrib()
    entity_counts_by_annotator: Dict[int, MentionCounts] = attrib()
    overall_type_counts: MentionCounts = attrib()
    type_counts_by_annotator: Dict[int, MentionCounts] = attrib()
    disagreements: DisagreementResult = attrib()


def new_labels_from_mentions(
    mentions: Sequence[Mention], seq_len: int
) -> Tuple[str, ...]:
    # TODO: Handle other encoding schemes probably, ugh.
    seq = ["O" for _ in range(seq_len)]
    for mention in mentions:
        seq[mention.span.start] = f"B-{mention.type}"
        for i in range(mention.span.start + 1, mention.span.end):
            seq[i] = f"I-{mention.type}"
    return tuple(seq)


def ngrams(tokens: Sequence[str], n: int = 1) -> Generator[Tuple[str, ...], None, None]:
    """
    Gets ngrams for a sequence of tokens with specified n
    """
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def yield_all_ngrams(
    tokens: Sequence[Union[str, Tuple[str, str], Tuple[str, int]]],
    max_n: Optional[int] = None,
) -> Generator[Tuple[Union[str, Tuple[str, str], Tuple[str, int]]], None, None]:
    if max_n is None:
        max_n = len(tokens)
    max_n = min([len(tokens), max_n])

    for n in range(1, max_n):
        for ngram in ngrams(tokens, n):
            yield ngram


def count_mention_spans(
    mention_type_counts: MentionCounts, docs: List[List[LabeledSequence]]
) -> Dict[Tuple[str], int]:
    counts = Counter(
        {mention_tokens: 0 for mention_tokens in mention_type_counts.mention_tokens()}
    )
    max_ent_len = max(len(ent) for ent in mention_type_counts.mention_tokens())
    for doc in docs:
        for labelsequence in doc:
            for ngram in yield_all_ngrams(labelsequence.tokens, max_ent_len):
                if ngram in counts:
                    counts[ngram] += 1
    return counts


def has_outside(token_label_span: Sequence[Tuple[str, str]]) -> bool:
    """Given list of token, label tuples, do any have outside"""
    # TODO: Should this be made to handle other label schemes?
    return any([t[1] == "O" for t in token_label_span])


def fleiss_kappa(
    units_to_labels: Mapping[Tuple[int, Span], Counter],
    num_annotators: int,
    labels: AbstractSet[str],
) -> float:
    pi_const = 1 / (num_annotators * (num_annotators - 1))
    pj_const = 1 / (len(units_to_labels) * num_annotators)
    p_i = [
        pi_const
        * sum(
            units_to_labels[unit][label] ** 2 - units_to_labels[unit][label]
            for label in labels
        )
        for unit in units_to_labels
    ]
    p_j = {
        label: pj_const * sum(units_to_labels[unit][label] for unit in units_to_labels)
        for label in labels
    }
    p_e = sum(pj ** 2 for pj in p_j.values())
    p_bar = (1 / len(units_to_labels)) * sum(p_i)

    numerator = p_bar - p_e
    denom = 1 - p_e
    if denom:
        return numerator / float(denom)
    return 0.0


def different_types(labels: Iterable[str]) -> bool:
    type_set = set(labels)
    return not len(type_set) == 1


def overlap(s1: Span, s2: Span) -> bool:
    if s1.start >= s2.end:
        return False
    elif s1.end <= s2.start:
        return False
    else:
        return True


def overlap_any(span: Span, agreed_spans: Iterable[Span]) -> bool:
    for s in agreed_spans:
        if overlap(span, s):
            return True
    return False


def no_overlap(
    mentions: Sequence[AnnotatorMention], label_sequence: AnnotatorLabeledSequence
):
    agreeing_annotators = set(m.annotator for m in mentions)
    agreed_spans = set(m.mention.span for m in mentions)
    for span in label_sequence.mentions_by_span:
        if span not in agreed_spans:
            for mention in label_sequence.mentions_by_span[span]:
                if mention.annotator not in agreeing_annotators and overlap_any(
                    mention.mention.span, agreed_spans
                ):
                    return False
    return True


def add_comment(comment_labels: List[Optional[str]], span: Span, comment: str) -> None:
    # Updates the comment labels
    for i in range(span.start, span.end):
        if not comment_labels[i]:
            comment_labels[i] = comment


def token_level_type_clash(
    labels_by_annotator: Dict[int, Tuple[str, ...]], span: Span
) -> bool:

    span_labels = [
        [label.lstrip("B-").lstrip("I-") for label in labels[span.start : span.end]]
        for labels in labels_by_annotator.values()
    ]
    for labels in zip(*span_labels):
        return len(set([label for label in labels if label != "O"])) > 1
    return False


def check_names_to_flag(
    mentions_to_flag_by_name: Dict[Tuple[str, ...], str],
    label_seq: AnnotatorLabeledSequence,
) -> List[Span]:
    if not mentions_to_flag_by_name:
        return []
    max_n = max(len(m) for m in mentions_to_flag_by_name)
    spans = []
    for token_seq in yield_all_ngrams(
        [(t, i) for i, t in enumerate(label_seq.tokens)], max_n=max_n
    ):
        tokens = tuple([t[0] for t in token_seq])
        if (
            tokens in mentions_to_flag_by_name
            and mentions_to_flag_by_name[tokens] == ENTITY
        ):
            # This uses the token indices, so +1 to the end token index
            spans.append(Span(int(token_seq[0][1]), token_seq[-1][1] + 1))
    return spans


def check_disagreements(
    annotator_label_sequences: Sequence[AnnotatorLabeledSequence],
    num_annotators: int,
    name_mention_counts: MentionCounts,
    type_mention_counts: MentionCounts,
):
    mentions_to_flag_by_type = type_mention_counts.mentions_to_flag()
    mentions_to_flag_by_name = name_mention_counts.mentions_to_flag(threshold=0.2)
    disagreements = 0
    disaggrements_by_type = 0
    disagreements_by_span = 0
    units_to_labels = defaultdict(Counter)
    units_to_mentions = defaultdict(list)
    labels = set("O")
    majority_voted_mentions = defaultdict(list)
    conflict_mentions = defaultdict(list)
    label_sequences_by_line = {}
    for label_sequence in annotator_label_sequences:
        label_sequences_by_line[label_sequence.provenance.starting_line] = label_sequence
        for span in label_sequence.mentions_by_span:
            unit = (label_sequence.provenance.starting_line, span)
            for annotator_mention in label_sequence.mentions_by_span[span]:
                labels.add(annotator_mention.mention.type)
                units_to_labels[unit][annotator_mention.mention.type] += 1
                units_to_mentions[unit].append(annotator_mention)
    comment_labels_by_line = {
        label_sequence.provenance.starting_line: [
            None for _ in range(len(label_sequence))
        ]
        for label_sequence in annotator_label_sequences
    }
    for unit in units_to_labels:
        if sum(units_to_labels[unit].values()) > num_annotators / 2 and no_overlap(
            units_to_mentions[unit], label_sequences_by_line[unit[0]]
        ):
            # Add majority type if more than half annotators choose the span
            majority_voted_mentions[unit[0]].append(
                Mention(unit[1], units_to_labels[unit].most_common(1)[0][0])
            )
        else:
            conflict_mentions[unit[0]].extend(
                mention for mention in units_to_mentions[unit]
            )
        if num_annotators - sum(units_to_labels[unit].values()) == 0:
            if different_types(units_to_labels[unit]):
                disaggrements_by_type += 1
                disagreements += 1
                add_comment(comment_labels_by_line[unit[0]], unit[1], "type")
        else:
            disagreements += 1
            disagreements_by_span += 1
            if token_level_type_clash(
                label_sequences_by_line[unit[0]].labels_by_annotator, unit[1]
            ):
                add_comment(comment_labels_by_line[unit[0]], unit[1], "span-type")
            else:
                add_comment(comment_labels_by_line[unit[0]], unit[1], "span")
        # Type entropy check
        for mention in units_to_mentions[unit]:
            if (
                label_sequences_by_line[unit[0]].span_tokens(mention.mention.span)
                in mentions_to_flag_by_type
            ):
                if (
                    units_to_labels[unit].most_common(1)[0][0]
                    != mentions_to_flag_by_type[
                        label_sequences_by_line[unit[0]].span_tokens(mention.mention.span)
                    ]
                ):
                    add_comment(comment_labels_by_line[unit[0]], unit[1], "check-type")

    for unit in units_to_labels:
        units_to_labels[unit]["O"] += num_annotators - sum(units_to_labels[unit].values())
        for span in check_names_to_flag(
            mentions_to_flag_by_name, label_sequences_by_line[unit[0]]
        ):
            # TODO: restrict to only likely names that are adjudicated to O?
            # Condition to restrict this to only O labels!
            majority_labels = new_labels_from_mentions(
                majority_voted_mentions[unit[0]], len(label_sequences_by_line[unit[0]])
            )
            if all(
                maj_label == "O" for maj_label in majority_labels[span.start : span.end]
            ):
                add_comment(comment_labels_by_line[unit[0]], span, "check-name")
    f_kappa = fleiss_kappa(units_to_labels, num_annotators, labels)
    new_sentences = tuple(
        [
            AdjudicationSequence.create_sequence(
                label_sequence,
                majority_voted_mentions[label_sequence.provenance.starting_line],
                conflict_mentions[label_sequence.provenance.starting_line],
                comment_labels_by_line[label_sequence.provenance.starting_line],
            )
            for label_sequence in annotator_label_sequences
        ]
    )
    return DisagreementResult(
        disagreements,
        disaggrements_by_type,
        disagreements_by_span,
        f_kappa,
        new_sentences,
    )


def print_results(adjudication_results: AdjudicationResult) -> None:
    print("Disagreement Stats: ")
    print(f"\tFleiss kappa: {adjudication_results.disagreements.fleiss_kappa}")
    total_disagree = adjudication_results.disagreements.disagreements
    by_type = adjudication_results.disagreements.disaggrements_by_type
    by_span = adjudication_results.disagreements.disagreements_by_span
    print(f"\tDisagreements: {total_disagree}")
    print(f"\tDisagreements by type: {by_type} {by_type / total_disagree:.3f}")
    print(f"\tDisagreements by span: {by_span} {by_span / total_disagree:.3f}")
    print()


def write_adjudicated_doc(
    majority_vote_sequences: Sequence[AdjudicationSequence],
    file_encoding: str,
    output_delim: str,
    output_file: str,
):
    with open(output_file, "w", encoding=file_encoding) as outfile:
        for sequence in majority_vote_sequences:
            for line in sequence.output_lines(delim=output_delim):
                print(line, file=outfile)
            print(file=outfile)


def write_entropy_info(entropy_out_dir: str, adjudication_results: AdjudicationResult):
    ent_entropy_filepath = os.path.join(entropy_out_dir, "entity_entropy_file.txt")
    type_entropy_filepath = os.path.join(entropy_out_dir, "type_entropy_file.txt")
    with open(ent_entropy_filepath, "w", encoding="utf8") as ent_entropy_file:
        for (
            mention,
            entropy,
        ) in adjudication_results.overall_entity_counts.highest_entropy_mentions():
            if entropy > 0.0:
                print(
                    f"{' '.join(mention)}\t{entropy}\t{adjudication_results.overall_entity_counts.most_common_types(mention, n=1)[0][0]}",
                    file=ent_entropy_file,
                )
    with open(type_entropy_filepath, "w", encoding="utf8") as type_entropy_file:
        for (
            mention,
            entropy,
        ) in adjudication_results.overall_type_counts.highest_entropy_mentions():
            if entropy > 0.0:
                print(
                    f"{' '.join(mention)}\t{entropy}\t{adjudication_results.overall_type_counts.most_common_types(mention, n=1)[0][0]}",
                    file=type_entropy_file,
                )


def adjudicate_documents(
    docs_by_annotator: Dict[int, List[List[LabeledSequence]]]
) -> AdjudicationResult:

    entity_entropy_by_annotator = {
        annotator: MentionCounts() for annotator in docs_by_annotator
    }
    type_entropy_by_annotator: Dict[int, MentionCounts] = {
        annotator: MentionCounts() for annotator in docs_by_annotator
    }
    overall_entity_entropy = MentionCounts()
    overall_type_entropy = MentionCounts()

    single_annotator_label_sequence = [
        label_seq for doc in docs_by_annotator[0] for label_seq in doc
    ]
    seq_id_to_span_to_mentions = defaultdict(lambda: defaultdict(list))
    labels_by_annotator_by_seq = defaultdict(dict)
    for annotator in docs_by_annotator:
        for doc in docs_by_annotator[annotator]:
            for labeled_sequence in doc:
                # annotator_mentions_by_span = defaultdict(list)
                labels_by_annotator_by_seq[labeled_sequence.provenance.starting_line][
                    annotator
                ] = labeled_sequence.labels
                for token_label_span in yield_all_ngrams(
                    labeled_sequence.tokens_with_labels()
                ):
                    tokens = tuple(t[0] for t in token_label_span)
                    if has_outside(token_label_span):
                        entity_entropy_by_annotator[annotator].put(tokens, NON_ENTITY)
                        overall_entity_entropy.put(tokens, NON_ENTITY)
                    else:
                        entity_entropy_by_annotator[annotator].put(tokens, ENTITY)
                        overall_entity_entropy.put(tokens, ENTITY)

                for mention in labeled_sequence.mentions:

                    # Count mentions
                    mention_tokens = labeled_sequence.mention_tokens(mention)
                    type_entropy_by_annotator[annotator].put(mention_tokens, mention.type)
                    overall_type_entropy.put(mention_tokens, mention.type)

                    # annotator_mentions_by_span[mention.span].append(
                    #     AnnotatorMention(mention, annotator)
                    # )
                    seq_id_to_span_to_mentions[labeled_sequence.provenance.starting_line][
                        mention.span
                    ].append(AnnotatorMention(mention, annotator))
    annotator_label_sequences = [
        AnnotatorLabeledSequence(
            labeled_sequence.tokens,
            seq_id_to_span_to_mentions[labeled_sequence.provenance.starting_line],
            labels_by_annotator_by_seq[labeled_sequence.provenance.starting_line],
            provenance=labeled_sequence.provenance,
        )
        for labeled_sequence in single_annotator_label_sequence
    ]
    disagreements = check_disagreements(
        annotator_label_sequences,
        num_annotators=len(docs_by_annotator),
        name_mention_counts=overall_entity_entropy,
        type_mention_counts=overall_type_entropy,
    )

    return AdjudicationResult(
        overall_entity_entropy,
        entity_entropy_by_annotator,
        overall_type_entropy,
        type_entropy_by_annotator,
        disagreements,
    )
