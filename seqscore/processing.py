from typing import Dict, Iterable, List, Set

from seqscore.model import LabeledSequence, Mention


class TypeMapper:
    def __init__(
        self,
        keep_types: Iterable[str],
        remove_types: Iterable[str],
        # TODO: Implement type mapping
        type_map: Dict[str, List[str]],
    ):
        # Copy as sets
        self.keep_types: Set[str] = set(keep_types)
        self.remove_types: Set[str] = set(remove_types)
        if self.keep_types and self.remove_types:
            raise ValueError("Cannot specify both keep_types and remove_types")

    def map_types(self, sequence: LabeledSequence) -> LabeledSequence:
        new_mentions: List[Mention] = []
        for mention in sequence.mentions:
            if self.keep_types and mention.type in self.keep_types:
                new_mentions.append(mention)
            elif self.remove_types and mention.type not in self.remove_types:
                new_mentions.append(mention)

        return sequence.with_mentions(new_mentions)


def modify_types(
    docs: List[List[LabeledSequence]],
    keep_types: Set[str],
    remove_types: Set[str],
    type_map: Dict[str, List[str]],
) -> List[List[LabeledSequence]]:
    mapper = TypeMapper(keep_types, remove_types, type_map)
    mapped_docs: List[List[LabeledSequence]] = []
    for doc in docs:
        mapped_docs.append([mapper.map_types(sequence) for sequence in doc])

    return mapped_docs
