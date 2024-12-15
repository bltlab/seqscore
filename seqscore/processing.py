from typing import Dict, Iterable, List, Set

from seqscore.model import LabeledSequence, Mention


class TypeMapper:
    def __init__(
        self,
        keep_types: Iterable[str],
        remove_types: Iterable[str],
        type_map: Dict[str, List[str]],
    ):
        # Copy keep/remove as sets
        self.keep_types: Set[str] = set(keep_types)
        self.remove_types: Set[str] = set(remove_types)
        # Since the CLI prevents these from both being specified, this can't be hit by tests
        if self.keep_types and self.remove_types:  # pragma: no cover
            raise ValueError("Cannot specify both keep_types and remove_types")

        # Invert the type map
        self.type_map: Dict[str, str] = {}
        for to_type, from_types in type_map.items():
            assert to_type  # Type cannot be blank
            for from_type in from_types:
                assert from_type  # Type cannot be blank
                if from_type in self.type_map:
                    raise ValueError(
                        f"Multiple mappings specified for type {repr(from_type)} in type map"
                    )
                else:
                    self.type_map[from_type] = to_type

    def map_types(self, sequence: LabeledSequence) -> LabeledSequence:
        new_mentions: List[Mention] = []
        for mention in sequence.mentions:
            if mention.type in self.type_map:
                mention = mention.with_type(self.type_map[mention.type])

            if (self.keep_types and mention.type not in self.keep_types) or (
                self.remove_types and mention.type in self.remove_types
            ):
                continue

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
