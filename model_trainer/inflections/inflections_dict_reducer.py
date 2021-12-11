import logging
from dataclasses import dataclass
from typing import Dict

from tqdm import tqdm

from utils.logging import measure_time

log = logging.getLogger(__name__)


@dataclass
class EquivalenceGroup:
    representative: str

    def __str__(self) -> str:
        return f"{self.representative} ({id(self)})"


class InflectionsDictReducer:
    def __init__(self):
        self.equivalence_groups: Dict[str, EquivalenceGroup] = {}

    def reduce(self, word: str, representative: str):
        equivalence_group = None
        if word in self.equivalence_groups:
            equivalence_group = self.equivalence_groups[word]
            equivalence_group.representative = representative
        if representative not in self.equivalence_groups:
            if equivalence_group is None:
                equivalence_group = EquivalenceGroup(representative)
            self.equivalence_groups[representative] = equivalence_group
        self.equivalence_groups[word] = self.equivalence_groups[representative]

    def generate_reduced_dict(self) -> dict:
        reduced = {}
        for word, equivalence_group in self.equivalence_groups.items():
            if word != equivalence_group.representative:
                reduced[word] = equivalence_group.representative
        return reduced


@measure_time
def reduce_inflections_dict(inflections: dict) -> dict:
    log.debug(f"Reducing dict of size {len(inflections)}")
    reducer = InflectionsDictReducer()
    for word, representative in tqdm(inflections.items()):
        reducer.reduce(word, representative)
    return reducer.generate_reduced_dict()
