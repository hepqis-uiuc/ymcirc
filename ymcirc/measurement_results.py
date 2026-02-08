"""Class for aggregating and analyzing measurement results from lattice quantum circuits."""
from __future__ import annotations
import logging
import warnings
from typing import Dict, Union
from ymcirc._abstract.lattice_data import LinkAddress
from ymcirc.conventions import LatticeStateEncoder, ONE
from ymcirc.parsed_lattice_result import ParsedLatticeResult

logger = logging.getLogger(__name__)


class MeasurementResults:
    """
    Aggregates measurement counts keyed by ParsedLatticeResult instances.

    Analogous to the counts dictionary returned by Qiskit circuit executions,
    but uses ParsedLatticeResult instances as keys for physics-aware analysis.

    Provides methods to compute expectation values of electric energy,
    vacuum persistence probability, and transition probabilities.

    Arguments:
        - counts: Dict mapping ParsedLatticeResult -> int (shot count).
        - encoder: LatticeStateEncoder for lattice geometry and encoding info.
    """

    def __init__(
        self,
        counts: Dict[ParsedLatticeResult, int],
        encoder: LatticeStateEncoder,
    ):
        self._counts = dict(counts)
        self._encoder = encoder
        self._total_shots = sum(counts.values())

    def get_link_electric_energy(self, link_address: LinkAddress) -> float:
        """
        Return the expectation value of the electric Casimir energy at a link.

        Computes the shot-weighted average of
        ParsedLatticeResult.get_link_electric_energy over all measurement
        outcomes. Links that return None are treated as contributing 0.

        Arguments:
            - link_address: Address of the link, e.g. ((0,0), 1).
        """
        total = 0.0
        for parsed, count in self._counts.items():
            energy = parsed.get_link_electric_energy(link_address)
            if energy is not None:
                total += energy * count
        return total / self._total_shots

    def get_lattice_electric_energy(self, average_result: bool = False) -> float:
        """
        Return the expectation value of the lattice electric energy.

        Computes the shot-weighted average of
        ParsedLatticeResult.get_lattice_electric_energy(average_result)
        over all measurement outcomes. The average_result flag is applied
        per-instance (controlling whether each instance reports total or
        per-link energy); the method always averages over all shots.

        Arguments:
            - average_result: Passed to each ParsedLatticeResult instance.
              If True, each instance returns energy per link.
              If False, each instance returns total energy.
        """
        total = 0.0
        for parsed, count in self._counts.items():
            energy = parsed.get_lattice_electric_energy(average_result=average_result)
            total += energy * count
        return total / self._total_shots

    @property
    def vacuum_persistence_probability(self) -> float:
        """
        Return the empirical probability for the entire lattice to be in vacuum.

        Vacuum is defined as all links being in the singlet state ONE = (0,0,0).
        """
        vacuum_shots = 0
        for parsed, count in self._counts.items():
            is_vacuum = True
            for vertex_addr, link_addrs in parsed.get_traversal_order():
                for link_addr in link_addrs:
                    link_state = parsed.get_link(link_addr)
                    if link_state != ONE:
                        is_vacuum = False
                        break
                if not is_vacuum:
                    break
            if is_vacuum:
                vacuum_shots += count
        return vacuum_shots / self._total_shots

    def get_transition_probability(self, state: ParsedLatticeResult) -> float:
        """
        Return the empirical probability of the given state.

        Arguments:
            - state: A ParsedLatticeResult to look up in the counts dict.
        """
        return self._counts.get(state, 0) / self._total_shots
