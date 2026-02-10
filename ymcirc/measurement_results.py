"""Class for aggregating and analyzing measurement results from lattice quantum circuits."""
from __future__ import annotations
import logging
from typing import Dict
import copy
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
        if not counts:
            raise ValueError("counts must be non-empty.")
        self._counts: Dict[ParsedLatticeResult, int] = dict(counts)
        self._encoder: LatticeStateEncoder = encoder
        self._total_shots: int = sum(counts.values())
        if self._total_shots <= 0:
            raise ValueError("Total shot count must be positive.")

    def get_counts(self, str_keys: bool = False) -> Dict[ParsedLatticeResult, int] | Dict[str, int]:
        """
        Return a copy of the underlying measurement data as a dict.

        Optionally, use (encoded state) bit strings as the keys.
        """
        if str_keys is False:
            counts_data = copy.deepcopy(self._counts)
        else:
            counts_data = {plr.global_lattice_measurement_bit_string: n_obs for plr, n_obs in copy.deepcopy(self._counts).items()}

        return counts_data

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

        Supports both full and partial states. For partial states (created
        via factory methods like from_links_and_vertices), matches against
        all measured (non-None) degrees of freedom.

        Arguments:
            - state: A ParsedLatticeResult to match against the counts.
        """
        matching_shots = 0
        for parsed, count in self._counts.items():
            if self._states_match(state, parsed):
                matching_shots += count
        return matching_shots / self._total_shots

    @staticmethod
    def _states_match(query: ParsedLatticeResult, candidate: ParsedLatticeResult) -> bool:
        """
        Check if all measured (non-None) DOFs in query match candidate.

        Returns True if every link and vertex that is not None in query
        has the same decoded value in candidate. Unmeasured (None) DOFs
        in query are treated as wildcards.
        """
        if query.dim != candidate.dim or query.shape != candidate.shape:
            return False
        for link_addr in query.link_addresses:
            q_state = query.get_link(link_addr)
            if q_state is not None and q_state != candidate.get_link(link_addr):
                return False
        for vertex_addr in query.vertex_addresses:
            q_state = query.get_vertex(vertex_addr)
            if q_state is not None and q_state != candidate.get_vertex(vertex_addr):
                return False
        return True
