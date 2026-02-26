"""Benchmark ParsedLatticeResult construction for small vs large bitstrings."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import timeit
from ymcirc._abstract import LatticeDef
from ymcirc.conventions import (
    IRREP_TRUNCATIONS, LatticeStateEncoder, PHYSICAL_PLAQUETTE_STATES,
    get_data_metadata, ONE
)
from ymcirc.parsed_lattice_result import ParsedLatticeResult


def make_encoder(dim_string, trunc_string, lattice_size):
    """Create a LatticeStateEncoder for the given config."""
    if dim_string == "d=3/2":
        dim = 1.5
    else:
        dim = int(dim_string[2:])
    link_bitmap = IRREP_TRUNCATIONS[trunc_string]
    plaquette_states = PHYSICAL_PLAQUETTE_STATES[dim_string][trunc_string]
    metadata = get_data_metadata(dim_string, trunc_string)
    forder = metadata.get("f_order", None)
    lattice_def = LatticeDef(dim, lattice_size, True, forder=forder)
    return LatticeStateEncoder(link_bitmap, plaquette_states, lattice_def)


def make_vacuum_bitstring(encoder):
    """Create the all-vacuum bitstring for the given encoder."""
    link_bits = encoder.encode_link_state_as_bit_string(ONE)
    vertex_bits = encoder.encode_vertex_state_as_bit_string(0) if encoder.expected_vertex_bit_string_length > 0 else ""
    bitstring = ""
    for _, link_addrs in encoder.lattice_def.get_traversal_order():
        bitstring += vertex_bits
        for _ in link_addrs:
            bitstring += link_bits
    return bitstring


def benchmark_config(dim_string, trunc_string, lattice_size, n_iterations=100):
    """Benchmark PLR construction for a given config."""
    encoder = make_encoder(dim_string, trunc_string, lattice_size)
    bitstring = make_vacuum_bitstring(encoder)
    lattice_def = encoder.lattice_def

    print(f"\n--- {dim_string}, {trunc_string}, size={lattice_size} ---")
    print(f"  Bitstring length: {len(bitstring)}")
    print(f"  Vertices: {lattice_def.n_vertices}, Links: {lattice_def.n_links}")

    def construct():
        ParsedLatticeResult(lattice_def.dim, lattice_def.shape[0], bitstring, encoder, lattice_def.periodic_boundary_conds)

    # Warmup
    construct()

    # Time it
    times = timeit.repeat(construct, number=1, repeat=n_iterations)
    avg_ms = (sum(times) / len(times)) * 1000
    min_ms = min(times) * 1000
    max_ms = max(times) * 1000
    print(f"  Avg: {avg_ms:.3f} ms, Min: {min_ms:.3f} ms, Max: {max_ms:.3f} ms  (over {n_iterations} runs)")
    return avg_ms


if __name__ == "__main__":
    print("=" * 60)
    print("ParsedLatticeResult construction benchmark")
    print("=" * 60)

    # Small: d=3/2, T1, size=2 (fewest vertices/links)
    t_small = benchmark_config("d=3/2", "T1", 2, n_iterations=200)

    # Medium: d=2, T1, size=2 (the case from the issue)
    t_medium = benchmark_config("d=2", "T1", 2, n_iterations=200)

    # Large: d=2, T1, size=4
    t_large = benchmark_config("d=2", "T1", 4, n_iterations=200)

    # Even larger: d=3/2, T2, size=4
    t_xlarge = benchmark_config("d=3/2", "T2", 4, n_iterations=200)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  d=3/2, T1, size=2:  {t_small:.3f} ms")
    print(f"  d=2,   T1, size=2:  {t_medium:.3f} ms")
    print(f"  d=2,   T1, size=4:  {t_large:.3f} ms")
    print(f"  d=3/2, T2, size=4:  {t_xlarge:.3f} ms")
    print(f"\n  Ratio (large/small): {t_large/t_small:.2f}x")
    print(f"  Ratio (medium/small): {t_medium/t_small:.2f}x")
