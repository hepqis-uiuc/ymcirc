"""Deeper investigation of ParsedLatticeResult bottleneck.

Breaks down PLR construction time into individual components
to identify which operations dominate the cost.

# TODO: Consider migrating to pytest-benchmark for automated regression detection.
# pytest-benchmark can save baseline results as JSON and fail CI if performance
# regresses beyond a configurable threshold. Install with `uv add --dev pytest-benchmark`
# and convert profile_components() into individual pytest fixture-based tests.

"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import timeit
import copy
from ymcirc._abstract import LatticeDef
from ymcirc.conventions import (
    IRREP_TRUNCATIONS, LatticeStateEncoder, PHYSICAL_PLAQUETTE_STATES,
    get_data_metadata, ONE
)
from ymcirc.parsed_lattice_result import ParsedLatticeResult


def make_encoder(dim_string, trunc_string, lattice_size):
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
    link_bits = encoder.encode_link_state_as_bit_string(ONE)
    vertex_bits = encoder.encode_vertex_state_as_bit_string(0) if encoder.expected_vertex_bit_string_length > 0 else ""
    bitstring = ""
    for _, link_addrs in encoder.lattice_def.get_traversal_order():
        bitstring += vertex_bits
        for _ in link_addrs:
            bitstring += link_bits
    return bitstring


def profile_components(dim_string, trunc_string, lattice_size, n_iterations=50):
    encoder = make_encoder(dim_string, trunc_string, lattice_size)
    bitstring = make_vacuum_bitstring(encoder)
    lattice_def = encoder.lattice_def

    print(f"\n{'='*60}")
    print(f"{dim_string}, {trunc_string}, size={lattice_size}")
    print(f"  Bitstring length: {len(bitstring)} bits")
    print(f"  Plaquette states count: {len(PHYSICAL_PLAQUETTE_STATES[dim_string][trunc_string])}")
    print(f"{'='*60}")

    # 1. Time deep copy of encoder
    times = timeit.repeat(lambda: copy.deepcopy(encoder), number=1, repeat=n_iterations)
    avg = (sum(times)/len(times))*1000
    print(f"  copy.deepcopy(encoder):       {avg:.3f} ms")

    # 2. Time deep copy of lattice_def
    times = timeit.repeat(lambda: copy.deepcopy(lattice_def), number=1, repeat=n_iterations)
    avg = (sum(times)/len(times))*1000
    print(f"  copy.deepcopy(lattice_def):   {avg:.3f} ms")

    # 3. Time super().__init__ (LatticeDef init)
    def do_latticedef_init():
        LatticeDef(lattice_def.dim, lattice_def.shape[0], lattice_def.periodic_boundary_conds, forder=lattice_def.forder)
    times = timeit.repeat(do_latticedef_init, number=1, repeat=n_iterations)
    avg = (sum(times)/len(times))*1000
    print(f"  LatticeDef.__init__():         {avg:.3f} ms")

    # 4. Time get_traversal_order (via property, which deep-copies lattice_def)
    times = timeit.repeat(lambda: encoder.lattice_def.get_traversal_order(), number=1, repeat=n_iterations)
    avg = (sum(times)/len(times))*1000
    print(f"  get_traversal_order():         {avg:.3f} ms")
    # Also time it without the property deep copy
    ld_ref = encoder._lattice  # Access internal ref directly
    times = timeit.repeat(lambda: ld_ref.get_traversal_order(), number=1, repeat=n_iterations)
    avg = (sum(times)/len(times))*1000
    print(f"  get_traversal_order() (no copy): {avg:.3f} ms")

    # 5. Time the decode loop only (without init overhead)
    traversal = ld_ref.get_traversal_order()
    def decode_loop_only():
        start = 0
        for (_, link_addrs) in traversal:
            n_links = len(link_addrs)
            chunk_len = encoder.expected_vertex_bit_string_length + n_links * encoder.expected_link_bit_string_length
            chunk = bitstring[start:start+chunk_len]
            v_end = encoder.expected_vertex_bit_string_length
            v_bits = chunk[:v_end]
            encoder.decode_bit_string_to_vertex_state(v_bits)
            links_sub = chunk[v_end:]
            for i in range(n_links):
                s = i * encoder.expected_link_bit_string_length
                e = s + encoder.expected_link_bit_string_length
                encoder.decode_bit_string_to_link_state(links_sub[s:e])
            start += chunk_len
    times = timeit.repeat(decode_loop_only, number=1, repeat=n_iterations)
    avg = (sum(times)/len(times))*1000
    print(f"  decode loop (no init):        {avg:.3f} ms")

    # 6. Time full PLR construction
    def full_construct():
        ParsedLatticeResult(lattice_def.dim, lattice_def.shape[0], bitstring, encoder, lattice_def.periodic_boundary_conds)
    times = timeit.repeat(full_construct, number=1, repeat=n_iterations)
    avg = (sum(times)/len(times))*1000
    print(f"  Full PLR construction:        {avg:.3f} ms")

    # 7. Time __repr__ of encoder (used on line 104)
    times = timeit.repeat(lambda: encoder.__repr__(), number=1, repeat=n_iterations)
    avg = (sum(times)/len(times))*1000
    print(f"  encoder.__repr__():           {avg:.3f} ms")


if __name__ == "__main__":
    profile_components("d=3/2", "T1", 2, n_iterations=100)
    profile_components("d=2", "T1", 2, n_iterations=100)
    profile_components("d=2", "T1", 4, n_iterations=100)
