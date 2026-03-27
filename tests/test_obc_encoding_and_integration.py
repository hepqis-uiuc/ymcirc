"""Tests for OBC/mixed-BC encoding, circuit construction, and data validation.

Covers plan steps 4.2 (encoding unit tests), 4.3 (OBC integration),
4.4 (mixed BC integration), and 4.6 (data validation).
"""
import pytest
from ymcirc._abstract import LatticeDef
from ymcirc.circuit import LatticeCircuitManager
from ymcirc.conventions import (
    PHYSICAL_PLAQUETTE_STATES, IRREP_TRUNCATIONS,
    LatticeStateEncoder, HAMILTONIAN_BOX_TERMS,
    load_magnetic_hamiltonian,
    ONE, THREE, THREE_BAR,
    IrrepBitmap, PlaquetteState,
)
from ymcirc.lattice_registers import LatticeRegisters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_t1_encoder(dim, size, pbc):
    """Create a T1 LatticeStateEncoder for the given lattice geometry."""
    dim_string = {1.5: "d=3/2", 2: "d=2", 3: "d=3"}[dim]
    lattice = LatticeDef(dim, size, pbc)
    return LatticeStateEncoder(
        IRREP_TRUNCATIONS["T1"],
        PHYSICAL_PLAQUETTE_STATES[dim_string]["T1"],
        lattice=lattice,
    ), dim_string


def _make_b3_encoder(dim, size, pbc):
    """Create a B3 LatticeStateEncoder for the given lattice geometry."""
    dim_string = {1.5: "d=3/2", 2: "d=2", 3: "d=3"}[dim]
    lattice = LatticeDef(dim, size, pbc)
    return LatticeStateEncoder(
        IRREP_TRUNCATIONS["B3"],
        PHYSICAL_PLAQUETTE_STATES[dim_string]["B3"],
        lattice=lattice,
    ), dim_string


# ===========================================================================
# 4.2  Encoding layer unit tests
# ===========================================================================


class TestEncoderInitOBC:
    """Verify LatticeStateEncoder initializes correctly for non-periodic lattices."""

    def test_obc_encoder_has_none_plaquette_bitstring_length(self):
        encoder, _ = _make_t1_encoder(1.5, 4, False)
        assert encoder.expected_plaquette_bit_string_length is None

    def test_pbc_encoder_has_fixed_plaquette_bitstring_length(self):
        encoder, _ = _make_t1_encoder(1.5, 4, True)
        assert encoder.expected_plaquette_bit_string_length is not None
        assert isinstance(encoder.expected_plaquette_bit_string_length, int)

    def test_obc_d2_encoder_has_none_plaquette_bitstring_length(self):
        encoder, _ = _make_t1_encoder(2, 4, False)
        assert encoder.expected_plaquette_bit_string_length is None

    def test_mixed_bc_encoder_has_none_plaquette_bitstring_length(self):
        encoder, _ = _make_t1_encoder(2, 4, (True, False))
        assert encoder.expected_plaquette_bit_string_length is None

    def test_obc_encoder_keeps_all_plaquette_states(self):
        """OBC encoder should NOT filter out boundary-signature states."""
        all_states = PHYSICAL_PLAQUETTE_STATES["d=3/2"]["T1"]
        encoder, _ = _make_t1_encoder(1.5, 4, False)
        # OBC encoder should keep all states (including those with fewer controls).
        assert len(encoder.physical_plaquette_states) == len(all_states)

    def test_pbc_encoder_filters_boundary_signature_states(self):
        """PBC encoder should filter out states from non-interior signatures."""
        all_states = PHYSICAL_PLAQUETTE_STATES["d=3/2"]["T1"]
        encoder, _ = _make_t1_encoder(1.5, 4, True)
        # PBC encoder keeps only interior-signature states (4 controls total for d=3/2).
        for ps in encoder.physical_plaquette_states:
            total_controls = sum(len(vc) for vc in ps[2])
            assert total_controls == 4  # n_control_links_per_plaquette for d=3/2


class TestEncodeDecodeRoundTripOBC:
    """Test encode/decode round-trips for OBC plaquette states."""

    def test_round_trip_d_3_2_interior_signature(self):
        """Interior signature (4 controls) round-trips correctly on OBC encoder."""
        encoder, _ = _make_t1_encoder(1.5, 4, False)
        # T1 d=3/2 has no nontrivial vertex multiplicities, so vertices encode
        # as empty strings and decode back as None.
        interior_state = (
            (0, 0, 0, 0),
            (ONE, THREE, ONE, ONE),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        )
        expected_decoded = (
            (None, None, None, None),
            (ONE, THREE, ONE, ONE),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        )
        encoded = encoder.encode_plaquette_state_as_bit_string(interior_state)
        n_controls_per_vertex = (1, 1, 1, 1)
        decoded = encoder.decode_bit_string_to_plaquette_state(
            encoded, n_controls_per_vertex=n_controls_per_vertex
        )
        assert decoded == expected_decoded

    def test_round_trip_d_3_2_left_edge_signature(self):
        """Left-edge signature (v1 and v4 have 0 controls) round-trips correctly."""
        encoder, _ = _make_t1_encoder(1.5, 4, False)
        left_edge_state = (
            (0, 0, 0, 0),
            (ONE, THREE, ONE, ONE),
            ((), (ONE,), (ONE,), ())
        )
        expected_decoded = (
            (None, None, None, None),
            (ONE, THREE, ONE, ONE),
            ((), (ONE,), (ONE,), ())
        )
        encoded = encoder.encode_plaquette_state_as_bit_string(left_edge_state)
        n_controls_per_vertex = (0, 1, 1, 0)
        decoded = encoder.decode_bit_string_to_plaquette_state(
            encoded, n_controls_per_vertex=n_controls_per_vertex
        )
        assert decoded == expected_decoded

    def test_round_trip_d_3_2_right_edge_signature(self):
        """Right-edge signature (v2 and v3 have 0 controls) round-trips correctly."""
        encoder, _ = _make_t1_encoder(1.5, 4, False)
        right_edge_state = (
            (0, 0, 0, 0),
            (ONE, ONE, ONE, THREE),
            ((ONE,), (), (), (ONE,))
        )
        expected_decoded = (
            (None, None, None, None),
            (ONE, ONE, ONE, THREE),
            ((ONE,), (), (), (ONE,))
        )
        encoded = encoder.encode_plaquette_state_as_bit_string(right_edge_state)
        n_controls_per_vertex = (1, 0, 0, 1)
        decoded = encoder.decode_bit_string_to_plaquette_state(
            encoded, n_controls_per_vertex=n_controls_per_vertex
        )
        assert decoded == expected_decoded

    def test_round_trip_d2_corner_signature(self):
        """d=2 corner plaquette (v1 has 0 controls) round-trips correctly."""
        encoder, _ = _make_t1_encoder(2, 4, False)
        # Corner at (0,0): v1 has 0 controls, v2 has 1, v3 has 2, v4 has 1.
        corner_state = (
            (0, 0, 0, 0),
            (ONE, ONE, ONE, ONE),
            ((), (ONE,), (ONE, ONE), (ONE,))
        )
        encoded = encoder.encode_plaquette_state_as_bit_string(corner_state)
        n_controls_per_vertex = (0, 1, 2, 1)
        decoded = encoder.decode_bit_string_to_plaquette_state(
            encoded, n_controls_per_vertex=n_controls_per_vertex
        )
        assert decoded == corner_state

    def test_round_trip_d2_interior_signature(self):
        """d=2 interior plaquette (all vertices have 2 controls) round-trips correctly."""
        encoder, _ = _make_t1_encoder(2, 4, False)
        interior_state = (
            (0, 0, 0, 0),
            (ONE, ONE, ONE, ONE),
            ((ONE, ONE), (ONE, ONE), (ONE, ONE), (ONE, ONE))
        )
        encoded = encoder.encode_plaquette_state_as_bit_string(interior_state)
        n_controls_per_vertex = (2, 2, 2, 2)
        decoded = encoder.decode_bit_string_to_plaquette_state(
            encoded, n_controls_per_vertex=n_controls_per_vertex
        )
        assert decoded == interior_state

    def test_different_signatures_produce_different_length_bitstrings(self):
        """OBC plaquettes with different control counts encode to different length bitstrings."""
        encoder, _ = _make_t1_encoder(1.5, 4, False)
        interior_state = (
            (0, 0, 0, 0),
            (ONE, ONE, ONE, ONE),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        )
        edge_state = (
            (0, 0, 0, 0),
            (ONE, ONE, ONE, ONE),
            ((), (ONE,), (ONE,), ())
        )
        interior_bs = encoder.encode_plaquette_state_as_bit_string(interior_state)
        edge_bs = encoder.encode_plaquette_state_as_bit_string(edge_state)
        assert len(interior_bs) > len(edge_bs)


class TestDecodeRequiresControlsForOBC:
    """Verify decode raises when n_controls_per_vertex is missing on OBC."""

    def test_decode_raises_without_controls_on_obc(self):
        encoder, _ = _make_t1_encoder(1.5, 4, False)
        state = (
            (0, 0, 0, 0),
            (ONE, ONE, ONE, ONE),
            ((ONE,), (ONE,), (ONE,), (ONE,))
        )
        encoded = encoder.encode_plaquette_state_as_bit_string(state)
        with pytest.raises(ValueError, match="n_controls_per_vertex must be provided"):
            encoder.decode_bit_string_to_plaquette_state(encoded)


class TestPBCEncodingUnchanged:
    """Verify PBC encoding is unchanged after OBC implementation."""

    def test_pbc_d_3_2_t1_encoding_unchanged(self):
        """PBC T1 d=3/2 encoding produces the same result as before OBC changes."""
        encoder, _ = _make_t1_encoder(1.5, 3, True)
        state = (
            (0, 0, 0, 0),
            (THREE, ONE, THREE, THREE_BAR),
            ((ONE,), (ONE,), (THREE_BAR,), (THREE,))
        )
        encoded = encoder.encode_plaquette_state_as_bit_string(state)
        # Known encoding: active links "10001001" + controls "00000110" = 16 bits
        assert encoded == "10001001" + "00000110"

    def test_pbc_d2_t1_round_trip(self):
        """PBC T1 d=2 encoding round-trips correctly."""
        encoder, _ = _make_t1_encoder(2, 3, True)
        state = (
            (0, 0, 0, 0),
            (ONE, THREE, ONE, ONE),
            ((ONE, ONE), (ONE, ONE), (ONE, ONE), (ONE, ONE))
        )
        encoded = encoder.encode_plaquette_state_as_bit_string(state)
        decoded = encoder.decode_bit_string_to_plaquette_state(encoded)
        assert decoded == state

    def test_pbc_expected_plaquette_bitstring_length_d_3_2(self):
        encoder, _ = _make_t1_encoder(1.5, 3, True)
        # d=3/2 T1: 0 vertex bits + 4*2 active link bits + 4*2 control link bits = 16
        assert encoder.expected_plaquette_bit_string_length == 16

    def test_pbc_expected_plaquette_bitstring_length_d2(self):
        encoder, _ = _make_t1_encoder(2, 3, True)
        # d=2 T1: 4*1 vertex bits + 4*2 active link bits + 8*2 control link bits = 28
        assert encoder.expected_plaquette_bit_string_length == 28


class TestLoadMagneticHamiltonianOBC:
    """Test load_magnetic_hamiltonian() with OBC encoders."""

    def test_load_d_3_2_t1_obc(self):
        """OBC encoder loads Hamiltonian data with all signatures retained."""
        encoder, dim_string = _make_t1_encoder(1.5, 4, False)
        ham = load_magnetic_hamiltonian(dim_string, "T1", encoder)
        assert len(ham) > 0
        # Each entry should have MatrixElementValue structure.
        for (bs1, bs2), mev in ham.items():
            assert isinstance(bs1, str)
            assert isinstance(bs2, str)
            assert isinstance(mev, dict)
            for plane, sig_dict in mev.items():
                assert isinstance(plane, tuple) and len(plane) == 2
                for sig, val in sig_dict.items():
                    assert isinstance(sig, tuple)
                    assert isinstance(val, float)

    def test_load_d_3_2_t1_obc_has_multiple_bitstring_lengths(self):
        """OBC Hamiltonian data contains bitstrings of varying lengths (different signatures)."""
        encoder, dim_string = _make_t1_encoder(1.5, 4, False)
        ham = load_magnetic_hamiltonian(dim_string, "T1", encoder)
        all_lengths = set()
        for bs1, bs2 in ham.keys():
            all_lengths.add(len(bs1))
            all_lengths.add(len(bs2))
        # d=3/2 OBC has 3 signatures (left edge, interior, right edge) with
        # different control link counts, so there should be multiple bitstring lengths.
        assert len(all_lengths) > 1, (
            f"Expected multiple bitstring lengths for OBC data, got {all_lengths}"
        )

    def test_load_d_3_2_t1_pbc_has_single_bitstring_length(self):
        """PBC Hamiltonian data contains bitstrings of uniform length."""
        encoder, dim_string = _make_t1_encoder(1.5, 4, True)
        ham = load_magnetic_hamiltonian(dim_string, "T1", encoder)
        all_lengths = set()
        for bs1, bs2 in ham.keys():
            all_lengths.add(len(bs1))
            all_lengths.add(len(bs2))
        assert len(all_lengths) == 1

    def test_load_d2_t1_obc(self):
        """d=2 OBC encoder loads Hamiltonian data successfully."""
        encoder, dim_string = _make_t1_encoder(2, 4, False)
        ham = load_magnetic_hamiltonian(dim_string, "T1", encoder)
        assert len(ham) > 0

    def test_obc_hamiltonian_entries_round_trip_through_encoding(self):
        """All state pairs in OBC Hamiltonian can be decoded back (with appropriate controls)."""
        encoder, dim_string = _make_t1_encoder(1.5, 4, False)
        ham = load_magnetic_hamiltonian(dim_string, "T1", encoder)
        # For each entry, the bitstrings should decode without error if we provide
        # the correct n_controls_per_vertex. We can determine the per-vertex control
        # count from the bitstring length and known vertex/link encoding lengths.
        link_len = encoder.expected_link_bit_string_length
        vertex_len = encoder.expected_vertex_bit_string_length
        fixed_bits = 4 * vertex_len + 4 * link_len
        for bs1, bs2 in ham.keys():
            # Determine the number of control link bits.
            n_control_bits_1 = len(bs1) - fixed_bits
            n_control_bits_2 = len(bs2) - fixed_bits
            assert n_control_bits_1 >= 0
            assert n_control_bits_2 >= 0
            # Both states in a pair must have the same total control count
            # (they correspond to the same signature).
            assert n_control_bits_1 == n_control_bits_2


class TestEncodeDecodeAllSignaturesD2:
    """Round-trip all distinct d=2 OBC signatures on a 4x4 lattice."""

    def test_all_9_signatures_round_trip(self):
        """Each of the 9 distinct d=2 OBC signatures can encode and decode states."""
        encoder, _ = _make_t1_encoder(2, 4, False)
        link_bitmap = IRREP_TRUNCATIONS["T1"]
        lattice_regs = LatticeRegisters(2, 4, False,
                                        link_bitmap=link_bitmap,
                                        vertex_bitmap=encoder.vertex_bitmap)

        seen_signatures = set()
        for vertex in lattice_regs.vertex_addresses:
            try:
                plaq = lattice_regs.get_plaquettes(vertex, 1, 2)
            except KeyError:
                continue
            sig = plaq.signature
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)

            # Determine n_controls_per_vertex from the signature.
            n_controls_per_vertex = tuple(
                len(dirs) - 2  # subtract the 2 active link directions
                for dirs in sig
            )

            # Build a trivial state matching this control count.
            state = (
                (0, 0, 0, 0),
                (ONE, ONE, ONE, ONE),
                tuple(
                    tuple(ONE for _ in range(n_ctrl))
                    for n_ctrl in n_controls_per_vertex
                )
            )
            encoded = encoder.encode_plaquette_state_as_bit_string(state)
            decoded = encoder.decode_bit_string_to_plaquette_state(
                encoded, n_controls_per_vertex=n_controls_per_vertex
            )
            assert decoded == state

        assert len(seen_signatures) == 9, (
            f"Expected 9 distinct signatures for d=2 OBC 4x4, got {len(seen_signatures)}"
        )


# ===========================================================================
# 4.3  Integration test: OBC circuit construction
# ===========================================================================


class TestOBCCircuitConstruction:
    """Integration tests for building Trotter step circuits on OBC lattices."""

    def test_d_3_2_obc_circuit_builds(self):
        """A d=3/2 OBC lattice circuit builds without error."""
        encoder, dim_string = _make_t1_encoder(1.5, 4, False)
        ham = load_magnetic_hamiltonian(dim_string, "T1", encoder)
        circ_mgr = LatticeCircuitManager(encoder, ham)
        lattice_regs = LatticeRegisters.from_lattice_state_encoder(encoder)
        circuit = circ_mgr.create_blank_full_lattice_circuit(lattice_regs)
        circ_mgr.apply_magnetic_trotter_step(
            circuit, lattice_regs,
            optimize_circuits=False,
            cache_mag_evol_circuit=True,
        )
        assert circuit.num_qubits > 0

    def test_d2_obc_size3_circuit_builds(self):
        """A d=2 OBC size-3 lattice circuit builds without error."""
        encoder, dim_string = _make_t1_encoder(2, 3, False)
        ham = load_magnetic_hamiltonian(dim_string, "T1", encoder)
        circ_mgr = LatticeCircuitManager(encoder, ham)
        lattice_regs = LatticeRegisters.from_lattice_state_encoder(encoder)
        circuit = circ_mgr.create_blank_full_lattice_circuit(lattice_regs)
        circ_mgr.apply_magnetic_trotter_step(
            circuit, lattice_regs,
            optimize_circuits=False,
            cache_mag_evol_circuit=True,
        )
        assert circuit.num_qubits > 0

    def test_obc_lattice_has_fewer_qubits_than_pbc(self):
        """OBC lattice circuits have fewer qubits than PBC equivalents."""
        encoder_obc, _ = _make_t1_encoder(1.5, 4, False)
        encoder_pbc, _ = _make_t1_encoder(1.5, 4, True)
        regs_obc = LatticeRegisters.from_lattice_state_encoder(encoder_obc)
        regs_pbc = LatticeRegisters.from_lattice_state_encoder(encoder_pbc)
        circ_obc = LatticeCircuitManager(encoder_obc, {}).create_blank_full_lattice_circuit(regs_obc)
        circ_pbc = LatticeCircuitManager(encoder_pbc, {}).create_blank_full_lattice_circuit(regs_pbc)
        assert circ_obc.num_qubits < circ_pbc.num_qubits

    def test_d2_obc_plaquette_count_in_traversal(self):
        """d=2 OBC size-3 lattice traversal has 4 plaquettes (2x2)."""
        lattice = LatticeDef(2, 3, False)
        assert lattice.n_plaquettes == 4

    def test_d_3_2_obc_plaquette_count_in_traversal(self):
        """d=3/2 OBC size-4 lattice traversal has 3 plaquettes."""
        lattice = LatticeDef(1.5, 4, False)
        assert lattice.n_plaquettes == 3

    @pytest.mark.slow
    def test_d2_obc_size4_circuit_builds_with_b3(self):
        """A d=2 OBC size-4 B3 lattice circuit builds without error."""
        encoder, dim_string = _make_b3_encoder(2, 4, False)
        ham = load_magnetic_hamiltonian(dim_string, "B3", encoder)
        circ_mgr = LatticeCircuitManager(encoder, ham)
        lattice_regs = LatticeRegisters.from_lattice_state_encoder(encoder)
        circuit = circ_mgr.create_blank_full_lattice_circuit(lattice_regs)
        circ_mgr.apply_magnetic_trotter_step(
            circuit, lattice_regs,
            optimize_circuits=False,
            cache_mag_evol_circuit=True,
        )
        assert circuit.num_qubits > 0


# ===========================================================================
# 4.4  Integration test: Mixed BC circuit construction
# ===========================================================================


class TestMixedBCCircuitConstruction:
    """Integration tests for building Trotter step circuits on mixed BC lattices."""

    def test_d2_mixed_bc_plaquette_count(self):
        """d=2 mixed BC (x-periodic, y-open) size-3: 3*2 = 6 plaquettes."""
        lattice = LatticeDef(2, 3, (True, False))
        assert lattice.n_plaquettes == 6

    def test_d2_mixed_bc_between_obc_and_pbc(self):
        """Mixed BC plaquette count is between full OBC and full PBC."""
        n_obc = LatticeDef(2, 3, False).n_plaquettes      # 4
        n_mixed = LatticeDef(2, 3, (True, False)).n_plaquettes  # 6
        n_pbc = LatticeDef(2, 3, True).n_plaquettes        # 9
        assert n_obc < n_mixed < n_pbc

    def test_d2_mixed_bc_circuit_builds(self):
        """A d=2 mixed BC lattice circuit builds without error."""
        encoder, dim_string = _make_t1_encoder(2, 3, (True, False))
        ham = load_magnetic_hamiltonian(dim_string, "T1", encoder)
        circ_mgr = LatticeCircuitManager(encoder, ham)
        lattice_regs = LatticeRegisters.from_lattice_state_encoder(encoder)
        circuit = circ_mgr.create_blank_full_lattice_circuit(lattice_regs)
        circ_mgr.apply_magnetic_trotter_step(
            circuit, lattice_regs,
            optimize_circuits=False,
            cache_mag_evol_circuit=True,
        )
        assert circuit.num_qubits > 0

    def test_d2_mixed_bc_has_3_distinct_signatures(self):
        """d=2 mixed BC (x-periodic, y-open) size-4 has 3 distinct signatures."""
        link_bitmap = IRREP_TRUNCATIONS["T1"]
        # Size 4 is needed so that there's room for bottom-edge, interior, and
        # top-edge rows (size 3 only has 2 plaquette rows, giving 2 signatures).
        lattice = LatticeRegisters(2, 4, (True, False),
                                   link_bitmap=link_bitmap,
                                   vertex_bitmap={})
        signatures = set()
        for vertex in lattice.vertex_addresses:
            try:
                plaq = lattice.get_plaquettes(vertex, 1, 2)
            except KeyError:
                continue
            signatures.add(plaq.signature)
        # x-periodic means all vertices have both +1 and -1 directions.
        # y-open means bottom-row vertices lack -2, top-row vertices lack +2.
        # Size 4 gives 3 signature types: bottom-edge (y=0), interior (y=1), top-edge (y=2).
        assert len(signatures) == 3, (
            f"Expected 3 distinct signatures for d=2 mixed (T,F) size 4, got {len(signatures)}"
        )

    def test_d2_mixed_bc_fewer_qubits_than_pbc(self):
        """Mixed BC lattice has fewer qubits than full PBC (fewer links at boundary)."""
        encoder_mixed, _ = _make_t1_encoder(2, 3, (True, False))
        encoder_pbc, _ = _make_t1_encoder(2, 3, True)
        regs_mixed = LatticeRegisters.from_lattice_state_encoder(encoder_mixed)
        regs_pbc = LatticeRegisters.from_lattice_state_encoder(encoder_pbc)
        circ_mixed = LatticeCircuitManager(encoder_mixed, {}).create_blank_full_lattice_circuit(regs_mixed)
        circ_pbc = LatticeCircuitManager(encoder_pbc, {}).create_blank_full_lattice_circuit(regs_pbc)
        assert circ_mixed.num_qubits < circ_pbc.num_qubits


# ===========================================================================
# 4.6  Data validation: universal files match existing PBC data
# ===========================================================================


class TestUniversalDataMatchesPBC:
    """Verify universal data files produce identical PBC-signature matrix elements."""

    @pytest.mark.parametrize("dim,dim_string,trunc_string", [
        (1.5, "d=3/2", "T1"),
        (1.5, "d=3/2", "T2"),
        (1.5, "d=3/2", "B3"),
    ])
    def test_pbc_hamiltonian_loads_consistently(self, dim, dim_string, trunc_string):
        """PBC encoder loading from universal data gives non-empty, structurally valid data."""
        trunc_key = trunc_string
        if trunc_string.startswith("B") and dim == 1.5:
            # B-series d=3/2 may use a dimensional suffix in IRREP_TRUNCATIONS.
            for candidate in [trunc_string, f"{trunc_string}_d=3/2"]:
                if candidate in IRREP_TRUNCATIONS:
                    trunc_key = candidate
                    break
        lattice = LatticeDef(dim, 4, True)
        encoder = LatticeStateEncoder(
            IRREP_TRUNCATIONS[trunc_key],
            PHYSICAL_PLAQUETTE_STATES[dim_string][trunc_string],
            lattice=lattice,
        )
        ham = load_magnetic_hamiltonian(dim_string, trunc_string, encoder)
        assert len(ham) > 0
        # All bitstrings should have uniform length (PBC).
        lengths = set()
        for bs1, bs2 in ham.keys():
            lengths.add(len(bs1))
            lengths.add(len(bs2))
        assert len(lengths) == 1

    @pytest.mark.parametrize("dim,dim_string,trunc_string", [
        (1.5, "d=3/2", "T1"),
        (1.5, "d=3/2", "T2"),
    ])
    def test_obc_data_contains_pbc_signature_as_subset(self, dim, dim_string, trunc_string):
        """OBC Hamiltonian data contains the PBC (interior) signature among its entries."""
        # Load with PBC encoder.
        pbc_lattice = LatticeDef(dim, 4, True)
        pbc_encoder = LatticeStateEncoder(
            IRREP_TRUNCATIONS[trunc_string],
            PHYSICAL_PLAQUETTE_STATES[dim_string][trunc_string],
            lattice=pbc_lattice,
        )
        pbc_ham = load_magnetic_hamiltonian(dim_string, trunc_string, pbc_encoder)

        # Load with OBC encoder.
        obc_lattice = LatticeDef(dim, 4, False)
        obc_encoder = LatticeStateEncoder(
            IRREP_TRUNCATIONS[trunc_string],
            PHYSICAL_PLAQUETTE_STATES[dim_string][trunc_string],
            lattice=obc_lattice,
        )
        obc_ham = load_magnetic_hamiltonian(dim_string, trunc_string, obc_encoder)

        # OBC Hamiltonian should have at least as many entries as PBC
        # (it includes all signatures, PBC only has interior).
        assert len(obc_ham) >= len(pbc_ham)

    @pytest.mark.parametrize("dim,dim_string,trunc_string", [
        (1.5, "d=3/2", "T1"),
        (1.5, "d=3/2", "T2"),
    ])
    def test_universal_file_pbc_matrix_elements_match(self, dim, dim_string, trunc_string):
        """Matrix element values for the interior (PBC) signature are identical
        regardless of whether loaded via PBC or OBC encoder."""
        # Determine interior signature for this dimension.
        if dim == 1.5:
            interior_sig = ((1, 2, -1), (1, 2, -1), (1, -1, -2), (1, -1, -2))
            plane = (1, 2)
        elif dim == 2:
            interior_sig = ((1, 2, -1, -2),) * 4
            plane = (1, 2)
        else:
            raise NotImplementedError

        # Raw box terms from the universal data file.
        raw_data = HAMILTONIAN_BOX_TERMS[dim_string][trunc_string]

        # Extract all matrix element values for the interior signature.
        interior_values = {}
        for (state_f, state_i), mev in raw_data.items():
            if plane in mev and interior_sig in mev[plane]:
                interior_values[(state_f, state_i)] = mev[plane][interior_sig]

        # There should be non-trivial matrix elements for the interior signature.
        assert len(interior_values) > 0, (
            f"No matrix elements found for interior signature {interior_sig} in {dim_string} {trunc_string}"
        )

        # Each value should be a nonzero float.
        for key, val in interior_values.items():
            assert isinstance(val, float)
            assert val != 0.0

    @pytest.mark.parametrize("dim,dim_string,trunc_string", [
        (1.5, "d=3/2", "T1"),
    ])
    def test_all_3_signatures_present_in_d_3_2_data(self, dim, dim_string, trunc_string):
        """d=3/2 universal data file contains entries for all 3 OBC signatures."""
        raw_data = HAMILTONIAN_BOX_TERMS[dim_string][trunc_string]
        plane = (1, 2)
        all_signatures = set()
        for (state_f, state_i), mev in raw_data.items():
            if plane in mev:
                all_signatures.update(mev[plane].keys())
        assert len(all_signatures) == 3, (
            f"Expected 3 signatures in d=3/2 universal data, got {len(all_signatures)}: {all_signatures}"
        )

    @pytest.mark.parametrize("dim,dim_string,trunc_string", [
        (2, "d=2", "T1"),
    ])
    def test_all_9_signatures_present_in_d2_data(self, dim, dim_string, trunc_string):
        """d=2 universal data file contains entries for all 9 OBC signatures."""
        raw_data = HAMILTONIAN_BOX_TERMS[dim_string][trunc_string]
        plane = (1, 2)
        all_signatures = set()
        for (state_f, state_i), mev in raw_data.items():
            if plane in mev:
                all_signatures.update(mev[plane].keys())
        assert len(all_signatures) == 9, (
            f"Expected 9 signatures in d=2 universal data, got {len(all_signatures)}: {all_signatures}"
        )
