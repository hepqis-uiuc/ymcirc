from ymcirc.conventions import (
    VERTEX_SINGLET_BITMAPS, IRREP_TRUNCATION_DICT_1_3_3BAR,
    IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8, ONE, THREE,
    THREE_BAR, SIX, SIX_BAR, EIGHT)


def test_singlet_bitmaps():
    print("Testing singlet bitmaps...")
    # Check that there are 8 vertex singlet bitmaps (3 dimensionalities * 2 irrep truncations + T1p in two dimensions).
    there_are_six_singlet_bitmaps = len(VERTEX_SINGLET_BITMAPS) == 8
    print(
        f"\nlen(VERTEX_SINGLET_BITMAPS) == 8? {there_are_six_singlet_bitmaps}."
    )
    assert there_are_six_singlet_bitmaps, f"Encountered {len(VERTEX_SINGLET_BITMAPS)} bitmaps."


def test_vertex_bitmaps_have_right_amount_of_singlets():
    # Check that all vertex bitmaps have the right amount of distinct singlets.
    # The values for expected_nums come from classical precomputation.
    cases = [
        "d=3/2, T1", "d=3/2, T2", "d=2, T1", "d=2, T2", "d=3, T1", "d=3, T2"
    ]
    expected_nums = [4, 16, 6, 66, 28, 1646]
    for current_case, expected_num_singlets in zip(cases, expected_nums):
        # Number check.
        test_result_num_singlets = len(
            VERTEX_SINGLET_BITMAPS[current_case]) == expected_num_singlets
        print(
            f'\nlen(VERTEX_SINGLET_BITMAPS["{current_case}"]) == {expected_num_singlets}? {test_result_num_singlets}.'
        )
        if test_result_num_singlets is False:
            print(
                f"Encountered {len(VERTEX_SINGLET_BITMAPS[current_case])} singlets."
            )

        assert test_result_num_singlets

        # Uniqueness of encoding check.
        test_result_singlets_have_unique_bit_string = expected_num_singlets == len(
            set(VERTEX_SINGLET_BITMAPS[current_case].values()))
        print(
            f'# of distinct bit string encodings for VERTEX_SINGLET_BITMAPS["{current_case}"] == {expected_num_singlets}? {test_result_singlets_have_unique_bit_string}.'
        )
        if test_result_singlets_have_unique_bit_string is False:
            print(
                f"Encountered {len(set(VERTEX_SINGLET_BITMAPS[current_case].values()))} distinct bit strings."
            )

        assert test_result_singlets_have_unique_bit_string


def test_lattice_encoder_fails_on_bad_bitmaps():
    print(
        "Checking that a vertex bitmap with non-unique bit string values causes ValueError."
    )
    good_link_bitmap = {ONE: "00", THREE: "10", THREE_BAR: "01"}
    non_unique_vertex_bitmap = {
        ((ONE, THREE, THREE), 1): "00",
        ((ONE, THREE, THREE_BAR), 1): "00",
        ((ONE, ONE, ONE), 1): "01"
    }
    try:
        LatticeStateEncoder(good_link_bitmap, non_unique_vertex_bitmap)
    except ValueError as e:
        print(f"Test passed. Raised ValueError: {e}")
    else:
        raise AssertionError("No ValueError raised.")

    print(
        "Checking that a link bitmap with non-unique bit string values causes ValueError."
    )
    non_unique_link_bitmap = {ONE: "00", THREE: "10", THREE_BAR: "10"}
    good_vertex_bitmap = {
        ((ONE, THREE, THREE), 1): "00",
        ((ONE, THREE, THREE_BAR), 1): "01",
        ((ONE, ONE, ONE), 1): "10"
    }
    try:
        LatticeStateEncoder(non_unique_link_bitmap, good_vertex_bitmap)
    except ValueError as e:
        print(f"Test passed. Raised ValueError: {e}")
    else:
        raise AssertionError("No ValueError raised.")

    print(
        "Checking that a vertex bitmap with bit strings of different lengths causes ValueError."
    )
    vertex_bitmap_different_string_lengths = {
        ((ONE, THREE, THREE), 1): "00",
        ((ONE, THREE, THREE_BAR), 1): "001",
        ((ONE, ONE, ONE), 1): "10"
    }
    try:
        LatticeStateEncoder(good_link_bitmap,
                            vertex_bitmap_different_string_lengths)
    except ValueError as e:
        print(f"Test passed. Raised ValueError: {e}")
    else:
        raise AssertionError("No ValueError raised.")

    print(
        "Checking that a link bitmap with bit strings of different lengths causes ValueError."
    )
    link_bitmap_different_string_lengths = {
        ONE: "00",
        THREE: "10",
        THREE_BAR: "010"
    }
    try:
        LatticeStateEncoder(link_bitmap_different_string_lengths,
                            good_vertex_bitmap)
    except ValueError as e:
        print(f"Test passed. Raised ValueError: {e}")
    else:
        raise AssertionError("No ValueError raised.")


def test_encode_decode_various_links():
    link_bitmap = {ONE: "00", THREE: "10", EIGHT: "11"}
    # Test data, not physically meaningful but has right format for creating creating an encoder.
    vertex_bitmap = {
        ((ONE, ONE, ONE), 1): "00",
        ((THREE, THREE, THREE), 1): "01",
        ((THREE, THREE, THREE), 2): "10"
    }
    lattice_encoder = LatticeStateEncoder(link_bitmap, vertex_bitmap)

    print(
        "Checking that the following link_bitmap is used to correctly encode/decode links:"
    )
    print(link_bitmap)
    for link_state, bit_string_encoding in link_bitmap.items():
        result_encoding = lattice_encoder.encode_link_state_as_bit_string(
            link_state)
        result_decoding = lattice_encoder.decode_bit_string_to_link_state(
            bit_string_encoding)
        assert result_encoding == bit_string_encoding, f"(result != expected): {result_encoding} != {bit_string_encoding}"
        assert result_decoding == link_state, f"(result != expected): {result_decoding} != {link_state}"
        print(f"Test passed. Validated {bit_string_encoding} <-> {link_state}")

    print("Verifying that unknown bit string is decoded to None.")
    assert lattice_encoder.decode_bit_string_to_link_state("01") is None
    print("Test passed.")


def test_encode_decode_various_vertices():
    link_bitmap = {ONE: "00", THREE: "10", EIGHT: "11"}
    # Test data, not physically meaningful but has right format for creating creating an encoder.
    vertex_bitmap = {
        ((ONE, ONE, ONE), 1): "00",
        ((THREE, THREE, THREE), 1): "01",
        ((THREE, THREE, THREE), 2): "10"
    }
    lattice_encoder = LatticeStateEncoder(link_bitmap, vertex_bitmap)

    print(
        "Checking that the following vertex_bitmap is used to correctly encode/decode vertices:"
    )
    print(vertex_bitmap)
    for vertex_state, bit_string_encoding in vertex_bitmap.items():
        result_encoding = lattice_encoder.encode_vertex_state_as_bit_string(
            vertex_state)
        result_decoding = lattice_encoder.decode_bit_string_to_vertex_state(
            bit_string_encoding)
        assert result_encoding == bit_string_encoding, f"(result != expected): {result_encoding} != {bit_string_encoding}"
        assert result_decoding == vertex_state, f"(result != expected): {result_decoding} != {vertex_state}"
        print(
            f"Test passed. Validated {bit_string_encoding} <-> {vertex_state}")

    print("Verifying that unknown bit string is decoded to None.")
    assert lattice_encoder.decode_bit_string_to_vertex_state("11") is None
    print("Test passed.")


def test_encoding_malformed_plaquette_fails():
    v1: VertexBag = ((ONE, THREE, THREE_BAR), 1)
    v2: VertexBag = ((ONE, THREE, THREE_BAR), 1)
    v3: VertexBag = ((ONE, THREE, THREE_BAR), 1)
    v4: VertexBag = ((ONE, ONE, ONE), 1)
    l1: IrrepWeight = THREE
    l2: IrrepWeight = THREE_BAR
    l3: IrrepWeight = ONE
    l4: IrrepWeight = ONE

    lattice_encoder = LatticeStateEncoder(
        link_bitmap=IRREP_TRUNCATION_DICT_1_3_3BAR,
        vertex_bitmap=VERTEX_SINGLET_BITMAPS["d=3/2, T1"])

    print("Checking that a wrong-length plaquette input fails to be encoded.")
    plaquette_wrong_length: PlaquetteState = (v1, v2, v3, l1, l2, l3)
    try:
        lattice_encoder.encode_plaquette_state_as_bit_string(
            plaquette_wrong_length)
    except ValueError as e:
        print(f"Test passed. Raised ValueError:\n{e}")
    else:
        raise AssertionError("Test failed. No ValueError raised.")

    print("Checking that a plaquette which isn't ordered with vertices first "
          "followed by links fails.")
    plaquette_wrong_ordering = (l1, l2, l3, l4, v1, v2, v3, v4)
    try:
        lattice_encoder.encode_plaquette_state_as_bit_string(
            plaquette_wrong_ordering)
    except ValueError as e:
        print(f"Test passed. Raised ValueError:\n{e}")
    else:
        raise AssertionError("Test failed. No ValueError raised.")

    print("Checking that a plaquette which has too many vertices fails.")
    plaquette_too_many_vertices = (v1, v2, v3, v4, v4, l1, l2, l3)
    try:
        lattice_encoder.encode_plaquette_state_as_bit_string(
            plaquette_too_many_vertices)
    except ValueError as e:
        print(f"Test passed. Raise ValueError:\n{e}")
    else:
        raise AssertionError("Test failed. No ValueError raised.")


def test_all_mag_hamiltonian_plaquette_states_have_unique_bit_string_encoding(
):
    """
    Check that all plaquette states appearing in the magnetic
    Hamiltonian box term can be encoded in unique bit strings.

    Attempts encoding on the following cases:
    - d=3/2, T1
    - d=3/2, T1p
    - d=3/2, T2
    - d=2, T1
    - d=2, T1p
    - d=3, T1
    """
    cases = [
        "d=3/2, T1", "d=3/2, T1p", "d=3/2, T2", "d=2, T1", "d=2, T1p",
        "d=3, T1"
    ]
    # Each expected_bitlength = 4 * (n_link_qubits + n_vertex_qubits) for the corresponding case.
    expected_bitlength = [
        4 * (2 + 0), 4 * (2 + 0), 4 * (4 + 3), 4 * (2 + 3), 4 * (2 + 1),
        4 * (2 + 5)
    ]
    print(
        "Checking that there is a unique bit string encoding available for all "
        "the plaquette states appearing in all the matrix elements for the "
        f"following cases:\n{cases}.")
    for current_expected_bitlength, current_case in zip(
            expected_bitlength, cases):
        dim_string, trunc_string = current_case.split(",")
        dim_string = dim_string.strip()
        trunc_string = trunc_string.strip()
        # Make encoder instance.
        if trunc_string in ["T1", "T1p"]:
            link_bitmap = IRREP_TRUNCATION_DICT_1_3_3BAR
        elif trunc_string in ["T2"]:
            link_bitmap = IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8
        else:
            raise ValueError(f"Unknown irrep truncation: '{trunc_string}'.")
        # No vertices needed for T1 and d=3/2
        vertex_bitmap = {} if current_case in [
            "d=3/2, T1", "d=3/2, T1p"
        ] else VERTEX_SINGLET_BITMAPS[current_case]
        lattice_encoder = LatticeStateEncoder(link_bitmap, vertex_bitmap)

        print(
            f"Case {current_case}: confirming all initial and final states "
            "appearing in the magnetic Hamiltonian box term can be succesfully encoded.\n"
            f"Link bitmap: {link_bitmap}\n"
            f"Vertex bitmap: {vertex_bitmap}")

        # Get the set of unique plaquette states.
        all_plaquette_states = set([
            final_and_initial_state_tuple[0] for final_and_initial_state_tuple
            in HAMILTONIAN_BOX_TERMS[current_case].keys()
        ] + [
            final_and_initial_state_tuple[1] for final_and_initial_state_tuple
            in HAMILTONIAN_BOX_TERMS[current_case].keys()
        ])

        # Attempt encodings and check for uniqueness.
        all_encoded_plaquette_bit_strings = []
        for plaquette_state in all_plaquette_states:
            plaquette_state_bit_string = lattice_encoder.encode_plaquette_state_as_bit_string(
                plaquette_state)
            assert len(
                plaquette_state_bit_string
            ) == current_expected_bitlength, f"len(plaquette_state_bit_string) == {len(plaquette_state_bit_string)}; expected len == {current_expected_bitlength}."
            all_encoded_plaquette_bit_strings.append(
                plaquette_state_bit_string)
        n_unique_plaquette_encodings = len(
            set(all_encoded_plaquette_bit_strings))
        assert n_unique_plaquette_encodings == len(
            all_plaquette_states
        ), f"Encountered {n_unique_plaquette_encodings} unique bit strings encoding {len(all_plaquette_states)} unique plaquette states."

        print("Test passed.")


def test_encoding_good_plaquette():
    # Check that the encoding is as expected.
    assert VERTEX_SINGLET_BITMAPS["d=3/2, T1"] == {
        ((ONE, ONE, ONE), 1): "00",
        ((ONE, THREE, THREE_BAR), 1): "01",
        ((THREE, THREE, THREE), 1): "10",
        ((THREE_BAR, THREE_BAR, THREE_BAR), 1): "11"
    }

    # Create test data and a LatticeEncoder.
    v1: VertexBag = ((ONE, THREE, THREE_BAR), 1)
    v2: VertexBag = ((ONE, THREE, THREE_BAR), 1)
    v3: VertexBag = ((ONE, THREE, THREE_BAR), 1)
    v4: VertexBag = ((ONE, ONE, ONE), 1)
    l1: IrrepWeight = THREE
    l2: IrrepWeight = THREE_BAR
    l3: IrrepWeight = ONE
    l4: IrrepWeight = ONE
    plaquette: PlaquetteState = (v1, v2, v3, v4, l1, l2, l3, l4)
    expected_bit_string = "01010100" + "10010000"  # vertices encoding + link encoding
    lattice_encoder = LatticeStateEncoder(IRREP_TRUNCATION_DICT_1_3_3BAR,
                                          VERTEX_SINGLET_BITMAPS["d=3/2, T1"])
    print("Checking that that the following plaquette is encoded in the bit "
          f"string {expected_bit_string}:\n"
          f"{plaquette}")
    resulting_bit_string = lattice_encoder.encode_plaquette_state_as_bit_string(
        plaquette)
    assert expected_bit_string == resulting_bit_string, f"Test failed, resulting_bit_string == {resulting_bit_string}."
    print("Test passed.")


def test_bit_string_decoding_to_plaquette():
    # Check that decoding of bit strings is as expected.
    # Case data tuple format:
    # case_name, encoded_plaquette, vertex_bitmap, link_bitmap, expected_decoded_plaquette.
    # Note that the data were manually constructed by irrep encoding bitmaps
    # with data in vertex singlet json files.
    cases = [
        (
            "d=3/2, T1",
            "10101001",
            {},  # No vertex data needed  for d=3/2, T1.
            IRREP_TRUNCATION_DICT_1_3_3BAR,
            (
                None,
                None,
                None,
                None,  # When using empty vertex bitmap, should get back None for decoded vertices.
                THREE,
                THREE,
                THREE,
                THREE_BAR)),
        (
            "d=3/2, T2",
            "1001" + "1100" + "0011" + "0001" +
            "110111000001",  # Vertex strings + link string
            VERTEX_SINGLET_BITMAPS["d=3/2, T2"],
            IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8,
            (((THREE_BAR, THREE_BAR, SIX), 1), ((SIX, EIGHT, SIX_BAR), 1),
             ((ONE, EIGHT, EIGHT), 1), ((ONE, THREE, THREE_BAR),
                                        1), SIX, EIGHT, ONE, THREE_BAR)),
        (
            "d=2, T1",
            "101" + "100" + "010" + "001" +
            "10101001",  # Vertex strings + link string
            VERTEX_SINGLET_BITMAPS["d=2, T1"],
            IRREP_TRUNCATION_DICT_1_3_3BAR,
            (((THREE, THREE, THREE_BAR, THREE_BAR),
              2), ((THREE, THREE, THREE_BAR, THREE_BAR),
                   1), ((ONE, THREE, THREE, THREE),
                        1), ((ONE, ONE, THREE, THREE_BAR), 1), THREE, THREE,
             THREE, THREE_BAR))
    ]

    print(
        "Checking decoding of bit strings corresponding to gauge-invariant plaquette states."
    )

    for current_case, encoded_plaquette, vertex_bitmap, link_bitmap, expected_decoded_plaquette in cases:
        print(
            f"Checking plaquette bit string decoding for a {current_case} plaquette..."
        )
        lattice_encoder = LatticeStateEncoder(link_bitmap, vertex_bitmap)
        resulting_decoded_plaquette = lattice_encoder.decode_bit_string_to_plaquette_state(
            encoded_plaquette)
        assert resulting_decoded_plaquette == expected_decoded_plaquette, f"Expected: {expected_decoded_plaquette}\nEncountered: {resulting_decoded_plaquette}"
        print(
            f"Test passed.\n{encoded_plaquette} successfully decoded to {resulting_decoded_plaquette}."
        )


def test_decoding_non_gauge_invariant_bit_string():
    print(
        "Checking decoding of a bit string corresponding to a non-gauge-invariant plaquette state."
    )
    # Configure test data and encoder.
    lattice_encoder = LatticeStateEncoder(
        IRREP_TRUNCATION_DICT_1_3_3BAR_6_6BAR_8,
        VERTEX_SINGLET_BITMAPS["d=3/2, T2"])
    # This should still succeed since it's possible for noise to result in such states.
    # In d=3/2, T2, it is unphysical to have all ONE link states with vertex bags all
    # in EIGHTs. But physical qubit errors could causes such a measurement outcome.
    expected_plaquette_broken_gauge_invariance = (((EIGHT, EIGHT, EIGHT), 1),
                                                  ((EIGHT, EIGHT, EIGHT), 1),
                                                  ((EIGHT, EIGHT, EIGHT),
                                                   1), ((EIGHT, EIGHT, EIGHT),
                                                        1), ONE, ONE, ONE, ONE)
    encoded_plaquette = "1101" + "1101" + "1101" + "1101" + "000000000000"  # Vertex strings + link string
    decoded_plaquette = lattice_encoder.decode_bit_string_to_plaquette_state(
        encoded_plaquette)

    assert decoded_plaquette == expected_plaquette_broken_gauge_invariance, f"Expected: {expected_plaquette_broken_gauge_invariance}\nEncountered: {decoded_plaquette}"
    print(
        f"Test passed.\n{encoded_plaquette} successfully decoded to {expected_plaquette_broken_gauge_invariance}."
    )


def test_decoding_garbage_bit_strings_result_in_none():
    # This should map to something since it's possible for noise to result in such states.
    link_bitmap = {
        ONE: "000",
        THREE: "100",
        EIGHT: "101",
        SIX: "111",
        SIX_BAR: "010"
    }
    # Test data, not physically meaningful but has right format for creating creating an encoder.
    vertex_bitmap = {
        ((ONE, ONE, ONE, ONE), 1): "000",
        ((ONE, ONE, ONE, THREE), 1): "001",
        ((THREE, THREE, THREE, EIGHT), 1): "010",
        ((THREE, THREE, THREE, EIGHT), 2): "100",
        ((SIX, EIGHT, EIGHT, SIX_BAR), 1): "110"
    }
    lattice_encoder = LatticeStateEncoder(link_bitmap, vertex_bitmap)

    # Link, vertex, and plaquette test data.
    bad_encoded_link = "001"
    bad_encoded_vertex = "111"
    encoded_plaquette_some_links_and_vertices_good_others_bad = \
        "000" + "000" + bad_encoded_vertex + "110" + "000" + "000" + bad_encoded_link + "100"
    expected_decoded_plaquette_some_links_and_vertices_good_others_bad = (
        ((ONE, ONE, ONE, ONE), 1), ((ONE, ONE, ONE, ONE), 1), None,
        ((SIX, EIGHT, EIGHT, SIX_BAR), 1), ONE, ONE, None, THREE)

    print(
        f"Checking {bad_encoded_link} decodes to None using the link bitmap: {link_bitmap}"
    )
    assert lattice_encoder.decode_bit_string_to_link_state("001") is None
    print("Test passed.")

    print(
        f"Checking {bad_encoded_vertex} decodes to None using the link bitmap: {vertex_bitmap}"
    )
    assert lattice_encoder.decode_bit_string_to_vertex_state("111") is None
    print("Test passed.")

    print(
        f"Checking {encoded_plaquette_some_links_and_vertices_good_others_bad} decodes to the plaquette:\n {expected_decoded_plaquette_some_links_and_vertices_good_others_bad}"
    )
    decoded_plaquette = lattice_encoder.decode_bit_string_to_plaquette_state(
        encoded_plaquette_some_links_and_vertices_good_others_bad)
    assert decoded_plaquette == expected_decoded_plaquette_some_links_and_vertices_good_others_bad, f"(decoded != expected): {decoded_plaquette}\n!=\n{expected_decoded_plaquette_some_links_and_vertices_good_others_bad}"
    print("Test passed.")


def test_decoding_fails_when_len_bit_string_doesnt_match_bitmaps():
    # This should map to something since it's possible for noise to result in such states.
    link_bitmap = {
        ONE: "000",
        THREE: "100",
        EIGHT: "101",
        SIX: "111",
        SIX_BAR: "010"
    }
    # Test data, not physically meaningful but has right format for creating creating an encoder.
    vertex_bitmap = {
        ((ONE, ONE, ONE, ONE), 1): "0000",
        ((ONE, ONE, ONE, THREE), 1): "0001",
        ((THREE, THREE, THREE, EIGHT), 1): "0010",
        ((THREE, THREE, THREE, EIGHT), 2): "1000",
        ((SIX, EIGHT, EIGHT, SIX_BAR), 1): "1100"
    }
    lattice_encoder = LatticeStateEncoder(link_bitmap, vertex_bitmap)

    print(
        "Testing that decoding links/vertices/plaquettes fails with wrong length bit string."
    )
    print(f"Using link bitmap: {link_bitmap}")
    print(f"Using vertex bitmap: {vertex_bitmap}")

    # Set up test data.
    expected_link_bit_string_length = len(list(link_bitmap.keys())[0])
    expected_vertex_bit_string_length = len(list(vertex_bitmap.keys())[0])
    expected_plaquette_bit_string_length = 4 * (
        expected_link_bit_string_length + expected_vertex_bit_string_length)
    bad_length_link_bit_string = "10"
    bad_length_vertex_bit_string = "101"
    bad_length_plaquette_bit_string = "00000000000000010000100011101"
    assert len(bad_length_link_bit_string) != expected_link_bit_string_length
    assert len(
        bad_length_vertex_bit_string) != expected_vertex_bit_string_length
    assert len(bad_length_plaquette_bit_string
               ) != expected_plaquette_bit_string_length

    print(
        f"Checking link bit string {bad_length_link_bit_string} fails to decode."
    )
    try:
        lattice_encoder.decode_bit_string_to_link_state(
            bad_length_link_bit_string)
    except ValueError as e:
        print(f"Test passed. Raised ValueError: {e}")
    else:
        raise AssertionError("ValueError not raised.")

    print(
        f"Checking vertex bit string {bad_length_vertex_bit_string} fails to decode."
    )
    try:
        lattice_encoder.decode_bit_string_to_vertex_state(
            bad_length_vertex_bit_string)
    except ValueError as e:
        print(f"Test passed. Raised ValueError: {e}")
    else:
        raise AssertionError("ValueError not raised.")

    print(
        f"Checking plaquette bit string {bad_length_plaquette_bit_string} fails to decode."
    )
    try:
        lattice_encoder.decode_bit_string_to_plaquette_state(
            bad_length_plaquette_bit_string)
    except ValueError as e:
        print(f"Test passed. Raised ValueError: {e}")
    else:
        raise AssertionError("ValueError not raised.")
