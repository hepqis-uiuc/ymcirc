def binary_to_gray(binary_str):
    """Convert a binary string to its Gray code representation."""
    gray = []
    gray.append(binary_str[0])  # The first bit is the same
    for i in range(1, len(binary_str)):
        # XOR the current bit with the previous bit
        gray_bit = str(int(binary_str[i]) ^ int(binary_str[i - 1]))
        gray.append(gray_bit)
    return ''.join(gray)

def gray_order_bitstrings(dict):
    """Sort a dictionary in Gray code order without generating the full sequence."""
    # Compute the Gray code for each bitstring and sort based on it
    sorted_bitstrings = dict(sorted(dict.items(), key=lambda x: binary_to_gray(x[0])))
    return sorted_bitstrings

def bitstrings_differ_in_one_bit(bitstring1,bitstring2):
    if len(bitstring1) != len(bitstring2):
        raise AttributeError("The length of the two bitstrings must be equal")
    else:
        diff_bit_count = 0
        for i in range(len(bitstring1)):
            if bitstring1[i] != bitstring2[i]:
                diff_bit_count += 1
                first_bit_of_diff = i
        if diff_bit_count ==1 :
            return True,first_bit_of_diff
        else:
            return False,None

