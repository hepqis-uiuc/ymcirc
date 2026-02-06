"""
Flux tube study on 4×4 periodic lattice using vertex-filtered Hamiltonian + Aer MPS.

Uses a vertex excitation constraint (max 2 excited links per vertex) to dramatically
reduce the per-plaquette Givens rotation count, making the 64-qubit simulation feasible.
"""

import numpy as np
import time
import ast
import warnings
import sys
import json
import argparse
import resource
import logging

warnings.filterwarnings('ignore')
logging.getLogger('ymcirc').setLevel(logging.WARNING)

from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

from ymcirc.conventions import (
    IRREP_TRUNCATIONS, PHYSICAL_PLAQUETTE_STATES,
    LatticeStateEncoder, load_magnetic_hamiltonian
)
from ymcirc._abstract.lattice_data import LatticeDef
from ymcirc.lattice_registers import LatticeRegisters
from ymcirc.circuit import LatticeCircuitManager

from compact_hamiltonian import diagonal_flux_tube_links


# --- Vertex excitation filter ---

_TRIVIAL = (0, 0, 0)

# Active link indices touching each vertex (l1=0, l2=1, l3=2, l4=3)
_VERTEX_ACTIVE = [[0, 3], [0, 1], [1, 2], [2, 3]]

def max_vertex_excitation(bs, encoder):
    """Max number of excited (non-trivial) links at any vertex of the plaquette."""
    _, active_links, control_links = encoder.decode_bit_string_to_plaquette_state(bs)
    n_ctrl_per_vertex = len(control_links) // 4
    worst = 0
    for vi, a_indices in enumerate(_VERTEX_ACTIVE):
        count = sum(1 for ai in a_indices if active_links[ai] != _TRIVIAL)
        c_start = vi * n_ctrl_per_vertex
        count += sum(1 for ci in range(c_start, c_start + n_ctrl_per_vertex)
                     if control_links[ci] != _TRIVIAL)
        if count > worst:
            worst = count
    return worst

def vertex_filtered_hamiltonian(encoder, max_exc=2, min_coeff=0.2):
    """Load per-plaquette Hamiltonian and filter by vertex excitation + coefficient."""
    full_ham = load_magnetic_hamiltonian(
        'd=2', 'T1', encoder, mag_hamiltonian_matrix_element_threshold=0
    )
    filtered = [
        (bs1, bs2, c) for bs1, bs2, c in full_ham
        if max(max_vertex_excitation(bs1, encoder), max_vertex_excitation(bs2, encoder)) <= max_exc
        and abs(c) >= min_coeff
    ]
    return filtered, len(full_ham)


# --- Simulation helpers ---

def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

def stamp(msg, t0):
    print(f"  [{time.time()-t0:7.1f}s] [mem={mem_mb():.0f}MB] {msg}", flush=True)

def get_link_qubit_map(circuit):
    link_to_qubits = {}
    global_idx = 0
    for reg in circuit.qregs:
        if reg.name.startswith('l:'):
            link_addr = ast.literal_eval(reg.name[2:])
            link_to_qubits[link_addr] = (global_idx, global_idx + 1)
        global_idx += reg.size
    return link_to_qubits

def apply_electric_evolution(circuit, lattice_regs, dt_e):
    c2_val = 4 / 3
    for link_addr in lattice_regs.link_addresses:
        link_reg = lattice_regs.get_link(link_addr)
        q0, q1 = link_reg[0], link_reg[1]
        phi = c2_val * dt_e
        circuit.rz(phi, q0)
        circuit.rz(phi, q1)
        circuit.rzz(-phi, q0, q1)

def classify_links(size, flux_tube_links, all_link_addresses):
    flux_set = set(tuple(l) for l in flux_tube_links)
    classification = {}
    for link_addr in all_link_addresses:
        addr = tuple(link_addr) if not isinstance(link_addr, tuple) else link_addr
        classification[addr] = 'on-path' if addr in flux_set else 'off-path'
    return classification


# --- Main simulation ---

def run_single(circ_mgr, lattice_regs, phys_states_encoded,
               flux_links, g_squared, dt, n_trotter, bond_dim, t0):
    """Run a single coupling value and return C₂ on all links."""
    circuit = circ_mgr.create_blank_full_lattice_circuit(lattice_regs)

    # Prepare flux tube initial state
    for link_addr in flux_links:
        link_reg = lattice_regs.get_link(link_addr)
        circuit.x(link_reg[0])  # rep 3 = |10>

    # Trotterized time evolution
    for step in range(n_trotter):
        dt_E = (g_squared / 2) * dt
        apply_electric_evolution(circuit, lattice_regs, dt_E)
        circ_mgr.apply_magnetic_trotter_step(
            circuit, lattice_regs,
            optimize_circuits=False,
            physical_states_for_control_pruning=phys_states_encoded,
            control_fusion=True,
            cache_mag_evol_circuit=True
        )

    stamp(f"Circuit built: {circuit.size()} gates", t0)

    # Bind parameters
    param_bindings = {
        p: np.sqrt(g_squared) if 'coupling_g' in p.name else dt
        for p in circuit.parameters
    }
    bound = circuit.assign_parameters(param_bindings)

    # Transpile
    t1 = time.time()
    transpiled = transpile(
        bound,
        basis_gates=['cx', 'u3', 'u2', 'u1', 'x', 'y', 'z',
                     'h', 'rx', 'ry', 'rz'],
        optimization_level=1
    )
    cx_count = transpiled.count_ops().get('cx', 0)
    stamp(f"Transpiled: {time.time()-t1:.1f}s, {cx_count:,} CX", t0)

    # Add ZZ expectation value measurements for each link
    link_qubit_map = get_link_qubit_map(transpiled)
    for link_addr, (q0, q1) in link_qubit_map.items():
        transpiled.save_expectation_value(
            SparsePauliOp('ZZ'), [q0, q1], label=f'zz_{link_addr}'
        )

    # Run MPS simulation
    sim = AerSimulator(
        method='matrix_product_state',
        matrix_product_state_max_bond_dimension=bond_dim
    )

    t1 = time.time()
    result = sim.run(transpiled).result()
    sim_time = time.time() - t1

    # Extract C₂ = (2/3)(1 - <ZZ>) for each link
    c2 = {}
    for link_addr in link_qubit_map:
        label = f'zz_{link_addr}'
        zz = np.real(result.data()[label])
        c2[link_addr] = (2 / 3) * (1 - zz)

    return c2, cx_count, sim_time


def run_coupling_scan(size=4, n_trotter=5, dt=0.3, bond_dim=64,
                      max_exc=2, min_coeff=0.2, g2_values=None):
    """Run coupling scan on lattice with vertex-filtered Hamiltonian + MPS."""
    if g2_values is None:
        g2_values = [8.0, 4.0, 2.0, 1.0, 0.5]

    t0 = time.time()

    lattice_def = LatticeDef(dimensions=2, size=size,
                             periodic_boundary_conds=True)
    physical_states = PHYSICAL_PLAQUETTE_STATES['d=2']['T1']
    encoder = LatticeStateEncoder(IRREP_TRUNCATIONS['T1'],
                                  physical_states, lattice_def)
    lattice_regs = LatticeRegisters.from_lattice_state_encoder(encoder)
    flux_links = diagonal_flux_tube_links(size)
    stamp(f"Lattice: {size}x{size}, {lattice_regs.n_total_qubits} qubits, "
          f"{lattice_regs.n_links} links", t0)

    # Build vertex-filtered Hamiltonian
    ham, n_full = vertex_filtered_hamiltonian(encoder, max_exc=max_exc,
                                              min_coeff=min_coeff)
    stamp(f"Hamiltonian: {len(ham)} Givens (from {n_full}, "
          f"max_exc<={max_exc}, |c|>={min_coeff})", t0)

    circ_mgr = LatticeCircuitManager(encoder, ham)
    phys_states_encoded = set(
        encoder.encode_plaquette_state_as_bit_string(s)
        for s in physical_states
    )
    circ_mgr.num_ancillas = 0

    link_class = classify_links(size, flux_links, lattice_regs.link_addresses)

    n_on = sum(1 for v in link_class.values() if v == 'on-path')
    n_off = sum(1 for v in link_class.values() if v == 'off-path')
    print(f"\n  Flux tube: {len(flux_links)} links ({flux_links})")
    print(f"  Links: {n_on} on-path, {n_off} off-path")
    print(f"  Trotter: {n_trotter} steps, dt={dt}")
    print(f"  MPS bond dimension: {bond_dim}")
    print(f"  Coupling values: {g2_values}")
    print()

    all_results = []
    for g_squared in g2_values:
        print(f"  g²={g_squared:6.2f}: ", flush=True)

        c2, cx_count, sim_time = run_single(
            circ_mgr, lattice_regs, phys_states_encoded,
            flux_links, g_squared, dt, n_trotter, bond_dim, t0
        )

        flux_c2 = sum(c2[tuple(l)] for l in flux_links)
        off_links = [l for l in c2 if tuple(l) not in
                     set(tuple(x) for x in flux_links)]
        off_c2 = sum(c2[l] for l in off_links)

        entry = {
            'g_squared': g_squared,
            'per_link_c2': {str(k): float(v) for k, v in c2.items()},
            'link_classification': {str(k): v for k, v in link_class.items()},
            'flux_c2': float(flux_c2),
            'off_c2': float(off_c2),
            'total_c2': float(flux_c2 + off_c2),
            'cx_count': cx_count,
            'bond_dim': bond_dim,
            'n_trotter': n_trotter,
            'dt': dt,
            'n_givens': len(ham),
        }
        all_results.append(entry)

        stamp(f"  Result: flux={flux_c2:.4f}  off={off_c2:.4f}  "
              f"total={flux_c2 + off_c2:.4f}  "
              f"[{cx_count:,} CX, sim={sim_time:.0f}s]", t0)
        print()
        sys.stdout.flush()

    stamp(f"All done! Total time: {time.time()-t0:.0f}s", t0)
    return all_results, flux_links, lattice_regs.link_addresses


# --- Visualization ---

def make_animation(data, flux_tube_links, all_link_addresses, size,
                   output_file='run/flux_tube_4x4_anim.gif'):
    """Create animated GIF of flux tube spreading across coupling values."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    from matplotlib.animation import FuncAnimation, PillowWriter

    flux_set = set(tuple(l) if not isinstance(l, tuple) else l
                   for l in flux_tube_links)

    # Global C₂ range
    all_c2 = []
    for entry in data:
        all_c2.extend(entry['per_link_c2'].values())
    vmax = max(all_c2) * 1.05
    vmin = 0

    cmap = cm.hot_r
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    def draw_frame(idx):
        ax.clear()
        entry = data[idx]
        g2 = entry['g_squared']
        c2_dict = entry['per_link_c2']

        for addr_str, c2_val in c2_dict.items():
            addr = ast.literal_eval(addr_str)
            vertex = addr[0]
            direction = addr[1]
            x, y = vertex

            lw = 6
            color = cmap(norm(c2_val))

            if direction == 1:  # Horizontal (+x)
                x_end = (x + 1) % size
                if x_end == 0 and x == size - 1:
                    ax.plot([x + 0.15, x + 0.5], [y, y], color=color,
                            linewidth=lw, solid_capstyle='round', zorder=2)
                    ax.plot([-0.5, -0.15], [y, y], color=color,
                            linewidth=lw, solid_capstyle='round', zorder=2,
                            linestyle='--', alpha=0.6)
                else:
                    ax.plot([x + 0.15, x + 0.85], [y, y], color=color,
                            linewidth=lw, solid_capstyle='round', zorder=2)
            else:  # Vertical (+y)
                y_end = (y + 1) % size
                if y_end == 0 and y == size - 1:
                    ax.plot([x, x], [y + 0.15, y + 0.5], color=color,
                            linewidth=lw, solid_capstyle='round', zorder=2)
                    ax.plot([x, x], [-0.5, -0.15], color=color,
                            linewidth=lw, solid_capstyle='round', zorder=2,
                            linestyle='--', alpha=0.6)
                else:
                    ax.plot([x, x], [y + 0.15, y + 0.85], color=color,
                            linewidth=lw, solid_capstyle='round', zorder=2)

            # Mark flux tube links with red dots
            is_flux = addr in flux_set or tuple(addr) in flux_set
            if is_flux:
                if direction == 1:
                    mid_x = x + 0.5 if x < size - 1 else x + 0.3
                    ax.plot(mid_x, y, 'o', color='#ff3333', markersize=4, zorder=3)
                else:
                    mid_y = y + 0.5 if y < size - 1 else y + 0.3
                    ax.plot(x, mid_y, 'o', color='#ff3333', markersize=4, zorder=3)

        # Draw vertices
        for x in range(size):
            for y in range(size):
                ax.plot(x, y, 'ko', markersize=5, zorder=4)

        flux_c2 = entry['flux_c2']
        off_c2 = entry['off_c2']
        n_flux = len(flux_tube_links)
        n_off = len(c2_dict) - n_flux

        ax.set_xlim(-0.7, size - 0.3)
        ax.set_ylim(-0.7, size - 0.3)
        ax.set_aspect('equal')
        ax.set_title(
            f'g² = {g2:.1f}    '
            f'⟨C₂⟩/link: flux={flux_c2/n_flux:.3f}  off={off_c2/n_off:.3f}',
            fontsize=13, pad=10
        )
        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        ax.grid(True, alpha=0.15)

    # Create animation
    anim = FuncAnimation(fig, draw_frame, frames=len(data),
                         interval=1500, repeat=True)

    # Add colorbar (draw once)
    draw_frame(0)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, label='⟨C₂⟩')

    fig.suptitle(
        f'Flux tube spreading on {size}×{size} torus (T1 SU(3))\n'
        f'Red dots = flux tube path',
        fontsize=14, y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    anim.save(output_file, writer=PillowWriter(fps=1))
    print(f"Saved animation to {output_file}")

    # Also save static summary plot
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    g2_vals = [e['g_squared'] for e in data]
    flux_vals = [e['flux_c2'] for e in data]
    off_vals = [e['off_c2'] for e in data]
    total_vals = [e['total_c2'] for e in data]
    n_flux = len(flux_tube_links)
    n_total = len(list(data[0]['per_link_c2'].keys()))
    n_off = n_total - n_flux

    ax1.plot(g2_vals, [f / n_flux for f in flux_vals], 'ro-',
             label=f'On-path ({n_flux} links)', markersize=8)
    ax1.plot(g2_vals, [o / n_off for o in off_vals], 'bs-',
             label=f'Off-path ({n_off} links)', markersize=8)
    ax1.set_xlabel('g²', fontsize=12)
    ax1.set_ylabel('⟨C₂⟩ per link', fontsize=12)
    ax1.set_title('Casimir per link vs coupling')
    ax1.legend()
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=4/3, color='r', linestyle='--', alpha=0.3)

    ax2.plot(g2_vals, flux_vals, 'ro-', label='On-path total', markersize=8)
    ax2.plot(g2_vals, off_vals, 'bs-', label='Off-path total', markersize=8)
    ax2.plot(g2_vals, total_vals, 'k^-', label='Total', markersize=8)
    ax2.axhline(y=n_flux * 4/3, color='r', linestyle='--', alpha=0.3)
    ax2.set_xlabel('g²', fontsize=12)
    ax2.set_ylabel('Total ⟨C₂⟩', fontsize=12)
    ax2.set_title('Total Casimir vs coupling')
    ax2.legend()
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)

    fig2.suptitle(
        f'Flux tube fuzzing: {size}×{size} T1-truncated SU(3)\n'
        f'{data[0]["n_givens"]} Givens/plaq, {data[0]["n_trotter"]} Trotter steps, '
        f'dt={data[0]["dt"]}, bd={data[0]["bond_dim"]}',
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    summary_file = output_file.replace('_anim.gif', '_summary.png')
    fig2.savefig(summary_file, dpi=150, bbox_inches='tight')
    print(f"Saved summary to {summary_file}")

    # Also save heatmap grid
    n = len(data)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig3, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1) if cols > 1 else np.array([[axes]])

    for idx, entry in enumerate(data):
        r, c = divmod(idx, cols)
        draw_frame_static(axes[r, c], entry, size, flux_set, cmap, norm)

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    sm2 = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm2.set_array([])
    fig3.colorbar(sm2, ax=axes, shrink=0.6, label='⟨C₂⟩')
    fig3.suptitle(
        f'Link Casimir ⟨C₂⟩ on {size}×{size} torus\n'
        'Red dots mark the flux tube path',
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    heatmap_file = output_file.replace('_anim.gif', '_heatmaps.png')
    fig3.savefig(heatmap_file, dpi=150, bbox_inches='tight')
    print(f"Saved heatmaps to {heatmap_file}")

    plt.close('all')


def draw_frame_static(ax, entry, size, flux_set, cmap, norm):
    """Draw a single heatmap frame on the given axes."""
    g2 = entry['g_squared']
    c2_dict = entry['per_link_c2']

    for addr_str, c2_val in c2_dict.items():
        addr = ast.literal_eval(addr_str)
        vertex = addr[0]
        direction = addr[1]
        x, y = vertex
        lw = 5
        color = cmap(norm(c2_val))

        if direction == 1:
            x_end = (x + 1) % size
            if x_end == 0 and x == size - 1:
                ax.plot([x + 0.15, x + 0.5], [y, y], color=color,
                        linewidth=lw, solid_capstyle='round', zorder=2)
            else:
                ax.plot([x + 0.15, x + 0.85], [y, y], color=color,
                        linewidth=lw, solid_capstyle='round', zorder=2)
        else:
            y_end = (y + 1) % size
            if y_end == 0 and y == size - 1:
                ax.plot([x, x], [y + 0.15, y + 0.5], color=color,
                        linewidth=lw, solid_capstyle='round', zorder=2)
            else:
                ax.plot([x, x], [y + 0.15, y + 0.85], color=color,
                        linewidth=lw, solid_capstyle='round', zorder=2)

        is_flux = addr in flux_set or tuple(addr) in flux_set
        if is_flux:
            if direction == 1:
                mid_x = x + 0.5 if x < size - 1 else x + 0.3
                ax.plot(mid_x, y, 'o', color='#ff3333', markersize=3, zorder=3)
            else:
                mid_y = y + 0.5 if y < size - 1 else y + 0.3
                ax.plot(x, mid_y, 'o', color='#ff3333', markersize=3, zorder=3)

    for x in range(size):
        for y in range(size):
            ax.plot(x, y, 'ko', markersize=4, zorder=4)

    ax.set_xlim(-0.7, size - 0.3)
    ax.set_ylim(-0.7, size - 0.3)
    ax.set_aspect('equal')
    ax.set_title(f'g² = {g2}', fontsize=10)
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.grid(True, alpha=0.15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='4×4 flux tube MPS simulation')
    parser.add_argument('--size', type=int, default=4)
    parser.add_argument('--bond-dim', type=int, default=64)
    parser.add_argument('--n-trotter', type=int, default=5)
    parser.add_argument('--dt', type=float, default=0.3)
    parser.add_argument('--max-exc', type=int, default=2)
    parser.add_argument('--min-coeff', type=float, default=0.2)
    parser.add_argument('--g2', type=float, nargs='+', default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    results, flux_links, all_links = run_coupling_scan(
        size=args.size,
        n_trotter=args.n_trotter,
        dt=args.dt,
        bond_dim=args.bond_dim,
        max_exc=args.max_exc,
        min_coeff=args.min_coeff,
        g2_values=args.g2,
    )

    if args.output:
        outbase = args.output.replace('.json', '')
    else:
        outbase = f'run/flux_scan_{args.size}x{args.size}_vtx'
    outfile = f'{outbase}.json'
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} data points to {outfile}")

    # Generate animation and plots
    make_animation(results, flux_links, all_links, args.size,
                   output_file=f'{outbase}_anim.gif')
