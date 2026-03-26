# pyclebsch: Boundary Conditions, Lattice Geometry, and Data Generation for ymcirc

A comprehensive reference document describing how pyclebsch handles lattice geometry, boundary conditions, plaquette signatures, truncation schemes, physical state construction, matrix element computation, and data generation for ymcirc.

---

## 1. Architecture Overview

pyclebsch is a Python package that computes SU(N) Clebsch-Gordan coefficients (CGCs) and lattice gauge theory Hamiltonian data (Wilson loop matrix elements). The codebase lives at:

```
pyclebsch/
  pyclebsch/
    cgc.py                            -- CGC computation engine
    su_n_operators.py                 -- SU(N) irrep algebra (dimensions, GT patterns, decompositions)
    symmetric_group/                  -- Young tableaux, symmetrizers, plethysm
    matrix_elements/
      lattice_data.py                 -- Lattice geometry, truncation, Gauss law, physical states
      plaquette_matrix_elements.py    -- Wilson loop matrix elements via site factor factorization
      helpers.py                      -- conjugate_irrep, get_irreps (Casimir-based irrep enumeration)
  run/
    gen_ymcirc_data.py                -- JSON data generation script for ymcirc
    su_n_wilson_loop.py               -- Interactive single-plaquette computation script
  tests/
    test_lattice_data.py              -- Tests for lattice geometry and plaquette signatures
```

The key computation pipeline is:
1. Define a lattice (`LatticeDef`) with sites, boundary conditions, and half-link ordering.
2. Enumerate sites, links, and plaquettes via `sites_links_and_plaquettes()`.
3. Compute allowed irreps and singlets via `irreps_and_singlets()`.
4. For each plaquette, compute physical states via `physical_plaquette_states()`.
5. For each plaquette, compute Wilson loop matrix elements via `calc_plaquette_elements()`.
6. Export data in JSON format for ymcirc consumption via `gen_ymcirc_data.py`.

---

## 2. Lattice Definitions: `LatticeDef`

**File**: `/Users/jasonelhaderi/Documents/School/UIUC/Projects/claude/pyclebsch/pyclebsch/matrix_elements/lattice_data.py`, lines 49-171

### 2.1 Core Parameters

```python
@dataclass
class LatticeDef:
    num_sites: tuple[int, int, int]       # Sites along each axis (x, y, z)
    PBCs: tuple[bool, bool, bool]         # Periodic boundary conditions per axis
    FORDER: list[LinkDirection]           # Permutation of [1, 2, 3, -1, -2, -3]
```

- **`num_sites`**: The number of lattice sites in each of the three spatial directions. Each entry must be >= 1. For periodic directions, at least 2 sites are required (a link must connect two distinct sites).
- **`PBCs`**: Per-axis boundary condition flags. `True` = periodic (torus-like), `False` = open (lattice terminates at boundary).
- **`FORDER`**: Defines the ordering of half-links at each site. A permutation of `[1, 2, 3, -1, -2, -3]` where `+d` = outgoing half-link in direction `d`, `-d` = incoming half-link from direction `d`. This ordering determines the position of each irrep in singlet tensor products and directly affects matrix element values (though the physics is invariant).

### 2.2 Lazy Properties

The `LatticeDef` class exposes three lazily-computed properties:

- **`planes`** (line 57): Which 2D planes support plaquettes. Subset of `[(1,2), (1,3), (2,3)]`.
- **`sites`** (line 71): `{coordinate: [half_link_directions]}` -- directions of half-links at each site, sorted by FORDER.
- **`links`** (line 80): `{(coordinate, direction): (start_site, end_site)}` -- all links with endpoint info.
- **`plaquettes`** (line 89): `{(coordinate, plane): (active_links, control_links, site_coords, unique_controls)}` -- full plaquette geometry.

### 2.3 Plane Existence (`_plane_exists`, line 137)

A plane `(i, j)` exists if at least one of these conditions holds:
1. Both directions have >= 2 sites (sufficient regardless of BCs).
2. One direction has >= 2 sites and the other has exactly 1 site with PBCs (the periodic direction wraps).
3. Both directions have exactly 1 site but both are periodic (both wrap).

This means a direction with only 1 site and no PBCs contributes no links in that direction, so no plaquettes can form in any plane involving that direction.

---

## 3. Boundary Conditions in Lattice Geometry

**File**: `/Users/jasonelhaderi/Documents/School/UIUC/Projects/claude/pyclebsch/pyclebsch/matrix_elements/lattice_data.py`, lines 174-301

### 3.1 How `sites_links_and_plaquettes()` Works

This function (line 174) constructs the full lattice geometry. It proceeds in three phases:

#### Phase 1: Sites (line 215)
All site coordinates are generated as Cartesian products of `range(num_sites[i])` for each axis. Every coordinate in the grid gets a site -- boundary conditions do not affect which sites exist.

#### Phase 2: Links (lines 222-235)
For each site and each direction `i` (1, 2, 3):
- **Non-periodic axis**: A link from site `s` in direction `i` exists only if `s[i] < num_sites[i] - 1`. The "last" site along a non-periodic axis has no outgoing link in that direction. This is the key line (225):
  ```python
  if s[i]==lengths[i] and not PBCs[i]:
      continue
  ```
  where `lengths[i] = num_sites[i] - 1` for non-periodic axes.
- **Periodic axis**: A link always exists. When `s[i] + 1 == lengths[i]` (i.e., `s[i] + 1 == num_sites[i]`) and PBCs are on, the link wraps: the ending site has coordinate 0 along that axis (line 229-230).

When a link is created, the starting site gets `+d` appended to its half-link list, and the ending site gets `-d` appended. This means:
- **Interior sites** (all periodic, or away from any boundary) have all 6 half-links: `[1, 2, 3, -1, -2, -3]`.
- **Boundary sites** (along non-periodic edges) have fewer half-links. A corner site of a fully open 3D lattice has only 3 half-links.

#### Phase 3: Plaquettes (lines 251-299)
For each site `s1` and each plane `(i+1, j+1)` with `i < j`:
- A plaquette is skipped if `s1` is at the boundary edge in either the `i` or `j` direction of a non-periodic axis (line 255):
  ```python
  if (s1[i]==lengths[i] and not PBCs[i]) or (s1[j]==lengths[j] and not PBCs[j]):
      continue
  ```
- Otherwise, the four corners `s1, s2, s3, s4` are computed, with wrapping applied for periodic directions (lines 258-271).
- Active links are labeled CCW: `l1=(s1,i+1)`, `l2=(s2,j+1)`, `l3=(s4,i+1)`, `l4=(s1,j+1)`.
- Control links (spectator links not part of the plaquette) are gathered per site, ordered by FORDER.
- **Unique controls** (line 296): A control link is marked as "unique" at a site only if it has not already appeared in a previous site's control list. This handles the case where periodic BCs cause two plaquette sites to share the same physical link as a control.

### 3.2 Concrete Effect of BCs on Link Counts

| Lattice | PBCs | Links per site | Comment |
|---------|------|----------------|---------|
| `[3,2,1]`, `[T,F,F]` | 1-dir periodic | 3 (boundary) to 4 (interior) | "d=3/2" cylinder |
| `[3,3,1]`, `[T,T,F]` | 2-dir periodic | 4 at every site | "d=2" torus |
| `[3,3,3]`, `[T,T,T]` | 3-dir periodic | 6 at every site | "d=3" torus |
| `[2,2,1]`, `[F,F,F]` | None | 2 (corner) to 3 (edge) | Single plaquette |
| `[4,4,1]`, `[F,F,F]` | None | 2-4 links per site | Large 2D OBC |
| `[2,2,2]`, `[F,F,F]` | None | 3 at every site | Single cube OBC |

---

## 4. Plaquette Signatures

**File**: `/Users/jasonelhaderi/Documents/School/UIUC/Projects/claude/pyclebsch/pyclebsch/matrix_elements/lattice_data.py`, lines 561-593

### 4.1 What a Signature Is

```python
PlaquetteSignature = (plane, (half_links_s1, half_links_s2, half_links_s3, half_links_s4))
```

A plaquette signature captures the **local topology** around a plaquette: which plane it lies in, and what half-links exist at each of its four corner sites (sorted by FORDER). Two plaquettes with the same signature have identical local structure and therefore identical matrix element computation.

The function `compute_plaquette_signature()` (line 561):
1. Looks up the plaquette in `lattice.plaquettes`.
2. If the plaquette exists, reads the four site coordinates and gathers each site's half-links, sorted by FORDER.
3. If the plaquette does NOT exist (boundary of non-periodic lattice), returns `(plane, ())` with an empty tuple.

### 4.2 Physical Meaning

- **Interior plaquette**: All four sites have the maximum number of half-links for the lattice dimension. All interior plaquettes on the same plane share the same signature.
- **Edge plaquette**: One or more sites sit on a boundary edge, having fewer half-links. This changes the number of control links at those sites.
- **Corner plaquette**: Sites at the intersection of two boundary edges have even fewer half-links.

The signature determines:
- How many control (spectator) links exist at each site.
- Which singlet tensor products are possible at each site.
- Which boundary condition constraints apply when glueing site factors.

### 4.3 Signature Counts for Various Lattice/BC Combinations

**Fully periodic lattices**: All plaquettes on a given plane have the same signature (all sites are topologically equivalent). For a d-dimensional periodic lattice, there is exactly **1 signature per plane**.

**Fully open (OBC) lattices**: Different plaquettes have different signatures depending on whether their sites are corners, edges, or interior. The number of distinct signatures depends on lattice dimension:

| Lattice config | PBCs | Planes | Distinct signatures | Notes |
|---------------|------|--------|-------------------|-------|
| `[N,M,1]` PBC | `[T,T,F]` | (1,2) | **1** | All sites identical (torus) |
| `[N,M,1]` PBC | `[T,F,F]` | (1,2) | **1** | Cylinder; all plaquette sites have same link count |
| `[N,M,1]` OBC | `[F,F,F]` | (1,2) | **up to 9** | Corner/edge/interior for each of 4 sites; for a large enough lattice (>= 4x4) all 9 types appear |
| `[N,M,1]` mixed | `[F,T,F]` | (1,2) | **up to 3** | Periodic in one plaquette direction reduces variation |
| `[N,M,L]` PBC | `[T,T,T]` | 3 planes | **3** (1 per plane) | All plaquettes identical within each plane |
| `[N,M,L]` OBC | `[F,F,F]` | 3 planes | **up to 54** (up to 18 per plane) | Many combinations of corner/edge/interior across 4 sites in 3D |
| `[2,2,2]` OBC | `[F,F,F]` | 3 planes | **6** | Each face of the cube has a unique signature |

This is confirmed by the test data in `/Users/jasonelhaderi/Documents/School/UIUC/Projects/claude/pyclebsch/tests/test_lattice_data.py`:

- **d=3/2, PBCs `[T,F,F]`, 2x2 lattice** (lines 99-119): Both plaquettes `((0,0,0),(1,2))` and `((1,0,0),(1,2))` have the **same** signature: `((1,2), ((1,-1,2),(1,-1,2),(1,-1,-2),(1,-1,-2)))`. This is because the periodic direction makes s1/s2 and s3/s4 topologically identical.

- **d=3/2, OBCs `[F,F,F]`, 4x2 lattice** (lines 121-148): Three plaquettes have **three distinct** signatures. The leftmost plaquette has s1 with only 2 half-links (corner), the middle plaquette is "interior" with all 4 sites having 3 links, and the rightmost has s2/s3 with only 2 half-links (far-side corners).

- **d=2, PBCs `[T,T,F]`, 2x2 lattice** (lines 151-177): All 4 plaquettes have the **same** signature with 4 half-links per site.

- **d=2, mixed BCs `[F,T,F]`, 4x3 lattice** (lines 179-228): 9 plaquettes with **3 distinct** signatures. Along the periodic y-direction all plaquettes in the same column are identical. Along the open x-direction, the leftmost column (s1/s4 missing `-1` half-link), center column (all 4 half-links), and rightmost column (s2/s3 missing `+1` half-link but having `-1`) differ.

- **d=3, OBC, 2x2x2 lattice** (lines 232-281): 6 plaquettes (cube faces) with **6 distinct** signatures (one per face), each in a different plane with different half-link configurations.

- **d=3, mixed BCs `[F,T,T]`, 4x3x3 lattice** (lines 283-311): Spot-checked signatures show "edge" plaquettes (5 half-links at boundary sites in the open direction) vs "interior" plaquettes (6 half-links at all sites) vs "face" plaquettes (all sites in periodic-only planes having 5 half-links uniformly).

### 4.4 Key Insight: Periodic vs Non-Periodic Signatures

For **periodic lattices**, translational invariance means every plaquette in a given plane sees the same local environment. Therefore, one set of matrix element data per plane suffices for the entire lattice. This is why the PBC cases in `gen_ymcirc_data.py` only specify `site_coords_for_comp: [(0, 0, 0)]`.

For **non-periodic (or mixed BC) lattices**, translational invariance is broken at the boundaries. Different plaquettes can have different signatures, meaning different sets of physical states and matrix elements. The data generation script must compute data for each distinct signature separately. This is why OBC cases specify multiple site coordinates:
```python
"site_coords_for_comp": [tuple(t) for t in product(range(N), repeat=d)]
```

---

## 5. Truncation Schemes

**File**: `/Users/jasonelhaderi/Documents/School/UIUC/Projects/claude/pyclebsch/pyclebsch/matrix_elements/lattice_data.py`, lines 303-375

### 5.1 T-Truncation (`truncation_mode='T'`)

Bounds the first component of the i-weight tuple. For SU(3) with `cutoff=1`, the allowed irreps are:
- `(0,0,0)` -- trivial
- `(1,0,0)` -- fundamental (3)
- `(1,1,0)` -- antifundamental (3-bar)

Implementation (line 325):
```python
trial_irreps = {R: calc_casimir(R) for R in sorted(r[::-1]+(0,) for r in combinations_with_replacement(range(cutoff+1), N-1))}
```

### 5.2 C-Truncation (`truncation_mode='C'`)

Bounds the quadratic Casimir of each individual link irrep. Uses `get_irreps()` from `helpers.py` (line 16) which iteratively increases T and checks Casimir values until no more irreps qualify.

### 5.3 B-Truncation (`truncation_mode='B'`)

Bounds the **sum** of quadratic Casimirs of all irreps meeting at each site. More restrictive than C-truncation because different sites (with different numbers of links due to boundary conditions) may support different irrep combinations.

Key logic (line 351):
```python
B_valid = sum_of_casimirs < cutoff or np.isclose(sum_of_casimirs, cutoff)
if truncation_mode in ['T','C'] or (truncation_mode=='B' and B_valid):
```

### 5.4 `irreps_and_singlets()` Output

Returns three objects:
1. **`link_irreps[n]`**: Sorted list of allowed irreps at sites with `n` half-links.
2. **`site_singlets[n]`**: `{control_link_irreps: {plaquette_link_irreps: multiplicity}}` -- all ways to form singlets, organized by number of links at a site.
3. **`conj_dict[R]`**: `{state_index: (conjugate_state_index, phase)}` -- maps basis states of R to conjugate states in R-bar.

The key point: `singlets` is indexed by `n` (number of half-links at a site), and boundary sites have fewer links than interior sites. This means **boundary and interior sites draw from different singlet dictionaries**, which is one mechanism by which BCs affect the set of physical states.

---

## 6. Physical Plaquette States

**File**: `/Users/jasonelhaderi/Documents/School/UIUC/Projects/claude/pyclebsch/pyclebsch/matrix_elements/lattice_data.py`, lines 377-559

### 6.1 State Format (pyclebsch internal, line 20)

A `PlaquetteState` is a 12-tuple:
```
(l1, l2, l3, l4, (s1_ctrls), (s2_ctrls), (s3_ctrls), (s4_ctrls), g1, g2, g3, g4)
```
- `l1-l4`: Active link i-weights (CCW from bottom-left)
- `(sN_ctrls)`: Tuples of control (spectator) link i-weights at each vertex, FORDER-ordered
- `g1-g4`: Singlet multiplicity indices (0-indexed), one per vertex

### 6.2 How `physical_plaquette_states()` Works

The function builds states site-by-site, progressively filtering:

1. **Compute the signature** (line 386): Determines the half-links at each of the 4 plaquette sites.

2. **Compute FORDER indices** (lines 405-423): For each site, determines which positions in the FORDER-sorted half-link list correspond to the plaquette's active links (i and j directions) and which correspond to control links in the i or j directions.

3. **Determine boundary condition constraints** (lines 432-435):
   ```python
   s2_has_BCs = (1 in i_ctrl_idxs) and (2 in i_ctrl_idxs) and (i_ctrl_idxs[2][1] not in unique_ctrls[1])
   s3_has_BCs = (2 in j_ctrl_idxs) and (3 in j_ctrl_idxs) and (j_ctrl_idxs[3][1] not in unique_ctrls[2])
   enforce_i_ctrls = (3 in i_ctrl_idxs) and (4 in i_ctrl_idxs) and (i_ctrl_idxs[4][1] not in unique_ctrls[3])
   enforce_j_ctrls = (1 in j_ctrl_idxs) and (4 in j_ctrl_idxs) and (j_ctrl_idxs[4][1] not in unique_ctrls[3])
   ```
   These booleans detect when two adjacent plaquette sites share a control link (physical link that is not an active plaquette link but appears at both sites). This happens when PBCs cause site wrapping, making two "different" plaquette sites actually adjacent in a way that they share non-active links.

4. **Site-by-site construction**:
   - **s1** (lines 451-465): Iterate over all singlets for sites with `num_links` half-links. For each singlet, conjugate outgoing half-link irreps (Gauss law convention: outgoing links are conjugated in the singlet).
   - **s2** (lines 468-490): Match s2's i-direction active link to s1's i-direction active link (`site[i_idxs[1]] == s[i_idxs[0]]`). If `s2_has_BCs`, also enforce that the shared i-direction control link has the same irrep.
   - **s3** (lines 493-515): Match s3's j-direction active link to s2's j-direction active link. If `s3_has_BCs`, enforce shared j-direction control link matching.
   - **s4** (lines 518-543): Match s4's i-direction active link to s3's i-direction active link AND s4's j-direction active link to s1's j-direction active link (closing the plaquette loop). Enforce `enforce_i_ctrls` and `enforce_j_ctrls` constraints if applicable.

5. **Enumerate multiplicity indices** (lines 547-557): For each valid combination of site singlets, loop over all combinations of multiplicity indices `(g1, g2, g3, g4)`.

### 6.3 How Boundary Conditions Affect Physical States

The set of physical states depends on BCs through multiple mechanisms:

1. **Different numbers of half-links at boundary sites** => different singlet dictionaries (`singlets[n]` for different `n`) => different possible irrep combinations.

2. **Shared control links** => additional constraints (the `*_has_BCs` and `enforce_*_ctrls` booleans) that reduce the number of valid state combinations.

3. **Fewer control links at boundary sites** => smaller `(sN_ctrls)` tuples in the state representation => fewer degrees of freedom.

For a fully periodic 2D lattice (e.g., `[3,3,1]` with `[T,T,F]`), every site has 4 half-links, and the singlet dictionary `singlets[4]` is used uniformly. For a single OBC plaquette (`[2,2,1]` with `[F,F,F]`), corner sites have only 2 half-links with no control links at all, so `singlets[2]` is used and the control-link tuples in the state are empty.

---

## 7. Matrix Element Computation

**File**: `/Users/jasonelhaderi/Documents/School/UIUC/Projects/claude/pyclebsch/pyclebsch/matrix_elements/plaquette_matrix_elements.py`

### 7.1 Site Factor Decomposition

The Wilson loop matrix element `<Pf|U_plaquette|Pi>` factorizes across the four plaquette vertices (line 14):

```
<Pf|U|Pi> = sqrt(d1 * d3) * SF(s1) * SF(s2) * SF(s3) * SF(s4)
```

where `d_k = dim(Ri_initial) * dim(Rj_initial) / (dim(Ri_final) * dim(Rj_final))` at the corresponding vertex.

### 7.2 `plaquette_site_factor()` (line 14)

Computes a single site factor. Each site's contribution involves CGCs from:
- **Top row**: Initial singlet -> trivial (Gi-th multiplicity)
- **Bottom row**: Final singlet -> trivial (Gf-th multiplicity)
- **Left column**: Ri_initial x UR_i -> Ri_final
- **Right column**: Rj_initial x UR_j -> Rj_final

The UR irreps (fundamental or antifundamental) and conjugation rules differ by site:

| Site | UR_i | UR_j | Ri conjugated? | Rj conjugated? |
|------|------|------|----------------|----------------|
| s1 | fund | afund | yes | yes |
| s2 | fund | fund | no | yes |
| s3 | afund | fund | no | no |
| s4 | afund | afund | yes | no |

### 7.3 `calc_plaquette_site_factors()` (line 163)

Orchestrates site factor computation for all four sites of a plaquette. The signature (line 176) determines the half-link structure at each site, which affects:
- How many links meet at each site (`num_links_at_site`)
- Which singlets are available (`singlets[num_links_at_site]`)
- Which irrep decompositions are pre-computed (`decomp_dict`)

Uses 5-worker multiprocessing by default.

### 7.4 `glue_plaquette_site_factors()` (line 293)

Combines four site factors into a full matrix element. Boundary condition constraints are applied during glueing (lines 421-425):

```python
s2_has_BCs = (1 in i_ctrl_idxs) and (2 in i_ctrl_idxs) and (i_ctrl_idxs[2] not in unique_ctrls[1])
s3_has_BCs = (2 in j_ctrl_idxs) and (3 in j_ctrl_idxs) and (j_ctrl_idxs[3] not in unique_ctrls[2])
enforce_i_ctrls = (3 in i_ctrl_idxs) and (4 in i_ctrl_idxs) and (i_ctrl_idxs[4] not in unique_ctrls[3])
enforce_j_ctrls = (1 in j_ctrl_idxs) and (4 in j_ctrl_idxs) and (j_ctrl_idxs[4] not in unique_ctrls[3])
```

These are identical to the BC checks in `physical_plaquette_states()` and ensure that shared control links (due to PBCs) have matching irreps when glueing site factors from adjacent sites.

### 7.5 `calc_plaquette_elements()` (line 376)

Top-level function that:
1. Computes the plaquette signature.
2. Determines BC constraints from unique controls.
3. Computes all site factors via `calc_plaquette_site_factors()`.
4. Organizes site factors into lookup structures for efficient glueing.
5. Glues site factors in parallel (each s1 seed is independent).

Returns `{(Pf_12tuple, Pi_12tuple): matrix_element_value}`.

### 7.6 How BCs Affect Matrix Elements

1. **Different signatures => different site factors**: Boundary sites have fewer half-links, leading to different singlet structures and thus different CGC products in the site factor computation.

2. **BC constraints during glueing**: When sites share control links, the glueing process must enforce irrep matching, which restricts which site factor combinations are valid.

3. **Different plaquettes may yield different matrix element sets**: On a non-periodic lattice, edge/corner plaquettes produce a different set of matrix elements than interior plaquettes.

---

## 8. Data Generation for ymcirc

**File**: `/Users/jasonelhaderi/Documents/School/UIUC/Projects/claude/pyclebsch/run/gen_ymcirc_data.py`

### 8.1 Format Conversion

The ymcirc format reorders the pyclebsch internal state format (lines 50-62):

```python
# pyclebsch: (l1, l2, l3, l4, c1, c2, c3, c4, g1, g2, g3, g4)
# ymcirc:    ((g1, g2, g3, g4), (l1, l2, l3, l4), (c1, c2, c3, c4))
```

### 8.2 Lattice Cases Defined

The script defines four lists of lattice cases (lines 84-339):

#### PBC T-truncation cases (`lattice_cases_T_truncations_PBC`, line 84)
All commented out. Include:
- `[3,2,1]` with `[T,F,F]` (d=3/2 cylinder) at T=1 and T=2
- `[3,3,1]` with `[T,T,F]` (d=2 torus) at T=1
- `[3,3,3]` with `[T,T,T]` (d=3 torus) at T=1

All specify `site_coords_for_comp: [(0, 0, 0)]` because periodic lattices have a single signature per plane.

#### Non-PBC T-truncation cases (`lattice_cases_T_trunctions_non_PBC`, line 130)
Active (uncommented). Include:
- `[2,2,1]` OBC (d=2, single plaquette) at T=1
- `[4,4,1]` OBC (d=2, large) at T=1
- `[2,2,2]` OBC (d=3, single cube) at T=1
- `[4,4,4]` OBC (d=3, large) at T=1

These specify `site_coords_for_comp` as all coordinate tuples (e.g., `product(range(4), repeat=2)` for 2D) to cover all possible signatures.

#### PBC B-truncation cases (`lattice_cases_B_truncations_PBC`, line 176)
All commented out. Various B-cutoff values for d=3/2, d=2, and d=3 periodic lattices.

#### Non-PBC B-truncation cases (`lattice_cases_B_truncations_non_PBC`, line 328)
One active case:
- `[3,4,1]` with `[T,F,F]` (d=2 mixed BCs) at B=3, with `site_coords_for_comp: [tuple(t) for t in product(range(2), repeat=2)]`

### 8.3 Data Generation Pipeline (lines 346-475)

For each lattice case:

1. **Setup**: Create `LatticeDef`, compute sites/links/plaquettes and truncated irreps/singlets.

2. **Plaquette states** (lines 375-398):
   - For each plane and each site coordinate in `site_coords_for_comp`:
     - Skip if plaquette address doesn't exist (boundary).
     - Compute `physical_plaquette_states()`.
     - Convert to ymcirc format (stringified tuples).
   - Deduplicate across all planes/sites: `list(set(plaq_states))`.

3. **Matrix elements** (lines 401-450):
   - For each plane and each site coordinate:
     - Skip non-existent plaquette addresses.
     - Compute `compute_plaquette_signature()` to get `plaquette_site_half_links`.
     - Compute `calc_plaquette_elements()`.
     - Store in nested dict: `{str((Pf,Pi)): {str(plane): {str(site_half_links): value}}}`.

4. **Validation** (lines 457-463): Assert that all states appearing in matrix element keys are a subset of the physical plaquette states.

5. **File output**: Two gzip-compressed JSON files per lattice case.

### 8.4 Output File Formats

**Plaquette states file** (`*_plaquette_states.json.gz`):
```json
{
  "metadata": {
    "dim": "d=3/2",
    "truncation_mode": "T",
    "num_sites": [3, 2, 1],
    "PBCs": [true, false, false],
    "cutoff": 1,
    "planes": ["(1, 2)"],
    "site_coords_for_comp": [[0, 0, 0]],
    "f_order": [1, 2, 3, -1, -2, -3]
  },
  "data": [
    "((0, 0, 0, 0), ((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), ((), (), (), ()))",
    ...
  ]
}
```

**Matrix elements file** (`*_magnetic_hamiltonian.json.gz`):
```json
{
  "metadata": { ... },
  "data": {
    "((Pf_ymcirc_fmt), (Pi_ymcirc_fmt))": {
      "(1, 2)": {
        "((half_links_s1, ...), ...)": float_value
      }
    }
  }
}
```

Key hierarchy: `str(Pf, Pi)` -> `str(plane)` -> `str(site_half_links)` -> float value.

The `site_half_links` key within the plane dict is precisely the **signature's second element** -- the tuple of half-link directions per site. This is how ymcirc identifies which signature a given matrix element belongs to. For periodic lattices, there is only one such key per plane; for non-periodic lattices, there may be multiple.

---

## 9. Summary: How Signature Multiplicity Scales

### Fully Periodic Lattices
- **1 signature per plane**, regardless of lattice size.
- A single computation at the origin suffices.
- PBC examples: `[N,M,1]` with `[T,T,F]`; `[N,M,L]` with `[T,T,T]`.

### Non-Periodic Lattices: 2D Case (`d=2`, single plane `(1,2)`)

For a sufficiently large 2D OBC lattice (`[N,M,1]` with `[F,F,F]`, N>=4, M>=4):

Each plaquette site can be one of three types along each non-periodic axis:
- **Left/bottom boundary**: Missing the `-d` half-link for direction d.
- **Interior**: Has both `+d` and `-d` half-links.
- **Right/top boundary**: Missing the `+d` half-link for direction d.

Since a plaquette's four sites occupy a 2x2 block, and each site's position relative to the boundary matters independently in each direction, the number of distinct signatures is at most **3 x 3 = 9** for a single plane (3 positions for the i-direction edge x 3 for the j-direction edge). Specifically:
1. Corner (s1 at (0,0)): s1 has 2 links, s2 has 3, s3 has 4, s4 has 3
2. Bottom edge: s1 has 3, s2 has 3, s3 has 4, s4 has 4
3. Bottom-right corner: s1 has 3, s2 has 2, s3 has 3, s4 has 4
4. Left edge: s1 has 3, s2 has 4, s3 has 4, s4 has 3
5. Interior: all sites have 4 links
6. Right edge: s1 has 4, s2 has 3, s3 has 3, s4 has 4
7. Top-left corner: s1 has 3, s2 has 4, s3 has 3, s4 has 2
8. Top edge: s1 has 4, s2 has 4, s3 has 3, s4 has 3
9. Top-right corner: s1 has 4, s2 has 3, s3 has 2, s4 has 3

For **mixed BCs** (one direction periodic, one open), the periodic direction eliminates variation, leaving at most **3 distinct signatures** per plane (left edge, interior, right edge in the open direction).

### Non-Periodic Lattices: 3D Case (`d=3`, three planes)

For a fully OBC 3D lattice (`[N,M,L]` with `[F,F,F]`), each plane can have up to 9 distinct signatures (by the same 2D argument within that plane), and the third (out-of-plane) direction adds another factor since out-of-plane control links vary. Sites can have 3 to 6 half-links. The maximum number of distinct signatures per plane is **up to 18** (9 in-plane positions x 2 for whether the out-of-plane direction is at the boundary or interior), giving a theoretical maximum of **54** across all three planes.

In practice, for small lattices like `[2,2,2]` OBC, there are exactly 6 plaquettes with 6 distinct signatures (each face of the cube is unique).

### Implications for ymcirc

- **Periodic lattices**: ymcirc needs one set of plaquette data per plane. Matrix elements can be reused for every plaquette in that plane.
- **Non-periodic lattices**: ymcirc needs separate plaquette state and matrix element data for each distinct signature. The `gen_ymcirc_data.py` script handles this by iterating over all site coordinates and recording matrix elements keyed by `(plane, site_half_links)`. ymcirc must look up the correct matrix element data based on each plaquette's signature.

---

## 10. File Reference Index

| Topic | File | Lines |
|-------|------|-------|
| `LatticeDef` class | `pyclebsch/matrix_elements/lattice_data.py` | 49-171 |
| `sites_links_and_plaquettes()` | `pyclebsch/matrix_elements/lattice_data.py` | 174-301 |
| Link creation (PBC wrapping) | `pyclebsch/matrix_elements/lattice_data.py` | 222-235 |
| Plaquette creation (BC skip) | `pyclebsch/matrix_elements/lattice_data.py` | 251-299 |
| `irreps_and_singlets()` | `pyclebsch/matrix_elements/lattice_data.py` | 303-375 |
| `physical_plaquette_states()` | `pyclebsch/matrix_elements/lattice_data.py` | 377-559 |
| BC constraint booleans | `pyclebsch/matrix_elements/lattice_data.py` | 432-435 |
| `compute_plaquette_signature()` | `pyclebsch/matrix_elements/lattice_data.py` | 561-593 |
| `PlaquetteSignature` type | `pyclebsch/matrix_elements/lattice_data.py` | 44-46 |
| `plaquette_site_factor()` | `pyclebsch/matrix_elements/plaquette_matrix_elements.py` | 14-161 |
| `calc_plaquette_site_factors()` | `pyclebsch/matrix_elements/plaquette_matrix_elements.py` | 163-291 |
| `glue_plaquette_site_factors()` | `pyclebsch/matrix_elements/plaquette_matrix_elements.py` | 293-374 |
| `calc_plaquette_elements()` | `pyclebsch/matrix_elements/plaquette_matrix_elements.py` | 376-502 |
| BC checks in glueing | `pyclebsch/matrix_elements/plaquette_matrix_elements.py` | 421-426 |
| `conjugate_irrep()` | `pyclebsch/matrix_elements/helpers.py` | 9-14 |
| `get_irreps()` | `pyclebsch/matrix_elements/helpers.py` | 16-56 |
| `gen_ymcirc_data.py` main script | `run/gen_ymcirc_data.py` | 65-475 |
| Format conversion function | `run/gen_ymcirc_data.py` | 50-62 |
| PBC lattice cases | `run/gen_ymcirc_data.py` | 84-129 |
| Non-PBC T-truncation cases | `run/gen_ymcirc_data.py` | 130-175 |
| Non-PBC B-truncation cases | `run/gen_ymcirc_data.py` | 328-339 |
| Plaquette signature tests | `tests/test_lattice_data.py` | 92-329 |

All file paths are relative to `/Users/jasonelhaderi/Documents/School/UIUC/Projects/claude/pyclebsch/`.
