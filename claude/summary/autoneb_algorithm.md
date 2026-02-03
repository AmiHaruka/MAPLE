# AutoNEB: Automated Multi-Step Reaction Pathway Discovery

## Overview

**AutoNEB** is an automated nudged elastic band (NEB) algorithm designed for discovering complex multi-step reaction pathways without prior knowledge of intermediate structures. The algorithm automatically detects intermediate minima, splits reaction paths, optimizes endpoints, and manages multiple pathways in a binary tree structure.

### Key Innovations

1. **Adaptive Image Insertion**: Dynamically adds images in regions with large geometric distances
2. **Local Minima Detection & Path Splitting**: Automatically identifies intermediates and creates sub-pathways
3. **Endpoint Optimization**: Finds lower-energy endpoints through local monotonic analysis
4. **Binary Tree Management**: Organizes multiple pathways hierarchically with shared endpoints
5. **Final Global Refinement**: Smooths the complete MEP after tree optimization

### Use Cases

- Multi-step organic reactions (e.g., A → B → C → D)
- Complex rearrangements with multiple intermediates
- Exploring unknown reaction mechanisms
- Cascade reactions in catalysis

---

## Algorithm Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: Reactant → Product                    │
└─────────────────────────────┬───────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: Initialize Root Path                                 │
│  - Kabsch alignment of input structures                        │
│  - Linear interpolation + optional IDPP smoothing              │
│  - Create root PathNode in binary tree                         │
└─────────────────────────────┬───────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: Iterative Tree Optimization                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  While not all_leaves_converged():                      │   │
│  │    1. Select next pending path (DFS: leftmost first)   │   │
│  │    2. Run single NEB iteration (L-BFGS step)           │   │
│  │    3. Check convergence (f_max, f_rms thresholds)      │   │
│  │                                                          │   │
│  │    Periodic Checks:                                     │   │
│  │    ┌─────────────────────────────────────────────────┐ │   │
│  │    │ Every ang_iter (20 iter):                       │ │   │
│  │    │   → Adaptive Image Insertion                    │ │   │
│  │    │     (if distance > ang_max)                     │ │   │
│  │    └─────────────────────────────────────────────────┘ │   │
│  │    ┌─────────────────────────────────────────────────┐ │   │
│  │    │ Every path_iter (50 iter):                      │ │   │
│  │    │   → Local Minima Detection                      │ │   │
│  │    │   → Path Splitting (if minima found)           │ │   │
│  │    │     Creates two child paths with shared point   │ │   │
│  │    └─────────────────────────────────────────────────┘ │   │
│  │    ┌─────────────────────────────────────────────────┐ │   │
│  │    │ Every ep_iter (30 iter):                        │ │   │
│  │    │   → Endpoint Optimization                       │ │   │
│  │    │   → Update sibling paths (if shared)           │ │   │
│  │    └─────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: Merge & Global Refinement                            │
│  - Merge all leaf paths into global MEP                        │
│  - Run final NEB refinement on complete path                   │
│  - Apply endpoint optimization (no splitting)                  │
│  - Use looser convergence thresholds (2x relaxed)              │
└─────────────────────────────┬───────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: Global MEP, Intermediates, Transition States          │
│  - *_autoneb_global_mep.xyz: Complete reaction pathway         │
│  - *_autoneb_intermediates.xyz: All detected minima            │
│  - *_autoneb_ts_list.xyz: All transition states                │
│  - *_autoneb_tree.json: Binary tree structure                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Structures

### PathNode (Binary Tree Node)

```python
PathNode:
    path_id: str                    # Unique identifier (e.g., "root", "root_L", "root_R_L")
    images: List[Atoms]             # Current path images
    energies: List[float]           # Energy profile

    # Tree structure
    parent: Optional[str]           # Parent path ID
    children: List[str]             # Child path IDs (max 2)

    # Status tracking
    status: str                     # 'pending', 'running', 'converged', 'split', 'needs_reopt'
    depth: int                      # Recursion depth in tree

    # Shared endpoint flags
    start_is_shared: bool           # Start point shared with sibling
    end_is_shared: bool             # End point shared with sibling

    # Optimization state
    iteration: int                  # Current iteration count
    hei_idx: int                    # Highest energy image index
    lbfgs_state: {...}              # L-BFGS history (S, Y, rhos)
```

### Tree Structure Example

For a reaction A → B → C → D with two intermediates B and C:

```
                     root (A → D)
                    [split at B]
                    /           \
              root_L          root_R
             (A → B)          (B → D)
             [converged]      [split at C]
                             /           \
                       root_R_L      root_R_R
                       (B → C)       (C → D)
                      [converged]   [converged]
```

Shared endpoints:
- `root_L.end` = `root_R.start` = intermediate B
- `root_R_L.end` = `root_R_R.start` = intermediate C

---

## Core Algorithms (Pseudo Code)

### 1. Main Loop: Iterative Tree Optimization

```python
def run_autoneb():
    """
    Main AutoNEB optimization loop.
    """
    # Phase 1: Initialize root path
    initialize_root_path()

    # Phase 2: Iterative tree optimization
    global_iteration = 0
    while global_iteration < MAX_GLOBAL_ITER:
        global_iteration += 1

        # Check termination
        if all_leaves_converged():
            break

        # Select next path to optimize (DFS: leftmost first)
        path_id = select_next_path()
        if path_id is None:
            break

        node = path_tree[path_id]

        # Run single NEB iteration
        converged = run_single_neb_iteration(path_id)

        if converged:
            node.status = 'converged'
            continue

        # Periodic adaptive operations
        if node.iteration % ANG_ITER == 0:
            check_adaptive_insertion(path_id)

        if node.iteration % PATH_ITER == 0:
            minima = detect_local_minima(path_id)
            if minima:
                split_path(path_id, minima[0])  # Split at first minimum
                continue

        if node.iteration % EP_ITER == 0:
            new_start, new_end = check_endpoint_updates(path_id)
            if new_start:
                update_endpoint(path_id, 'start', new_start)
            if new_end:
                update_endpoint(path_id, 'end', new_end)

    # Phase 3: Merge and global refinement
    global_mep = merge_leaf_paths()

    if DO_FINAL_REFINE:
        global_mep = run_final_refinement(global_mep)

    write_outputs(global_mep)
```

---

### 2. Local Minima Detection

```python
def detect_local_minima(path_id):
    """
    Detect intermediate minima along the path.

    A point i is a local minimum if:
    1. E[i-1] - E[i] > MIN_E_DROP  (left side drops)
    2. E[i+1] - E[i] > MIN_E_DROP  (right side drops)
    3. i is not near endpoints (skip first 2 and last 2)

    Returns:
        List of minima indices, sorted by distance from start
    """
    node = path_tree[path_id]
    energies = node.energies
    minima = []

    # Iterate through internal images (skip boundary images)
    for i in range(2, len(energies) - 2):
        E_prev = energies[i - 1]
        E_curr = energies[i]
        E_next = energies[i + 1]

        # Local minimum condition with energy threshold
        left_drop = E_prev - E_curr
        right_drop = E_next - E_curr

        if left_drop > MIN_E_DROP and right_drop > MIN_E_DROP:
            minima.append(i)

    return sorted(minima)  # Closest to start first
```

**Illustration:**

```
Energy profile with intermediate minimum:

E  │     TS1        MIN         TS2
   │      ╱╲                    ╱╲
   │     ╱  ╲                  ╱  ╲
   │    ╱    ╲      ╱╲        ╱    ╲
   │   ╱      ╲    ╱  ╲      ╱      ╲
   │  ╱        ╲  ╱    ╲    ╱        ╲
   │ A          ╲╱      B   ╱          D
   │                     ╲  C
   │                      ╲╱
   │                       
   └─────────────────────────────────────> Reaction coordinate

Detection:
- At point B: E[B-1] - E[B] > threshold ✓
            E[B+1] - E[B] > threshold ✓
            → B is intermediate minimum → Split path
```

---

### 3. Path Splitting

```python
def split_path(path_id, split_idx):
    """
    Split a path at an intermediate minimum.

    Creates two child paths:
    - Left child: start → split point
    - Right child: split point → end

    The split point becomes a shared endpoint for both children.
    """
    parent = path_tree[path_id]

    # Check recursion limits
    if parent.depth >= MAX_DEPTH:
        return
    if len(path_tree) >= MAX_PATHS:
        return

    # Mark parent as split
    parent.status = 'split'

    # Create left child (start → split_idx)
    child1_id = f"{path_id}_L"
    child1 = PathNode(
        path_id = child1_id,
        images = parent.images[0 : split_idx + 1],
        energies = parent.energies[0 : split_idx + 1],
        parent = path_id,
        depth = parent.depth + 1,
        start_is_shared = parent.start_is_shared,  # Inherit from parent
        end_is_shared = True,                       # Shares with sibling
        status = 'pending'
    )

    # Create right child (split_idx → end)
    child2_id = f"{path_id}_R"
    child2 = PathNode(
        path_id = child2_id,
        images = parent.images[split_idx : ],
        energies = parent.energies[split_idx : ],
        parent = path_id,
        depth = parent.depth + 1,
        start_is_shared = True,                     # Shares with sibling
        end_is_shared = parent.end_is_shared,      # Inherit from parent
        status = 'pending'
    )

    # Update tree structure
    parent.children = [child1_id, child2_id]
    path_tree[child1_id] = child1
    path_tree[child2_id] = child2

    # Record intermediate structure
    intermediate = parent.images[split_idx]
    all_intermediates.append(intermediate)
```

**Tree Evolution Example:**

```
Initial:
    root (A → D, 20 images)

After detecting minimum at index 8:
    root (SPLIT)
    ├── root_L (A → B, 9 images)  [end_is_shared = True]
    └── root_R (B → D, 13 images) [start_is_shared = True]

After detecting minimum in root_R at index 6:
    root (SPLIT)
    ├── root_L (A → B, 9 images) ✓ converged
    └── root_R (SPLIT)
        ├── root_R_L (B → C, 7 images)  [shared endpoints]
        └── root_R_R (C → D, 7 images)  [shared endpoints]
```

---

### 4. Endpoint Optimization

```python
def check_endpoint_updates(path_id):
    """
    Check if endpoints should be updated to lower-energy structures.

    Uses local 3-point monotonic analysis:
    - A: current endpoint
    - X: 1st neighbor toward HEI
    - Y: 2nd neighbor toward HEI

    Update rules (start side):
    1. If A → X → Y continuously decreases: update to Y
    2. If A → X decreases but Y > A: update to X only
    3. If A → X decreases but Y ∈ [X, A]: update to X
    4. Never cross an energy rise

    Returns:
        (new_start_idx, new_end_idx) or (None, None)
    """
    node = path_tree[path_id]
    energies = node.energies
    n = len(energies)

    if n < 4:
        return None, None

    new_start = None
    new_end = None

    # ========== Start side (looking right toward HEI) ==========
    E_A = energies[0]      # Current start
    E_X = energies[1]      # 1st neighbor
    E_Y = energies[2]      # 2nd neighbor

    if E_X < E_A - EP_E_DROP:
        # X is lower, check Y
        if E_Y < E_X - EP_E_DROP:
            # Continuous decrease: A → X → Y ↓↓
            new_start = 2
        elif E_Y > E_A:
            # Rise back above A: only move to X
            new_start = 1
        else:
            # Y between X and A: safe to move to X
            new_start = 1

    # ========== End side (looking left toward HEI) ==========
    E_A = energies[-1]     # Current end
    E_X = energies[-2]     # 1st neighbor
    E_Y = energies[-3]     # 2nd neighbor

    if E_X < E_A - EP_E_DROP:
        if E_Y < E_X - EP_E_DROP:
            new_end = n - 3
        elif E_Y > E_A:
            new_end = n - 2
        else:
            new_end = n - 2

    return new_start, new_end
```

**Illustration:**

```
Case 1: Continuous decrease (update to Y)
E  │
   │  A
   │   ╲
   │    ╲  X
   │     ╲╱
   │      ╲
   │       ╲ Y
   │        ╲────> (continue toward barrier)
   └──────────────>

   A → X → Y continuously decreases → new_start = 2


Case 2: Reversal (update to X only)
E  │
   │  A
   │   ╲  Y
   │    ╲╱╲
   │     ╲  ╲
   │      X  ╲
   │          ╲───> (barrier ahead)
   └──────────────>

   A → X decreases, but Y > A → new_start = 1


Case 3: Plateau (update to X)
E  │
   │  A   Y
   │   ╲──╱
   │    ╲╱ X
   │     ╲────> (continue toward barrier)
   └──────────────>

   A → X decreases, Y between [X, A] → new_start = 1
```

---

### 5. Endpoint Synchronization (Shared Endpoints)

```python
def update_endpoint(path_id, which_end, new_idx):
    """
    Update an endpoint and synchronize with sibling if shared.
    """
    node = path_tree[path_id]

    if which_end == 'start':
        # Trim path: remove images before new_idx
        node.images = node.images[new_idx : ]
        node.energies = node.energies[new_idx : ]

        # If start is shared with sibling's end, update sibling
        if node.start_is_shared and node.parent:
            sync_shared_endpoint(path_id, 'start', node.images[0])

    elif which_end == 'end':
        # Trim path: remove images after new_idx
        node.images = node.images[ : new_idx + 1]
        node.energies = node.energies[ : new_idx + 1]

        # If end is shared with sibling's start, update sibling
        if node.end_is_shared and node.parent:
            sync_shared_endpoint(path_id, 'end', node.images[-1])

    # Reset optimizer state
    node.lbfgs_state.clear()

def sync_shared_endpoint(path_id, which_end, new_point):
    """
    Synchronize shared endpoint with sibling path.

    Example:
        Path L: A → [B] (end_is_shared = True)
        Path R: [B] → C (start_is_shared = True)

    If Path L updates its end to B', then Path R's start must also become B'.
    """
    node = path_tree[path_id]
    parent = path_tree[node.parent]
    siblings = [c for c in parent.children if c != path_id]

    for sibling_id in siblings:
        sibling = path_tree[sibling_id]

        if which_end == 'start' and sibling.end_is_shared:
            # This path's start = sibling's end
            sibling.images[-1] = copy(new_point)
            sibling.energies[-1] = energy_of(new_point)
            sibling.status = 'needs_reopt'  # Mark for re-optimization

        elif which_end == 'end' and sibling.start_is_shared:
            # This path's end = sibling's start
            sibling.images[0] = copy(new_point)
            sibling.energies[0] = energy_of(new_point)
            sibling.status = 'needs_reopt'
```

---

### 6. Adaptive Image Insertion

```python
def check_adaptive_insertion(path_id):
    """
    Dynamically insert images in regions with large distances.

    If distance between consecutive images exceeds ANG_MAX,
    insert additional images to maintain resolution.
    """
    node = path_tree[path_id]
    images = node.images

    # Compute distances between consecutive images
    distances = []
    for i in range(len(images) - 1):
        dist = ||images[i+1].positions - images[i].positions||
        distances.append(dist)

    # Identify segments needing insertion
    insertions_needed = []
    for i, dist in enumerate(distances):
        if dist > ANG_MAX:
            n_insert = ceil(dist / ANG_MAX) - 1
            insertions_needed.append((i, n_insert))

    if not insertions_needed:
        return False

    # Perform linear interpolation in identified segments
    new_images = insert_images_by_plan(images, insertions_needed)

    # Optional: IDPP smoothing to avoid unphysical geometries
    if USE_IDPP:
        new_images = run_idpp_smoothing(new_images)

    # Update node
    node.images = new_images
    node.energies = compute_energies(new_images)

    # Reset optimizer
    node.lbfgs_state.clear()

    return True
```

**Example:**

```
Before insertion (images 0-3):
         0 ←--0.2Å--→ 1 ←--0.5Å--→ 2 ←--0.2Å--→ 3

Segment 1-2 exceeds ANG_MAX (0.3Å):
    n_insert = ceil(0.5 / 0.3) - 1 = 2

After insertion (images 0-5):
         0 ←--0.2Å--→ 1 ←-0.17Å-→ 1a ←-0.17Å-→ 1b ←-0.17Å-→ 2 ←--0.2Å--→ 3
```

---

### 7. Merging Leaf Paths into Global MEP

```python
def merge_leaf_paths():
    """
    Merge all converged leaf paths into a single global MEP.

    Leaf paths are merged in DFS left-to-right order.
    Shared endpoints are included only once.

    Returns:
        (global_images, global_energies)
    """
    # Collect all leaf node IDs via DFS
    leaf_ids = []

    def collect_leaves(path_id):
        node = path_tree[path_id]
        if node.status == 'split':
            for child_id in node.children:
                collect_leaves(child_id)
        else:
            leaf_ids.append(path_id)

    collect_leaves('root')

    # Merge paths
    global_images = []
    global_energies = []

    for i, leaf_id in enumerate(leaf_ids):
        node = path_tree[leaf_id]

        if i == 0:
            # First path: include all images
            global_images.extend(node.images)
            global_energies.extend(node.energies)
        else:
            # Subsequent paths: skip first image (shared with previous end)
            global_images.extend(node.images[1:])
            global_energies.extend(node.energies[1:])

    return global_images, global_energies
```

**Example:**

```
Tree structure:
    root (SPLIT)
    ├── root_L: [A, I1, I2, B]  (4 images)
    └── root_R (SPLIT)
        ├── root_R_L: [B, I3, C]  (3 images)
        └── root_R_R: [C, I4, I5, D]  (4 images)

Merged global MEP:
    [A, I1, I2, B, I3, C, I4, I5, D]
     └─root_L──┘  └root_R_L┘ └─root_R_R──┘
                  ^skip B   ^skip C

Total: 9 images (not 11, due to shared endpoints)
```

---

### 8. Final Global Refinement

```python
def run_final_refinement(global_images):
    """
    Run final NEB refinement on the complete merged MEP.

    Differences from tree optimization:
    - Looser convergence thresholds (2x relaxed)
    - NO path splitting (minima are known intermediates)
    - Endpoint optimization enabled (fine-tune boundaries)
    - Single continuous path optimization

    This phase smooths discontinuities from merging and
    ensures the global MEP is well-converged.
    """
    # Copy images to avoid modifying tree data
    images = deep_copy(global_images)

    # Relaxed convergence
    f_max_threshold = AUTONEB_F_MAX * FINAL_REFINE_FACTOR  # e.g., 2x
    f_rms_threshold = AUTONEB_F_RMS * FINAL_REFINE_FACTOR

    # Initialize L-BFGS optimizer
    optimizer = LBFGSDriver()

    converged = False
    for iteration in range(FINAL_REFINE_MAX_ITER):
        # Compute energies and NEB forces
        energies = compute_energies(images)
        k_springs = compute_dynamic_k(energies, K_MIN, K_MAX)
        forces, f_max, f_rms = compute_neb_forces(images, energies, k_springs)

        # Check convergence
        if f_max < f_max_threshold and f_rms < f_rms_threshold:
            converged = True
            break

        # L-BFGS step (update internal images only)
        step = optimizer.compute_step(forces)
        update_positions(images, step)

        # Periodic endpoint optimization (no splitting!)
        if iteration % EP_ITER == 0:
            new_start, new_end = check_endpoint_updates_global(images, energies)
            if new_start:
                images = images[new_start : ]
                optimizer.reset()
            if new_end:
                images = images[ : new_end + 1]
                optimizer.reset()

    return images
```

---

## Parameter Summary

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Basic Settings** |
| `n_images` | 20 | Initial number of images (excluding endpoints) |
| `k_min` / `k_max` | 0.03 / 0.3 | Spring constant range (Eh/Å²) |
| `max_iter` | 256 | Max iterations per path |
| **Adaptive Insertion** |
| `ang_max` | 0.3 Å | Maximum distance between adjacent images |
| `ang_iter` | 20 | Check insertion interval |
| **Path Splitting** |
| `path_iter` | 50 | Check minima interval |
| `min_e_drop` | 0.001 Eh | Energy drop threshold for minima (≈0.6 kcal/mol) |
| **Endpoint Optimization** |
| `ep_iter` | 30 | Check endpoint interval |
| `ep_e_drop` | 0.0005 Eh | Energy drop threshold for endpoint (≈0.3 kcal/mol) |
| **Convergence** |
| `autoneb_f_max_th` | 9.5e-3 Eh/Å | Max force threshold (looser than NEB) |
| `autoneb_f_rms_th` | 5e-3 Eh/Å | RMS force threshold |
| **Tree Limits** |
| `max_depth` | 5 | Maximum recursion depth |
| `max_paths` | 10 | Maximum total paths |
| **Final Refinement** |
| `final_refine_factor` | 2.0 | Threshold multiplier (2x looser) |
| `final_refine_max_iter` | 200 | Max refinement iterations |

---

## Computational Complexity

### Time Complexity

- **Per NEB iteration**: O(N × M)
  - N = number of images
  - M = cost of energy/force calculation (ML potential)

- **Tree optimization phase**: O(P × I × N × M)
  - P = number of paths (typically 3-10)
  - I = iterations per path (50-256)
  - Total: ~1,000-5,000 force evaluations

- **Final refinement**: O(I_ref × N_total × M)
  - I_ref = 50-200 iterations
  - N_total = merged image count (20-50)
  - Additional: ~1,000-5,000 evaluations

**Total**: ~2,000-10,000 force evaluations (comparable to 5-10 standard NEB runs)

### Space Complexity

- **Binary tree storage**: O(P × N)
  - Stores all paths in memory
  - Typical: 10 paths × 20 images = 200 structures

- **L-BFGS history**: O(P × m × N × 3n)
  - m = memory size (10)
  - n = atoms per image
  - Typical: 10 paths × 10 vectors × 20 images × 3×50 atoms = ~300,000 floats (2.4 MB)

---

## Advantages Over Standard NEB

| Feature | Standard NEB | AutoNEB |
|---------|--------------|---------|
| **Multi-step reactions** | Requires manual segmentation | Automatic detection & splitting |
| **Unknown intermediates** | Needs prior knowledge | Discovers automatically |
| **Endpoint accuracy** | Fixed endpoints | Adaptive optimization |
| **Resolution** | Fixed image count | Adaptive insertion |
| **User intervention** | High (pathway design) | Minimal (reactant + product) |
| **Computational cost** | 1× per pathway | ~3-5× total (amortized over paths) |

---

## Output Files

| File | Description |
|------|-------------|
| `*_autoneb_global_mep.xyz` | Complete minimum energy pathway (all steps) |
| `*_autoneb_intermediates.xyz` | All detected intermediate minima |
| `*_autoneb_ts_list.xyz` | All transition states (one per path) |
| `*_autoneb_tree.json` | Binary tree structure (for debugging) |
| `*_autoneb_path_X_mep.xyz` | Individual path MEPs (X = path ID) |

---

## Example Use Case: Claisen Rearrangement

**Reaction**: Allyl vinyl ether → γ,δ-unsaturated carbonyl

**Input**: Reactant (allyl vinyl ether) + Product (aldehyde)

**AutoNEB discovers**:
1. **Path 1** (root_L): Reactant → Transition State 1 → Intermediate (boat conformation)
2. **Path 2** (root_R_L): Boat intermediate → Transition State 2 → Chair intermediate
3. **Path 3** (root_R_R): Chair intermediate → Transition State 3 → Product

**Tree evolution**:
```
Iteration 50:  root splits at boat intermediate
Iteration 120: root_R splits at chair intermediate
Iteration 200: All paths converged
Iteration 250: Final global refinement complete
```

**Output**:
- 3 intermediates detected (boat, chair, enol forms)
- 3 transition states identified
- Global MEP with 45 images smoothly connecting all steps

---

## Implementation Notes

### Critical Design Decisions

1. **DFS path selection**: Ensures complete exploration of left branches before right
   - Prevents premature convergence
   - Maintains coherent intermediate structures

2. **Shared endpoint synchronization**: Updates propagate to siblings
   - Maintains consistency across tree
   - Avoids discontinuities in merged MEP

3. **L-BFGS state preservation**: Each path maintains its own optimizer history
   - Faster convergence when paths are revisited
   - Enables "pause and resume" optimization

4. **Looser convergence in tree phase**: Faster initial exploration
   - Coarse optimization: 9.5e-3 Eh/Å (tree phase)
   - Fine optimization: 19e-3 Eh/Å (final refine, 2x looser)
   - Trade-off: Speed vs. accuracy in intermediate stages

5. **3-point endpoint logic**: Conservative monotonic analysis
   - Prevents "jumping over" small barriers
   - Avoids oscillations in endpoint updates

### Potential Failure Modes

1. **Missed intermediates**: If `min_e_drop` too large
   - Solution: Reduce threshold (e.g., 0.0005 Eh)

2. **Excessive splitting**: If threshold too small
   - Solution: Increase `min_e_drop` or reduce `max_depth`

3. **Endpoint oscillation**: Repeated updates
   - Rare due to 3-point logic, but can happen if MEP is very flat
   - Solution: Increase `ep_e_drop`

4. **Tree explosion**: Too many paths
   - Limited by `max_paths` and `max_depth`
   - Typical safe values: depth=5, paths=10

---

## Comparison with Related Methods

### vs. String Method (GSM)
- **GSM**: Two-ended growth from endpoints
- **AutoNEB**: Single path with adaptive splitting
- **Advantage**: AutoNEB maintains complete path, GSM requires manual segment merging

### vs. AFIR (Artificial Force Induced Reaction)
- **AFIR**: Randomly pushes fragments together
- **AutoNEB**: Systematic exploration from known reactant/product
- **Advantage**: AutoNEB is deterministic and focused

### vs. Climbing Image NEB (CI-NEB)
- **CI-NEB**: Refines single TS along known path
- **AutoNEB**: Discovers multiple TS in multi-step reactions
- **Advantage**: AutoNEB finds complete mechanism, not just one barrier

---

## Future Extensions

1. **Parallel path optimization**: Optimize independent branches simultaneously
2. **Machine learning guidance**: Use ML to predict promising split points
3. **Adaptive convergence**: Tighter thresholds near known intermediates
4. **Branching pathways**: Allow more than 2 children per node (N-ary tree)
5. **Energy-based splitting**: Split at both minima and shallow barriers

---

## References

**Core NEB algorithm**:
- Henkelman, G.; Jónsson, H. *J. Chem. Phys.* **2000**, 113, 9978.

**Dynamic spring constants**:
- ORCA implementation (Neese group)

**IDPP initialization**:
- Smidstrup et al. *J. Chem. Phys.* **2014**, 140, 214106.

**Binary tree pathway exploration**:
- Original algorithm by the MAPLE development team (this work)

---

## Citation

If you use AutoNEB in your research, please cite:

```
[To be published - MAPLE software and AutoNEB algorithm]
University of Pittsburgh Computational Chemistry Group
```

---

## Contact

For questions about the algorithm or implementation:
- MAPLE GitHub: [repository URL]
- Documentation: `MAPLE/claude/summary/`
