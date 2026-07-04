# MGOpt-TLC

**Nonlinear Multigrid for Total Lifted Content (TLC) Minimization**

This repository contains the implementation of **MGOpt-TLC**, a nonlinear multigrid (MG/Opt) framework for minimizing the *Total Lifted Content (TLC)* energy to compute injective, flip-free mappings of simplicial meshes.

---

## Requirements

- MATLAB (tested on R2024a)
- A C/C++ compiler supported by MATLAB (`mex`)
- No external dependencies

---

## Build

Before running any experiments, compile the MEX routines:

build\_mex.m

---

## Run

Run the main script in MATLAB:

main.m

---

## Input Format

The solver expects an **OBJ file** (`input.obj`) containing:

- **Vertices (`v`)**: 3D coordinates
- **Faces (`f`)**: Triangle topology
- **Texture Coordinates (`vt`)**:
  - Store the initial guess via Tutte flattening

---

## Output

- Optimized injective mapping
- Final mesh: `<filePath>/output.obj`



\- Optimized injective mapping

\- Final mesh: `<filePath>/output.obj`






