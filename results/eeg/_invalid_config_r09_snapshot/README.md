# Prior effective-kernel check — INVALID_CONFIG for the requested confirmation

Formal status: **INVALID_CONFIG**.

These files are a non-destructive snapshot of the earlier Gate 1 calculation performed with:

- construction: `single_kernel`;
- `K=15`;
- `r=0.9`;
- `past_mass=0.3`;
- git commit identified during classification: `6b4b4ea68fd040d29729d5a8405476e14e15fd69`;
- working-tree `results/eeg/hp_selected.json` SHA256:
  `d5f0ebcf283df39505ad9dd3985f4c5c0f1c7a16518b20f4faac64a36db64da1`.

The numerical result is valid for that r=0.9 configuration: the implementation-faithful tangent
recurrence passed its frozen local-response tolerances and the external separable ansatz
`W_K(z)R(z)` failed. It is **not** a valid confirmatory Gate 1 result for the requested frozen
r=0.7 configuration.

The generic algebraic conclusion remains valid independently of this mismatch:

`C[zI-AW_K(z)]^{-1}B` is not generically equal to `W_K(z)C[zI-A]^{-1}B`.

No source artifact was deleted or overwritten when this snapshot was created. The committed
`HEAD` version of `results/eeg/hp_selected.json` contains r=0.7, but the current working-tree file
contains r=0.9 because it has uncommitted changes. The corrected confirmatory run subsequently
read the committed r=0.7 blob without modifying that dirty file; none of the numerical arrays or
metrics in this r=0.9 snapshot were reused.

Key copied hashes:

- `theory_vs_sim_check.csv`:
  `f945985ddfad0e90b1db1753e0211782b61192b6e9a8f0c8dd55a577203c4b5a`;
- `theory_vs_sim_responses.npz`:
  `e674d9bdad94ba5b3ac30e65e70290fcb85b00d5e4d668ca9131a7ae100e7f32`.
