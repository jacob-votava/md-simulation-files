# Atomistic IEG Substituent Paper

Simulation inputs, restart files, force field optimization data, and analysis scripts for the atomistic IEG substituent study.

## Contents

- **`simulation_preparation/`** -- `.mdp` templates, Slurm helpers, parameterization scripts, and OpenFF force field files used to initialize GROMACS simulations.
- **`simulations_restart_files/`** -- Restart snapshots (`trem_last_run.{tpr,gro}`) and topologies for every temperature-replica-exchange simulation (monomer and dimer phase behavior).
- **`ff_optimization/`** -- OpenFF BespokeFit torsion-drive scan results (B3LYP-D3BJ/DZVP), input SDF files, and QM/MM comparison plots for reproducing the DFT calculations.
- **`sample_analysis/`** -- Analysis scripts used to generate the figures in the paper and supporting information (see below).

## Analysis Scripts (`sample_analysis/`)

Each script is a self-contained Python file that reads raw simulation output and reproduces the corresponding figure.

#### Prerequisites

1. **Simulation data (user-provided).** The scripts read raw GROMACS output (`.tpr`, `.xtc`, `.edr`, `.log`) that you must supply. The expected directory layout under `SIMULATION_ROOT` is:
   ```
   SIMULATION_ROOT/
   ├── N_monomer/simulations/tREM/{T}K/trem_gpu.{tpr,xtc,edr,log}
   ├── M_monomer/simulations/tREM/{T}K/trem_gpu.{tpr,xtc,edr,log}
   └── dimers/{PM,MP,P2,M2}/simulations/tREM/{T}K/trem_gpu.{tpr,xtc,edr,log}
   ```
   where `{T}` is the temperature in Kelvin (e.g. `300K`, `400K`, …).

2. **QM data (included).** The OpenFF torsion-scan data in `ff_optimization/` is used by the DFT/parity scripts. Point `QM_DATA_ROOT` to that directory.

#### Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Create the locked environment (all dependencies are pinned in `uv.lock`):
   ```bash
   cd sample_analysis
   uv sync
   ```

#### Running scripts

Set the following environment variables to point to your data, then run any script with `uv run`:

```bash
export SIMULATION_ROOT="/path/to/simulation/root"          # GROMACS trajectories (see layout above)
export DATA_ROOT="/path/to/working/directory"               # working dir for cached intermediate results
export QM_DATA_ROOT="/path/to/openff/qm/data"              # OpenFF QM torsion-scan data
export FRAGMENT_RG_DATA_ROOT="/path/to/fragment/rg/cache"   # working dir for fragment Rg cache

uv run python polished_figures/monomer_phase_diagram.py
uv run python supp_info/all_phi_hists_300K.py
# etc.
```

Each script is self-contained and can be run independently. Output figures are written to `polished_figures/` and `supp_info/`.

A convenience Slurm script (`remake_figures.slurm`) is also provided to run all 15 scripts sequentially — edit the path variables at the top of the file and submit with `sbatch remake_figures.slurm`.

### Main figures (`polished_figures/`)

| Script | Description |
|--------|-------------|
| `monomer_phase_diagram.py` | Nematic order parameter S(T) and density vs temperature for P1/M1 monomers (Maier-Saupe TNI fit, block-averaged density from EDR files) |
| `dimer_phase_diagram.py` | Same analysis for four dimer systems (PM, MP, P2, M2) |
| `dft_torsion_panel.py` | QM vs MM torsion energy scans (initial and optimized force field) |
| `parity_panel.py` | QM-MM energy parity scatter (before/after optimization, Pearson R) |
| `rg_figure.py` | Radius of gyration distributions at six temperatures + variance vs T |
| `dimer_conformational_analysis.py` | 2D hexbin of Rg vs bend angle from MD trajectories |
| `dimer_bend_rg_joyplots.py` | Temperature-dependent ridgeline plots of Rg and bend angle |
| `fragment_rg_and_angle_analysis.py` | Fragment-resolved Rg distributions and inter-fragment angle analysis |
| `ml_train_std_only.py` | XGBoost Rg regressor from circular-std dihedral features (N=256 clustering) with SHAP importance |

### Supporting information (`supp_info/`)

| Script | Description |
|--------|-------------|
| `phi_dist_vs_temp.py` | Temperature-dependent ridgeline plots for phi_1 and phi_6 |
| `all_phi_hists_300K.py` | 15-panel dihedral histograms at 300 K (phi_0, phi_2-phi_16) |
| `ml_training_metrics.py` | Violin plots of circular-std features + ML model parity |
| `mc_acceptance_table.py` | Parse replica-exchange acceptance probabilities from GROMACS logs |
| `order_parameter_timeseries.py` | S(t) time series for convergence verification |
| `fragment_dft_plots.py` | Fragment torsion energy scans with RDKit structure insets |

### Shared styling (`figure_params/`)

`figure_params.py` defines colors, fonts, markers, and axis-formatting helpers shared across all scripts.
