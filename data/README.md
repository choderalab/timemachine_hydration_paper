# Data/Analysis

The following serves to navigate the data generation/analysis processes described in the manuscript.

## `espaloma-0.3.2` (unoptimized) Absolute Hydration Free Energy Calculations (AHFEs)
The original (unoptimized) `espaloma-0.3.2` AHFE calculations can be run by executing `bsub < run_freesolv.orig.sh` on a [SLURM](https://slurm.schedmd.com) workload manager.
The `TMPDIR` path will need to be modified to an appropriate directory on your device. 
It is also recommended to make a directory titled `freesolv_esp` in which the executable can be run to ensure consistency of directory paths in the subsequent analysis.
The execution will generate a series of `.pkl` files containing data on the AHFE calculation for each ligand in the `FreeSolv` dataset. 

The `.pkl` files aggregated (and saved to `agg_freesolv_data.pkl`), analyzed, and visualized with `freesolv_postprocess.ipynb`.

## ESS Calibration and Comparison of ESS-regularized vs w/o ESS-regularized Refitting 
The reweighting/refitting procedure uses ESS regularization, whose threshold is calibrated against a bootstrapping procedure of the refit `espaloma-0.3.2` (`agg_freesolv_data.pkl`) data.
The data are loaded, optimized, analyzed, and plotted in `freesolv_refit.ESS_calibration_and_figs.ipynb`.

The comparison of absolute residual CDFs of ESS-regularized refitting vs without is shown in `nESS_vs_nonESS_comparison.ipynb`. Access to the following step's refitting experiment `.pkl` is required; hence, proceed to next step before executing analysis.

## Fine-tuning via Zwanzig Reweighting and ESS Regularization
Fine-tuning with Zwanzig reweighting and ESS Regularization is executed via `bsub < run_freesolv_refit.sh` on a SLURM workload manager. It calls `env_refitting_aux.freesolv.py` with different combinations of train/validate/test splits and rank r with a total of 48 independent refitting experiments. Each experiment generates a `.pkl` containing refitting data and metadata. 

Again, the `TMPDIR` path will need to be modified to an appropriate directory on your device.

The corresponding analysis of all the experiments is performed and visualized in `env_refitting_aux.freesolv_analysis.ipynb`

## Optimized/Resimulated FreeSolv AHFE Calculations and Comparison Analysis
The charge-optimized AHFE calculations can be run by executing `bsub < run_freesolv.opt.sh` on a [SLURM](https://slurm.schedmd.com) workload manager.
The `TMPDIR` path will need to be modified to an appropriate directory on your device consistent with the original calculations.
It is recommended to make a directory titled `freesolv_esp_ixn` in which the executable can be run to ensure consistency of directory paths in the subsequent analysis.
Also, the `agg_freesolv_filepath` pointing to the `agg_freesolv_data.pkl` in the `run_freesolv.opt.py` must be set to the location wherein it was generated. 
The executable will generate a series of `.pkl` files containing data on the AHFE calculation for each ligand in the `FreeSolv` dataset. 

The correlation plot of the Optimized/Resimulated and Optimized/Reweighted AHFEs, as well as the Optimized/Resimulated and Unoptimized/Simulated correlation with experimental data are shown in `freesolv_ixn_round2.ipynb`.


