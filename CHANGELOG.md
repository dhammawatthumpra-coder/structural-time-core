# Changelog

All notable changes to the `structural-time-core` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.4] - 2026-06-17

### Fixed
- `BoundedTemporalDensityCalculator.compute_T_ops()`: Corrected docstring
  that incorrectly described the formula as "Bell-curve". The formula
  `exp(-(alpha*dK_dt*dist)²)` is a monotonically decreasing Gaussian
  (stability indicator), not a bell-curve. Renamed interpretation from
  "experienced temporal density" to "structural stability indicator".
- `tests/test_dynamics.py`: Fixed misleading comment "minimal experienced
  time flow / steady state" for T_ops=1.0 case. Corrected to "maximum
  structural stability, minimal structural change".

### Added  
- `BoundedTemporalDensityCalculator.compute_T_K_paper()`: New method
  implementing the exact paper formula from §2.5:
  `T(K) = alpha * dK_dt * dist * exp(-beta * dist²)` with default
  alpha=1.0, beta=0.5 per §3.1. This is a true bell-curve and matches
  the temporal density as defined in the Dynamics paper. Results from
  Item 10 (2026-06-17): Active T(K)=0.0031, Critical T(K)=0.2281,
  ratio=74.3×. See §5.6.11 of the Dynamics paper.

### Notes
- The behavior of compute_T_ops() is UNCHANGED — only docstring fixed.
  Existing users are not affected.
- The two methods (compute_T_ops and compute_T_K_paper) are intentionally
  different: T_ops is for real-time stability monitoring; compute_T_K_paper
  is for temporal density as defined theoretically.

---

## [0.1.3] - 2026-05-25

### Added
- Created `QuickStart.ipynb` notebook configured for Google Colab to enable interactive model testing without local setup.
- Added "Open in Colab" badges to English and Thai README files.

---

## [0.1.2] - 2026-05-25

These changes are published as a maintenance release addressing feedback on the core library architecture and academic framing.

### Added
- Created `examples/sociology_adapter_demo.py` containing the relocated `SociologyAdapter` with active runtime warnings and a standalone runnable demonstration block.
- Implemented active runtime warnings (`warnings.warn`) to alert users about theoretical/metaphorical boundaries in `LogicalCompatibilityChecker` and `SociologyAdapter`.
- Added unit test cases to verify the relocated `SociologyAdapter` and assert that it correctly triggers a `UserWarning` upon instantiation.

### Changed
- Relocated the conceptual/illustrative `SociologyAdapter` from the core library (`structural_time_core/adapters/sociology.py`) to `examples/sociology_adapter_demo.py` to clean up core API scope.
- Renamed `TheoryGuidedClustering` to `HybridRegimeClustering` in the `analytics` module and updated all references, examples, and tests, resolving semantic contradiction of using unsupervised KMeans for a theory-guided class.
- Updated all English and Thai documentation (`README.md`, `README_TH.md`, `docs/api.md`, `docs/examples.md`, `docs/index.md`) to reflect the package structural changes, renaming, and warnings.

---

## [0.1.1] - 2026-05-25

### Added
- Added `Epistemic Disclaimer` in both English and Thai README files detailing independent validation boundaries between Level A and Level B.
- Added `classify_by_theory` method for deterministic regime classification without fitting KMeans.
- Created `tests/test_analytics.py` cases to test the deterministic classifier.

---

## [0.1.0] - 2026-05-25

### Added
- Initial MVP release of `structural-time-core` Python library.
- Core Modules: `adapters` (Transformer, Sociology), `ontology` (StateSpaceManager, LogicalCompatibilityChecker), `dynamics` (QuarticPotentialSolver, GradientFlowIntegrator, BoundedTemporalDensityCalculator), `analytics` (TheoryGuidedClustering, VisualizationAPI).
- Unit tests coverage for all core components.
- Automated MkDocs documentation configuration and GitHub Actions deploy workflows.
