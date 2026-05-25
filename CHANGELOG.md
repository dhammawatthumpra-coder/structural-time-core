# Changelog

All notable changes to the `structural-time-core` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - Post-v0.1.1 Maintenance

These changes are committed directly to `main` without creating a new official tag/release (maintaining v0.1.1 release optics).

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
