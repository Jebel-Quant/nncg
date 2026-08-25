# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and entries are generated from [Conventional Commits](https://www.conventionalcommits.org).

## [0.5.1] - 2026-08-25

### New Features
- *(inner)* Add GlobalNystrom — sketch A once, mask per free block (#62)
- MPRGP solver + active-set comparison notebook (#63)

### Bug Fixes
- *(certificate)* Report +0.0 for a certified optimum on every platform (#91)

### Documentation
- Record grouped test layout as accepted Rhiza deviation (#59) (#61)

### Maintenance
- *(tests)* Consolidate suite into tests/test_nncg mirroring src modules (#37)
- *(paper)* Pin paper claims not covered by the existing suite (#38)
- *(solver)* Split _run pivot logic into helpers (#39) (#40)
- Chore(deps)(deps): bump the github-actions group with 6 updates (#41)
- Update rhiza to v1.1.0 (#45)
- *(solver,inner)* Reduce _run and _nystrom complexity (#46) (#48)
- *(solver)* Assert the max_outer contract instead of a magic count (#47) (#49)
- *(inner)* Reduce Exact to A complexity (#43) (#50)
- Update rhiza to v1.1.2 (#51)
- *(inner,krylov)* Extract helpers to drop B-ranked blocks to grade A (#52) (#53)
- Update rhiza to v1.1.3 (#54)
- *(solver,inner)* Extract cohesive modules to lift maintainability (#55) (#56)
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 3 updates (#57)
- Update rhiza to v1.2.1 (#58)
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 2 updates (#65)
- Chore(deps)(deps): bump the github-actions group with 13 updates (#64)
- *(pyproject)* Modernize Python version and license metadata (#66)
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 2 updates (#68)
- Chore(deps)(deps): bump docker/login-action in the github-actions group (#67)
- Update rhiza to v1.2.5 (#69)
- Update rhiza to v1.3.0 (#75)
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 2 updates (#80)
- Chore(deps)(deps): bump the github-actions group with 3 updates (#79)
- Update rhiza to v1.3.2 (#82)
- *(ci)* Bump the rhiza pin to v1.3.3 (#84)
- Update rhiza to v1.3.3 (#85)
- Chore(deps)(deps): bump the github-actions group with 2 updates (#83)
- *(mprgp)* Extract the three moves, and document the solvers by example (#88)
- *(mprgp)* Extract the three moves, and document the solvers by example (#89)
- Update rhiza to v1.3.4 (#90)
- Update rhiza to v1.4.2 and migrate the make layer to rhiza-task (#92)
- Delete the excluded rhiza stubs instead of freezing them (#94)
- Update rhiza to v1.5.0 (#95)
- Prune exclude entries the template no longer ships (#96)
- Sync the legal bundle, keeping this repo's LICENSE (#97)
- Drop the exclude entries for the retired mutation/fuzzing workflows (#98)
- Update rhiza to v1.6.0 (#99)
- Add .zenodo.json for the Zenodo GitHub release archive (#101)

### Other Changes
- Show the CodeFactor grade in the README (#70)
- Document the grouped test layout and add public-API doctests (#74)
- Restore a discoverable bumpversion config in pyproject.toml (#76)
- Render the full public API in docs/api.md (#78)
- Drop the redundant lint dependency group (#81)
- Delete .github/ISSUE_TEMPLATE directory
- Delete .github/DISCUSSION_TEMPLATE directory (#93)

## [0.5.0] - 2026-07-06

### New Features
- Add solve_nnqp / solve_nnqp_eq one-call convenience wrappers (#34)

### Documentation
- Fix stale inner-solver references in solver.py docstrings (#36)

### Other Changes
- Bump version 0.4.2 → 0.5.0

## [0.4.2] - 2026-07-06

### Bug Fixes
- Estimate rcond_free once per free set on the exact path (#30) (#31)
- *(exact)* Make the rcond_free conditioning guard opt-out (#32) (#33)

### Other Changes
- Bump version 0.4.0 → 0.4.2

## [0.4.0] - 2026-07-06

### New Features
- Add precond.py with Jacobi and randomized Nyström preconditioners (#27)

### Maintenance
- Layer the API as ActiveSetSolver + pluggable InnerSolver (#29)

### Other Changes
- Bump version 0.3.2 → 0.4.0

## [0.3.2] - 2026-07-05

### Bug Fixes
- Return exact warm start in zero iterations for CG/PCG (#26)

### Other Changes
- Bump version 0.3.1 → 0.3.2

## [0.3.1] - 2026-07-05

### Bug Fixes
- Honor warm start across all inner solvers (#24) (#25)

### Other Changes
- Bump version 0.3.0 → 0.3.1

## [0.3.0] - 2026-07-05

### Maintenance
- Remove baseline comparison suite (maintained in mean_variance_solvers) (#23)

### Other Changes
- Bump version 0.2.2 → 0.3.0

## [0.2.2] - 2026-07-04

### New Features
- Type solve_nnqp inner as Literal and reject unknown solvers
- Support inner="pcg"/"exact" in solve_nnqp_eq
- Alternative solvers as benchmark baselines (osqp, clarabel, lawson-hanson, duchi)
- Add FISTA baseline, share accelerated proximal-gradient core with Duchi

### Bug Fixes
- Address rcond recomputation, PCG test flakiness, eq API symmetry
- Restore indentation mangled by automated PR-fix commits

### Documentation
- Add marimo notebooks to the companion book
- Inline planted-optimum generators in notebook 03, add matplotlib dev dep
- Explain in krylov.py why CG/PCG stay in-house vs scipy (#22)

### Other Changes
- Potential fix for pull request finding
- Potential fix for pull request finding
- Potential fix for pull request finding
- Merge pull request #15 from Jebel-Quant/docs/marimo-notebooks
- Merge pull request #16 from Jebel-Quant/inner-solver-literal
- Merge pull request #17 from Jebel-Quant/eq-inner-solver
- Potential fix for pull request finding
- Potential fix for pull request finding
- Potential fix for pull request finding
- Potential fix for pull request finding
- Potential fix for pull request finding
- Potential fix for pull request finding
- Merge pull request #21 from Jebel-Quant/feat/alternative-solvers
- Bump version 0.2.1 → 0.2.2

## [0.2.1] - 2026-07-03

### Bug Fixes
- Satisfy fmt/docs/typecheck gates for hoisted free-set matvec

### Performance
- Hoist the free-set restriction out of the inner CG loop

### Maintenance
- Bump cvx-linalg to >=0.9.6

### Other Changes
- Expose warm-starting in solve_nnqp_eq
- Merge pull request #11 from Jebel-Quant/eq-warm-start
- Merge pull request #14 from Jebel-Quant/bump-cvx-linalg-0.9.6
- Merge pull request #13 from Jebel-Quant/perf/hoist-free-set-restriction
- Bump version 0.2.0 → 0.2.1

## [0.2.0] - 2026-07-03

### Other Changes
- Initial release: nncg 0.1.0
- Adopt cvx-linalg as a runtime dependency and fix mypy strict errors
- Bring test coverage to 100%
- Fix lowest-direct CI failure and make validate
- Fix CI: typed matvec closure for mypy, scale-aware eq-test thresholds
- Delete .github/workflows/release.yml
- Merge pull request #4 from Jebel-Quant/tschm-patch-1
- Merge branch 'main' into cvx-linalg-dependency
- Deduplicate _matvec after merging main
- Merge pull request #3 from Jebel-Quant/cvx-linalg-dependency
- Fix stale mkdocs.yml and add the mkdocstrings API reference
- Merge pull request #5 from Jebel-Quant/fix-mkdocs
- Take the quadratic term as a cvx.linalg SymmetricOperator only
- Read the Jacobi preconditioner off op.diag (cvx-linalg 0.9.5)
- Merge pull request #6 from Jebel-Quant/operator-input
- Validate operator dimension and guard exact inner against singular free blocks
- Move the planted-problem generators out of the package into tests/
- Potential fix for pull request finding
- Potential fix for pull request finding
- Potential fix for pull request finding
- Potential fix for pull request finding
- Potential fix for pull request finding
- Potential fix for pull request finding
- Make tests a package so the tests.problems imports resolve
- Sort tests.problems imports into the first-party block
- Merge pull request #7 from Jebel-Quant/input-validation
- Share the active-set driver between the two solvers
- Drop cg and pcg from the package root
- Merge pull request #9 from Jebel-Quant/active-set-refactor
- Fill out the README badge row
- Merge pull request #10 from Jebel-Quant/readme-badges
- Bump version 0.1.0 → 0.2.0

<!-- generated by git-cliff -->
