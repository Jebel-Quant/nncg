## Makefile (repo-owned)
# Keep this file small. It can be edited without breaking template sync.

# Override template default: include mkdocstrings plugin for API docs
MKDOCS_EXTRA_PACKAGES = --with 'mkdocstrings[python]'

# Always include the Rhiza API (template-managed)
include .rhiza/rhiza.mk

# Optional machine-local overrides and extra targets (not committed)
-include local.mk
