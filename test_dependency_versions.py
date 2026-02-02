#!/usr/bin/env python3

from pathlib import Path
import re
import sys
from typing import Optional

file_dir = Path(__file__).parent


def extract_version_from_file(filepath: Path, pattern: str) -> Optional[str]:
    with open(filepath, "r") as f:
        for line in f:
            match = re.search(
                pattern + r'\s*>?=*\s*"?([0-9]+(?:\.[^\s"\.\,]+){2})"?', line
            )
            if match:
                return match.group(1)
    raise Exception("pattern not found")


def fail(msg: str) -> None:
    print(msg)
    sys.exit(1)


print("Checking emu_base for emu-sv and emu-mps:")

# emu_sv -> emu_base
sv_dep = extract_version_from_file(file_dir / "ci/emu_sv/pyproject.toml", "emu-base")
print(f" - emu-sv depends on emu-base version {sv_dep}")

# emu_mps -> emu_base
mps_dep = extract_version_from_file(file_dir / "ci/emu_mps/pyproject.toml", "emu-base")
print(f" - emu-mps depends on emu-base version {mps_dep}")

# emu_base version
base_version = extract_version_from_file(file_dir / "emu_base/__init__.py", "__version__")
print(f" - emu-base is version {base_version}")

# emu_sv version
sv_version = extract_version_from_file(file_dir / "emu_sv/__init__.py", "__version__")
print(f" - emu-sv is version {sv_version}")

# emu_mps version
mps_version = extract_version_from_file(file_dir / "emu_mps/__init__.py", "__version__")
print(f" - emu-mps is version {mps_version}")

# citation version

citation_version = extract_version_from_file(file_dir / "CITATION.cff", "^version:")
print(f" - citation version is {citation_version}")

if (
    base_version != citation_version
    or mps_version != citation_version
    or sv_version != citation_version
):
    fail(
        f" emu_base {base_version} "
        f"or emu_sv {sv_version} "
        f"or emu_mps {mps_version} "
        f"do not match the citation version {citation_version}"
    )

if mps_dep != base_version:
    fail(f" - emu-mps dependency version {mps_dep} != emu_base {base_version}")

if sv_dep != base_version:
    fail(f" - emu-sv dependency version {sv_dep} != emu_base {base_version}")


print("Checking pulser_core version:")

# pulser_core -> emu_base
pulser_base_dep = extract_version_from_file(
    file_dir / "ci/emu_base/pyproject.toml", r"pulser-core\[torch\]"
)
print(f" - emu-base depends on pulser-core version {pulser_base_dep}")

# pulser_core in root
pulser_root_dep = extract_version_from_file(
    file_dir / "pyproject.toml", r"pulser-core\[torch\]"
)
print(f" - The root package depends on pulser-core version {pulser_root_dep}")

if pulser_root_dep != pulser_base_dep:
    fail(f"{pulser_root_dep} != {pulser_base_dep}")

# pulser_core in .pre-commit-config.yaml
pulser_pre_commit_version = extract_version_from_file(
    file_dir / ".pre-commit-config.yaml", r"pulser-core\[torch\]"
)
print(f" - pre-commit uses pulser_core version {pulser_pre_commit_version}")

if pulser_pre_commit_version != pulser_root_dep:
    fail(
        f" - pulser-core in .pre-commit version {pulser_pre_commit_version}"
        f" != pulser-core in pyproject.toml {pulser_root_dep}"
    )
print("All checks passed.")
