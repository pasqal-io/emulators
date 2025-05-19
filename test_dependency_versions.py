#!/usr/bin/env python3

import re
import sys
from typing import Optional


def extract_version_from_file(filepath: str, pattern: str) -> Optional[str]:
    with open(filepath, "r") as f:
        for line in f:
            if re.search(pattern, line):
                match = re.search(r"([0-9]+(?:\.[0-9*]+){2})", line)
                if match:
                    return match.group()
    return None


def fail(msg: str) -> None:
    print(msg)
    sys.exit(1)


print("Checking emu_base for emu-sv and emu-mps:")

# emu_sv -> emu_base
sv_dep = extract_version_from_file("ci/emu_sv/pyproject.toml", "emu-base")
print(f" - emu-sv depends on emu-base version {sv_dep}")

# emu_mps -> emu_base
mps_dep = extract_version_from_file("ci/emu_mps/pyproject.toml", "emu-base")
print(f" - emu-mps depends on emu-base version {mps_dep}")

# emu_base version
base_version = extract_version_from_file("emu_base/__init__.py", "version")
print(f" - emu-base is version {base_version}")

# emu_sv version
sv_version = extract_version_from_file("emu_sv/__init__.py", "version")
print(f" - emu-sv is version {sv_version}")

# emu_mps version
mps_version = extract_version_from_file("emu_mps/__init__.py", "version")
print(f" - emu-mps is version {mps_version}")

# citation version

citation_version = extract_version_from_file("CITATION.cff", "^version")
print(f" - citation version is {citation_version}")

if (
    base_version != citation_version
    or mps_version != citation_version
    or sv_version != citation_version
):
    fail(
        f" emu_base {base_version}"
        "or emu_sv {sv_version}"
        "or emu_mps {mps_version}"
        "do not match the citation version {citation_version}"
    )

if mps_dep != base_version:
    fail(f" - emu-mps dependency version {mps_dep} != emu_base {base_version}")

if sv_dep != base_version:
    fail(f" - emu-sv dependency version {sv_dep} != emu_base {base_version}")


print("Checking pulser_core version:")

# pulser_core -> emu_base
pulser_base_dep = extract_version_from_file("ci/emu_base/pyproject.toml", "pulser-core")
print(f" - emu-base depends on pulser-core version {pulser_base_dep}")

# pulser_core in root
pulser_root_dep = extract_version_from_file("pyproject.toml", "pulser-core")
print(f" - The root package depends on pulser-core version {pulser_root_dep}")

if pulser_root_dep != pulser_base_dep:
    fail(f"{pulser_root_dep} != {pulser_base_dep}")


print("Checking torch:")

# torch in .pre-commit-config.yaml
torch_pre_commit_version = extract_version_from_file(".pre-commit-config.yaml", "torch")
print(f" - pre-commit uses torch version {torch_pre_commit_version}")

# torch in pyproject.toml
torch_pyproject_version = extract_version_from_file("pyproject.toml", "torch")
print(f" - pyproject.toml uses torch version {torch_pyproject_version}")

if torch_pre_commit_version != torch_pyproject_version:
    fail(
        f" - torch in .pre-commit version {torch_pre_commit_version}"
        f" != torch in pyproject.toml {torch_pyproject_version}"
    )

print("All checks passed.")
