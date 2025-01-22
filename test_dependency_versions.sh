#!/bin/bash
set -e

mps_string="$(grep emu-base ci/emu_mps/pyproject.toml)"
[[ "$mps_string" =~ .*([0-9]\.[0-9]\.[0-9]).* ]]
mps_dep="${BASH_REMATCH[1]}"
echo "emu-mps depends on emu-base version $mps_dep"

base_string="$(grep version emu_base/__init__.py)"
[[ "$base_string" =~ .*([0-9]\.[0-9]\.[0-9]).* ]]
base_version="${BASH_REMATCH[1]}"
echo "emu-base is version $base_version"

if [[ "$mps_dep" != "$base_version" ]]
then
    exit 1
fi

torch_pre_commit_string="$(grep torch .pre-commit-config.yaml)"
[[ "$torch_pre_commit_string" =~ .*([0-9]\.[0-9]\.[0-9]).* ]]
torch_pre_commit_version="${BASH_REMATCH[1]}"
echo "pre-commit uses torch version $torch_pre_commit_version"

torch_pyproject_string="$(grep torch pyproject.toml)"
[[ "$torch_pyproject_string" =~ .*([0-9]\.[0-9]\.[0-9]).* ]]
torch_pyproject_version="${BASH_REMATCH[1]}"
echo "pyproject.toml uses torch version $torch_pyproject_version"

if [[ "$torch_pre_commit_version" != "$torch_pyproject_version" ]]
then
    exit 1
fi
