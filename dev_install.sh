#!/bin/bash
for pyproject in ci/*/pyproject.toml; do dir=$(dirname "$pyproject"); pip install -e "$dir"; done
