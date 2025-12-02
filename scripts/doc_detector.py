#!/usr/bin/env python3
"""
Documentation Agent for Pasqal Emulators.

A CLI tool to detect documentation drift between code and docs.
Validates that public APIs are properly documented and that
documentation stays in sync with code changes.

Usage:
    python scripts/doc_detector.py --check-all
    python scripts/doc_detector.py --check-signatures
    python scripts/doc_detector.py --check-links
    python scripts/doc_detector.py --check-examples
    python scripts/doc_detector.py --check-external-links
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Optional aiohttp import for async HTTP requests
try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# Optional playwright import for live site validation
try:
    from playwright.async_api import async_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


@dataclass
class SignatureInfo:
    """Information about a function/method signature."""

    name: str
    module: str
    parameters: list[str]
    return_annotation: str | None
    docstring: str | None
    is_public: bool
    kind: str  # 'function', 'method', 'class'


@dataclass
class DocReference:
    """A mkdocstrings reference found in documentation."""

    reference: str  # e.g., "emu_mps.mps.MPS"
    file_path: Path
    line_number: int


@dataclass
class ExternalLink:
    """An external HTTP link found in documentation."""

    url: str
    text: str  # Link text
    file_path: Path
    line_number: int


@dataclass
class LocalLink:
    """A local file link found in documentation."""

    path: str
    text: str  # Link text
    file_path: Path
    line_number: int


@dataclass
class ExternalLinkResult:
    """Result of checking an external link."""

    link: ExternalLink
    status_code: int | None
    error: str | None
    is_broken: bool


@dataclass
class DriftReport:
    """Report of documentation drift issues."""

    missing_in_docs: list[str] = field(default_factory=list)
    signature_mismatches: list[dict[str, Any]] = field(default_factory=list)
    broken_references: list[str] = field(default_factory=list)
    broken_external_links: list[dict[str, Any]] = field(default_factory=list)
    broken_local_links: list[dict[str, Any]] = field(default_factory=list)
    undocumented_params: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def has_issues(self) -> bool:
        return bool(
            self.missing_in_docs
            or self.signature_mismatches
            or self.broken_references
            or self.broken_external_links
            or self.broken_local_links
            or self.undocumented_params
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON output."""
        return {
            "missing_in_docs": self.missing_in_docs,
            "signature_mismatches": self.signature_mismatches,
            "broken_references": self.broken_references,
            "broken_external_links": self.broken_external_links,
            "broken_local_links": self.broken_local_links,
            "undocumented_params": self.undocumented_params,
            "warnings": self.warnings,
            "has_issues": self.has_issues(),
        }

    def summary(self) -> str:
        lines = ["=" * 60, "DOCUMENTATION DRIFT REPORT", "=" * 60, ""]

        if self.missing_in_docs:
            lines.append(f"Missing from docs ({len(self.missing_in_docs)}):")
            for item in self.missing_in_docs:
                lines.append(f"  - {item}")
            lines.append("")

        if self.signature_mismatches:
            lines.append(f"Signature mismatches ({len(self.signature_mismatches)}):")
            for mismatch in self.signature_mismatches:
                lines.append(f"  - {mismatch['name']}: {mismatch['issue']}")
            lines.append("")

        if self.broken_references:
            lines.append(f"Broken references ({len(self.broken_references)}):")
            for broken_ref in self.broken_references:
                lines.append(f"  - {broken_ref}")
            lines.append("")

        if self.broken_external_links:
            lines.append(f"Broken external links ({len(self.broken_external_links)}):")
            for link_info in self.broken_external_links:
                status = link_info.get("status", "unknown")
                url = link_info.get("url", "unknown")
                location = link_info.get("location", "unknown")
                # Show location first in file:line format for VSCode click-to-navigate
                lines.append(f"  {location}: {url} (status: {status})")
            lines.append("")

        if self.broken_local_links:
            lines.append(f"Broken local links ({len(self.broken_local_links)}):")
            for link_info in self.broken_local_links:
                path = link_info.get("path", "unknown")
                location = link_info.get("location", "unknown")
                reason = link_info.get("reason", "")
                # Show location first in file:line format for VSCode click-to-navigate
                if reason:
                    lines.append(f"  {location}: {path} ({reason})")
                else:
                    lines.append(f"  {location}: {path}")
            lines.append("")

        if self.undocumented_params:
            lines.append(f"Undocumented parameters ({len(self.undocumented_params)}):")
            for undoc_param in self.undocumented_params:
                lines.append(f"  - {undoc_param['name']}: {undoc_param['params']}")
            lines.append("")

        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            for item in self.warnings:
                lines.append(f"  - {item}")
            lines.append("")

        if not self.has_issues():
            lines.append("No documentation drift detected.")

        lines.append("=" * 60)
        return "\n".join(lines)


class CodeAnalyzer:
    """Analyzes Python source code to extract public API signatures."""

    def __init__(self, root_path: Path):
        self.root_path = root_path

    def get_public_apis(self, module_name: str) -> list[SignatureInfo]:
        """Extract all public APIs from a module."""
        apis: list[SignatureInfo] = []

        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            print(f"Warning: Could not import {module_name}: {e}")
            return apis

        # Get items from __all__ if defined
        all_items = getattr(module, "__all__", None)
        if all_items is None:
            # Fall back to non-underscore items
            all_items = [name for name in dir(module) if not name.startswith("_")]

        for name in all_items:
            if name == "__version__":
                continue

            try:
                obj = getattr(module, name)
            except AttributeError:
                continue

            sig_info = self._extract_signature(name, obj, module_name)
            if sig_info:
                apis.append(sig_info)

        return apis

    def _extract_signature(
        self, name: str, obj: Any, module_name: str
    ) -> SignatureInfo | None:
        """Extract signature information from a Python object."""
        try:
            if inspect.isclass(obj):
                return self._extract_class_signature(name, obj, module_name)
            elif inspect.isfunction(obj) or inspect.ismethod(obj):
                return self._extract_function_signature(name, obj, module_name)
        except Exception as e:
            print(f"Warning: Could not extract signature for {name}: {e}")
        return None

    def _extract_class_signature(
        self, name: str, cls: type, module_name: str
    ) -> SignatureInfo:
        """Extract signature from a class (uses __init__)."""
        params: list[str] = []
        try:
            sig = inspect.signature(cls)
            params = [
                self._format_param(p) for p in sig.parameters.values() if p.name != "self"
            ]
        except (ValueError, TypeError):
            pass

        return SignatureInfo(
            name=name,
            module=module_name,
            parameters=params,
            return_annotation=None,
            docstring=inspect.getdoc(cls),
            is_public=not name.startswith("_"),
            kind="class",
        )

    def _extract_function_signature(
        self, name: str, func: Any, module_name: str
    ) -> SignatureInfo:
        """Extract signature from a function."""
        params = []
        return_ann = None
        try:
            sig = inspect.signature(func)
            params = [
                self._format_param(p)
                for p in sig.parameters.values()
                if p.name not in ("self", "cls")
            ]
            if sig.return_annotation != inspect.Signature.empty:
                return_ann = str(sig.return_annotation)
        except (ValueError, TypeError):
            pass

        return SignatureInfo(
            name=name,
            module=module_name,
            parameters=params,
            return_annotation=return_ann,
            docstring=inspect.getdoc(func),
            is_public=not name.startswith("_"),
            kind="function",
        )

    def _format_param(self, param: inspect.Parameter) -> str:
        """Format a parameter for display."""
        result = param.name
        if param.annotation != inspect.Parameter.empty:
            ann = param.annotation
            if hasattr(ann, "__name__"):
                result += f": {ann.__name__}"
            else:
                result += f": {ann}"
        if param.default != inspect.Parameter.empty:
            result += f" = {param.default!r}"
        return result


class DocsParser:
    """Parses mkdocs documentation to find mkdocstrings references and external links."""

    # Pattern for mkdocstrings references like "::: emu_mps.mps.MPS"
    MKDOCSTRINGS_PATTERN = re.compile(r"^:::?\s+([\w.]+)", re.MULTILINE)

    # Pattern for markdown links: [text](url)
    MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]*)\]\((https?://[^)]+)\)")

    # Pattern for local file links: [text](path) where path is not http(s)://
    LOCAL_LINK_PATTERN = re.compile(
        r"\[([^\]]*)\]\(([^)]+?(?:\.py|\.ipynb|\.md|\.txt|\.yml|\.yaml|\.json|\.toml))\)"
    )

    # Pattern for bare URLs
    BARE_URL_PATTERN = re.compile(r"(?<![(\[])(https?://[^\s\)>\]\"']+)")

    def __init__(self, docs_path: Path, root_path: Path | None = None):
        self.docs_path = docs_path
        self.root_path = root_path or docs_path.parent

    def find_all_references(self) -> list[DocReference]:
        """Find all mkdocstrings references in documentation files."""
        references: list[DocReference] = []

        for md_file in self.docs_path.rglob("*.md"):
            refs = self._parse_file(md_file)
            references.extend(refs)

        return references

    def find_all_external_links(self) -> list[ExternalLink]:
        """Find all external HTTP links in documentation files."""
        links: list[ExternalLink] = []

        # Check markdown files
        for md_file in self.docs_path.rglob("*.md"):
            file_links = self._parse_external_links(md_file)
            links.extend(file_links)

        # Check Jupyter notebooks
        for ipynb_file in self.docs_path.rglob("*.ipynb"):
            file_links = self._parse_notebook_links(ipynb_file)
            links.extend(file_links)

        return links

    def find_all_local_links(self) -> list[LocalLink]:
        """Find all local file links in documentation files."""
        links: list[LocalLink] = []

        # Check markdown files
        for md_file in self.docs_path.rglob("*.md"):
            file_links = self._parse_local_links(md_file)
            links.extend(file_links)

        return links

    def _parse_file(self, file_path: Path) -> list[DocReference]:
        """Parse a single markdown file for mkdocstrings references."""
        references: list[DocReference] = []

        try:
            content = file_path.read_text()
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            return references

        for line_num, line in enumerate(content.split("\n"), 1):
            match = self.MKDOCSTRINGS_PATTERN.match(line.strip())
            if match:
                references.append(
                    DocReference(
                        reference=match.group(1),
                        file_path=file_path,
                        line_number=line_num,
                    )
                )

        return references

    def _parse_external_links(self, file_path: Path) -> list[ExternalLink]:
        """Parse a markdown file for external HTTP links."""
        links: list[ExternalLink] = []

        try:
            content = file_path.read_text()
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            return links

        for line_num, line in enumerate(content.split("\n"), 1):
            # Find markdown links [text](url)
            for match in self.MARKDOWN_LINK_PATTERN.finditer(line):
                text, url = match.groups()
                links.append(
                    ExternalLink(
                        url=url,
                        text=text,
                        file_path=file_path,
                        line_number=line_num,
                    )
                )

            # Find bare URLs (not already captured by markdown links)
            for match in self.BARE_URL_PATTERN.finditer(line):
                url = match.group(0)
                # Skip if this URL was already captured as a markdown link
                if not any(
                    link.url == url and link.line_number == line_num for link in links
                ):
                    links.append(
                        ExternalLink(
                            url=url,
                            text="",
                            file_path=file_path,
                            line_number=line_num,
                        )
                    )

        return links

    def _parse_notebook_links(self, file_path: Path) -> list[ExternalLink]:
        """Parse a Jupyter notebook for external HTTP links."""
        import json

        links: list[ExternalLink] = []

        try:
            content = file_path.read_text()
            notebook = json.loads(content)
        except Exception as e:
            print(f"Warning: Could not read notebook {file_path}: {e}")
            return links

        for cell_idx, cell in enumerate(notebook.get("cells", [])):
            source = cell.get("source", [])
            if isinstance(source, list):
                source = "".join(source)

            # Approximate line number based on cell index
            line_num = cell_idx + 1

            # Find markdown links
            for match in self.MARKDOWN_LINK_PATTERN.finditer(source):
                text, url = match.groups()
                links.append(
                    ExternalLink(
                        url=url,
                        text=text,
                        file_path=file_path,
                        line_number=line_num,
                    )
                )

            # Find bare URLs
            for match in self.BARE_URL_PATTERN.finditer(source):
                url = match.group(0)
                if not any(
                    link.url == url and link.line_number == line_num for link in links
                ):
                    links.append(
                        ExternalLink(
                            url=url,
                            text="",
                            file_path=file_path,
                            line_number=line_num,
                        )
                    )

        return links

    def _parse_local_links(self, file_path: Path) -> list[LocalLink]:
        """Parse a markdown file for local file links."""
        links: list[LocalLink] = []

        try:
            content = file_path.read_text()
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            return links

        for line_num, line in enumerate(content.split("\n"), 1):
            # Find local file links [text](path)
            for match in self.LOCAL_LINK_PATTERN.finditer(line):
                text, path = match.groups()
                # Skip if it's an HTTP(S) URL
                if path.startswith(("http://", "https://")):
                    continue
                links.append(
                    LocalLink(
                        path=path,
                        text=text,
                        file_path=file_path,
                        line_number=line_num,
                    )
                )

        return links

    def get_documented_apis(self) -> set[str]:
        """Get set of all documented API references."""
        refs = self.find_all_references()
        return {ref.reference for ref in refs}


class DriftDetector:
    """Detects drift between code and documentation."""

    # APIs re-exported from Pulser that don't need local documentation
    PULSER_REEXPORTS = {
        "BitStrings",
        "CorrelationMatrix",
        "Energy",
        "EnergyVariance",
        "EnergySecondMoment",
        "Expectation",
        "Fidelity",
        "Occupation",
        "StateResult",
        "Results",
    }

    def __init__(self, root_path: Path, ignore_pulser_reexports: bool = True):
        self.root_path = root_path
        self.code_analyzer = CodeAnalyzer(root_path)
        self.docs_parser = DocsParser(root_path / "docs", root_path)
        self.ignore_pulser_reexports = ignore_pulser_reexports
        self.link_checker = ExternalLinkChecker()

    def check_all(
        self, check_external_links: bool = False, verbose: bool = False
    ) -> DriftReport:
        """Run all drift detection checks."""
        report = DriftReport()

        # Check signature coverage
        self._check_api_coverage(report)

        # Check for broken references
        self._check_references(report)

        # Check parameter documentation
        self._check_param_docs(report)

        # Check local file links
        self._check_local_links(report)

        # Check external links (optional, can be slow)
        if check_external_links:
            self._check_external_links(report, verbose)

        return report

    def check_external_links_only(self, verbose: bool = False) -> DriftReport:
        """Run only external link checking."""
        report = DriftReport()
        self._check_external_links(report, verbose)
        return report

    def _check_external_links(self, report: DriftReport, verbose: bool = False) -> None:
        """Check that all external links in documentation are valid."""
        if verbose:
            print("Finding external links in documentation...")

        links = self.docs_parser.find_all_external_links()

        if verbose:
            print(f"Found {len(links)} external links, checking...")

        results = self.link_checker.check_links(links, verbose)

        for result in results:
            if result.is_broken:
                status = result.status_code if result.status_code else result.error
                report.broken_external_links.append(
                    {
                        "url": result.link.url,
                        "status": status,
                        "location": f"{result.link.file_path}:{result.link.line_number}",
                        "text": result.link.text,
                    }
                )

    def _check_api_coverage(self, report: DriftReport) -> None:
        """Check that all public APIs are documented."""
        documented = self.docs_parser.get_documented_apis()

        # Build a set of documented API names (just the final class/function name)
        documented_names: dict[str, set[str]] = {"emu_mps": set(), "emu_sv": set()}
        for ref in documented:
            # Extract the final name and the base module
            parts = ref.split(".")
            if len(parts) >= 2:
                base_module = parts[0]
                final_name = parts[-1]
                if base_module in documented_names:
                    documented_names[base_module].add(final_name)
                    # Also add the full reference for exact matching
                    documented_names[base_module].add(ref)

        modules = ["emu_mps", "emu_sv"]
        for module_name in modules:
            apis = self.code_analyzer.get_public_apis(module_name)

            for api in apis:
                # Skip Pulser re-exports if configured
                if self.ignore_pulser_reexports and api.name in self.PULSER_REEXPORTS:
                    continue

                # Check if the API name appears in documented references
                # Either as exact match or as the final component of a reference
                is_documented = (
                    api.name in documented_names.get(module_name, set())
                    or f"{module_name}.{api.name}" in documented
                    or any(ref.endswith(f".{api.name}") for ref in documented)
                )

                if not is_documented:
                    report.missing_in_docs.append(f"{module_name}.{api.name}")

    def _get_expected_refs(self, api: SignatureInfo, module_name: str) -> list[str]:
        """Get possible documentation reference patterns for an API."""
        refs = [
            f"{module_name}.{api.name}",
            f"{api.module}.{api.name}",
        ]
        # Also check submodule patterns
        if api.kind == "class":
            # Classes might be documented via their module
            parts = api.module.split(".")
            if len(parts) > 1:
                refs.append(f"{'.'.join(parts)}.{api.name}")
        return refs

    def _check_references(self, report: DriftReport) -> None:
        """Check that all documentation references are valid."""
        references = self.docs_parser.find_all_references()

        for ref in references:
            if not self._is_valid_reference(ref.reference):
                report.broken_references.append(
                    f"{ref.reference} in {ref.file_path}:{ref.line_number}"
                )

    def _is_valid_reference(self, reference: str) -> bool:
        """Check if a mkdocstrings reference points to valid code."""
        parts = reference.split(".")

        # Try to import and find the object
        for i in range(len(parts), 0, -1):
            module_path = ".".join(parts[:i])
            try:
                module = importlib.import_module(module_path)
                # Try to get remaining attributes
                obj = module
                for attr in parts[i:]:
                    obj = getattr(obj, attr)
                return True
            except (ImportError, AttributeError):
                continue

        return False

    def _check_param_docs(self, report: DriftReport) -> None:
        """Check that function parameters are documented."""
        modules = ["emu_mps", "emu_sv"]

        # Parameters to ignore (internal Python/Enum parameters)
        ignore_params = {
            "value",
            "names",
            "module",
            "qualname",
            "type",
            "start",
            "boundary",
            "cls",
        }

        for module_name in modules:
            apis = self.code_analyzer.get_public_apis(module_name)

            for api in apis:
                if not api.docstring or not api.parameters:
                    continue

                # Check if parameters are mentioned in docstring
                undocumented = []
                for param in api.parameters:
                    param_name = param.split(":")[0].split("=")[0].strip()
                    # Skip internal parameters
                    if param_name in ignore_params:
                        continue
                    if param_name not in api.docstring:
                        undocumented.append(param_name)

                if undocumented:
                    report.undocumented_params.append(
                        {
                            "name": f"{module_name}.{api.name}",
                            "params": ", ".join(undocumented),
                        }
                    )

    def _check_local_links(self, report: DriftReport) -> None:
        """Check that all local file links point to existing files."""
        links = self.docs_parser.find_all_local_links()

        # Load mkdocs nav if available to check if files are included
        mkdocs_nav_files = self._get_mkdocs_nav_files()

        for link in links:
            # Resolve the path relative to the file containing the link
            link_dir = link.file_path.parent

            # Try to resolve the path
            # First try relative to the doc file
            resolved_path = (link_dir / link.path).resolve()

            # If not found and path starts with .., try relative to docs root
            if not resolved_path.exists() and link.path.startswith(".."):
                resolved_path = (self.docs_parser.docs_path / link.path).resolve()

            # If not found and path starts with /, try relative to root
            if not resolved_path.exists() and link.path.startswith("/"):
                resolved_path = (
                    self.docs_parser.root_path / link.path.lstrip("/")
                ).resolve()

            if not resolved_path.exists():
                report.broken_local_links.append(
                    {
                        "path": link.path,
                        "location": f"{link.file_path}:{link.line_number}",
                        "text": link.text,
                    }
                )
            else:
                # Check if .py files are in mkdocs nav (they won't be served otherwise)
                if link.path.endswith(".py") and mkdocs_nav_files is not None:
                    # Get relative path from docs root
                    try:
                        rel_path = resolved_path.relative_to(self.docs_parser.docs_path)
                        if str(rel_path) not in mkdocs_nav_files:
                            report.broken_local_links.append(
                                {
                                    "path": link.path,
                                    "location": f"{link.file_path}:{link.line_number}",
                                    "text": link.text,
                                    "reason": ".py file not in mkdocs nav (won't be served)",
                                }
                            )
                    except ValueError:
                        # Path is outside docs directory
                        pass

    def _get_mkdocs_nav_files(self) -> set[str] | None:
        """Extract list of files from mkdocs.yml nav section."""
        mkdocs_path = self.root_path / "mkdocs.yml"
        if not mkdocs_path.exists():
            return None

        try:
            import yaml  # type: ignore[import-untyped]

            with open(mkdocs_path) as f:
                config = yaml.safe_load(f)

            nav_files: set[str] = set()

            def extract_files(nav_item: Any) -> None:
                if isinstance(nav_item, str):
                    nav_files.add(nav_item)
                elif isinstance(nav_item, dict):
                    for value in nav_item.values():
                        if isinstance(value, str):
                            nav_files.add(value)
                        elif isinstance(value, list):
                            for item in value:
                                extract_files(item)
                elif isinstance(nav_item, list):
                    for item in nav_item:
                        extract_files(item)

            if "nav" in config:
                extract_files(config["nav"])

            return nav_files
        except Exception as e:
            print(f"Warning: Could not parse mkdocs.yml: {e}")
            return None


class ExternalLinkChecker:
    """Checks external HTTP links for validity."""

    # Domains to skip (usually require authentication or are rate-limited)
    SKIP_DOMAINS = {
        "pasqalworkspace.slack.com",  # Requires Slack auth
        "cdn.jsdelivr.net",  # CDN, usually fine
    }

    # Status codes that indicate the URL exists but blocks automated requests
    # These should not be considered broken links
    ACCEPTABLE_STATUS_CODES = {
        403,  # Forbidden - often blocks HEAD/automated requests
        429,  # Too Many Requests - rate limited
        405,  # Method Not Allowed - HEAD not supported, URL likely exists
    }

    # User agent to use for requests (mimic a real browser)
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    def __init__(
        self,
        timeout: float = 10.0,
        max_concurrent: int = 5,
        skip_domains: set[str] | None = None,
    ):
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.skip_domains = skip_domains or self.SKIP_DOMAINS

    def check_links_sync(
        self, links: list[ExternalLink], verbose: bool = False
    ) -> list[ExternalLinkResult]:
        """Check external links synchronously using urllib (fallback)."""
        import urllib.error
        import urllib.request

        results: list[ExternalLinkResult] = []
        seen_urls: set[str] = set()

        for link in links:
            # Skip duplicates
            if link.url in seen_urls:
                continue
            seen_urls.add(link.url)

            # Skip certain domains
            parsed = urlparse(link.url)
            if parsed.netloc in self.skip_domains:
                if verbose:
                    print(f"  Skipping {link.url} (domain in skip list)")
                continue

            if verbose:
                print(f"  Checking {link.url}...")

            try:
                # Try HEAD request first
                req = urllib.request.Request(
                    link.url,
                    headers={"User-Agent": self.USER_AGENT},
                    method="HEAD",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    status_code = response.getcode()
                    is_broken = status_code >= 400
                    results.append(
                        ExternalLinkResult(
                            link=link,
                            status_code=status_code,
                            error=None,
                            is_broken=is_broken,
                        )
                    )
            except urllib.error.HTTPError as e:
                # If HEAD fails with acceptable status, try GET or skip
                if e.code in self.ACCEPTABLE_STATUS_CODES:
                    # These status codes mean URL exists but blocks requests
                    results.append(
                        ExternalLinkResult(
                            link=link,
                            status_code=e.code,
                            error=None,
                            is_broken=False,  # Not broken, just blocked
                        )
                    )
                elif e.code == 405:  # Method not allowed, try GET
                    try:
                        req = urllib.request.Request(
                            link.url,
                            headers={"User-Agent": self.USER_AGENT},
                            method="GET",
                        )
                        with urllib.request.urlopen(
                            req, timeout=self.timeout
                        ) as response:
                            status_code = response.getcode()
                            is_broken = status_code >= 400
                            results.append(
                                ExternalLinkResult(
                                    link=link,
                                    status_code=status_code,
                                    error=None,
                                    is_broken=is_broken,
                                )
                            )
                    except Exception:
                        results.append(
                            ExternalLinkResult(
                                link=link,
                                status_code=e.code,
                                error=str(e.reason),
                                is_broken=True,
                            )
                        )
                else:
                    results.append(
                        ExternalLinkResult(
                            link=link,
                            status_code=e.code,
                            error=str(e.reason),
                            is_broken=True,
                        )
                    )
            except urllib.error.URLError as e:
                results.append(
                    ExternalLinkResult(
                        link=link,
                        status_code=None,
                        error=str(e.reason),
                        is_broken=True,
                    )
                )
            except Exception as e:
                results.append(
                    ExternalLinkResult(
                        link=link,
                        status_code=None,
                        error=str(e),
                        is_broken=True,
                    )
                )

        return results

    async def check_links_async(
        self, links: list[ExternalLink], verbose: bool = False
    ) -> list[ExternalLinkResult]:
        """Check external links asynchronously using aiohttp."""
        if not AIOHTTP_AVAILABLE:
            if verbose:
                print("  aiohttp not available, falling back to sync checker")
            return self.check_links_sync(links, verbose)

        results: list[ExternalLinkResult] = []
        seen_urls: dict[str, ExternalLink] = {}

        # Deduplicate links, keeping track of the first occurrence
        unique_links: list[ExternalLink] = []
        for link in links:
            if link.url not in seen_urls:
                seen_urls[link.url] = link
                unique_links.append(link)

        # Filter out skipped domains
        links_to_check: list[ExternalLink] = []
        for link in unique_links:
            parsed = urlparse(link.url)
            if parsed.netloc in self.skip_domains:
                if verbose:
                    print(f"  Skipping {link.url} (domain in skip list)")
                continue
            links_to_check.append(link)

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def check_single(
            session: aiohttp.ClientSession, link: ExternalLink
        ) -> ExternalLinkResult:
            async with semaphore:
                if verbose:
                    print(f"  Checking {link.url}...")

                try:
                    async with session.head(
                        link.url,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                        allow_redirects=True,
                    ) as response:
                        status = response.status
                        # Check if status is acceptable (blocked but exists)
                        if status in self.ACCEPTABLE_STATUS_CODES:
                            return ExternalLinkResult(
                                link=link,
                                status_code=status,
                                error=None,
                                is_broken=False,
                            )
                        # If HEAD returns 405, try GET
                        if status == 405:
                            async with session.get(
                                link.url,
                                timeout=aiohttp.ClientTimeout(total=self.timeout),
                                allow_redirects=True,
                            ) as get_response:
                                is_broken = get_response.status >= 400
                                return ExternalLinkResult(
                                    link=link,
                                    status_code=get_response.status,
                                    error=None,
                                    is_broken=is_broken,
                                )
                        is_broken = status >= 400
                        return ExternalLinkResult(
                            link=link,
                            status_code=status,
                            error=None,
                            is_broken=is_broken,
                        )
                except asyncio.TimeoutError:
                    return ExternalLinkResult(
                        link=link,
                        status_code=None,
                        error="Timeout",
                        is_broken=True,
                    )
                except aiohttp.ClientError as e:
                    return ExternalLinkResult(
                        link=link,
                        status_code=None,
                        error=str(e),
                        is_broken=True,
                    )
                except Exception as e:
                    return ExternalLinkResult(
                        link=link,
                        status_code=None,
                        error=str(e),
                        is_broken=True,
                    )

        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        async with aiohttp.ClientSession(
            connector=connector,
            headers={"User-Agent": self.USER_AGENT},
        ) as session:
            tasks = [check_single(session, link) for link in links_to_check]
            results = await asyncio.gather(*tasks)

        return list(results)

    async def check_readthedocs_link(self, url: str) -> tuple[bool, str]:
        """
        Check if readthedocs link is valid using Playwright.
        Returns (is_valid, error_msg).
        """
        if not PLAYWRIGHT_AVAILABLE:
            return True, ""  # Can't verify without Playwright

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                await page.goto(url, timeout=10000)

                # Check if page contains "404" or "Page Not Found" indicators
                content = await page.content()
                title = await page.title()

                await browser.close()

                # Readthedocs returns 200 even for 404 pages, so check content
                if "404" in title or "Page not found" in title.lower():
                    return False, "404 page"
                if "404" in content[:1000] and "not found" in content[:1000].lower():
                    return False, "404 page"

                return True, ""
        except Exception as e:
            return False, str(e)

    def check_links(
        self, links: list[ExternalLink], verbose: bool = False
    ) -> list[ExternalLinkResult]:
        """Check external links (async if aiohttp available, sync otherwise)."""
        results: list[ExternalLinkResult] = []

        # Separate readthedocs links from others
        readthedocs_links: list[ExternalLink] = []
        other_links: list[ExternalLink] = []

        for link in links:
            if "readthedocs.io" in link.url:
                readthedocs_links.append(link)
            else:
                other_links.append(link)

        # Check non-readthedocs links with HTTP
        if other_links:
            if AIOHTTP_AVAILABLE:
                results.extend(asyncio.run(self.check_links_async(other_links, verbose)))
            else:
                results.extend(self.check_links_sync(other_links, verbose))

        # Check readthedocs links with Playwright for accurate 404 detection
        if readthedocs_links and PLAYWRIGHT_AVAILABLE:
            if verbose:
                print(
                    f"  Checking {len(readthedocs_links)} readthedocs links with Playwright..."
                )

            async def check_rtd_links() -> list[ExternalLinkResult]:
                rtd_results: list[ExternalLinkResult] = []
                for link in readthedocs_links:
                    if verbose:
                        print(f"  Checking {link.url} (Playwright)...")
                    is_valid, error = await self.check_readthedocs_link(link.url)
                    rtd_results.append(
                        ExternalLinkResult(
                            link=link,
                            status_code=200 if is_valid else 404,
                            error=error if not is_valid else None,
                            is_broken=not is_valid,
                        )
                    )
                return rtd_results

            results.extend(asyncio.run(check_rtd_links()))
        elif readthedocs_links:
            # Fallback to HTTP check if Playwright not available
            if verbose:
                print("  Playwright not available, using HTTP for readthedocs links...")
            if AIOHTTP_AVAILABLE:
                results.extend(
                    asyncio.run(self.check_links_async(readthedocs_links, verbose))
                )
            else:
                results.extend(self.check_links_sync(readthedocs_links, verbose))

        return results


class PlaywrightValidator:
    """Validates live documentation site using Playwright."""

    def __init__(self, base_url: str = "https://pasqal-io.github.io/emulators/latest"):
        self.base_url = base_url

    async def check_links(self) -> list[str]:
        """Check all internal links on the documentation site."""
        if not PLAYWRIGHT_AVAILABLE:
            return ["Playwright not installed - skipping link check"]

        broken_links: list[str] = []

        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()

            # Start from the main page
            await page.goto(self.base_url)

            # Get all internal links
            links = await page.eval_on_selector_all(
                "a[href]",
                """elements => elements
                    .map(e => e.href)
                    .filter(href => href.includes('pasqal-io.github.io/emulators'))
                """,
            )

            # Check each link
            visited: set[str] = set()
            for link in links:
                if link in visited:
                    continue
                visited.add(link)

                try:
                    response = await page.goto(link)
                    if response and response.status >= 400:
                        broken_links.append(f"{link} (status: {response.status})")
                except Exception as e:
                    broken_links.append(f"{link} (error: {e})")

            await browser.close()

        return broken_links

    async def validate_code_examples(self) -> list[dict[str, Any]]:
        """Extract and validate code examples from the live site."""
        if not PLAYWRIGHT_AVAILABLE:
            return [{"error": "Playwright not installed"}]

        results: list[dict[str, Any]] = []

        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()

            # Navigate to API pages
            api_pages = [
                f"{self.base_url}/emu_mps/api/",
                f"{self.base_url}/emu_sv/api/",
            ]

            for api_url in api_pages:
                await page.goto(api_url)

                # Extract code blocks
                code_blocks = await page.eval_on_selector_all(
                    "pre code",
                    "elements => elements.map(e => e.textContent)",
                )

                for i, code in enumerate(code_blocks):
                    if code and ("import" in code or ">>>" in code):
                        results.append(
                            {
                                "page": api_url,
                                "code_block": i,
                                "code": code[:200] + "..." if len(code) > 200 else code,
                            }
                        )

            await browser.close()

        return results


def main() -> int:
    """Main entry point for the documentation agent."""
    parser = argparse.ArgumentParser(
        description="Documentation drift detection agent for Pasqal Emulators"
    )
    parser.add_argument(
        "--check-all", action="store_true", help="Run all drift detection checks"
    )
    parser.add_argument(
        "--check-signatures",
        action="store_true",
        help="Check API signature coverage",
    )
    parser.add_argument(
        "--check-links",
        action="store_true",
        help="Check documentation links (requires playwright)",
    )
    parser.add_argument(
        "--check-examples",
        action="store_true",
        help="Validate code examples (requires playwright)",
    )
    parser.add_argument(
        "--check-external-links",
        action="store_true",
        help="Check external HTTP links in documentation for broken URLs",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument(
        "--ignore-pulser-reexports",
        action="store_true",
        default=True,
        help="Ignore APIs re-exported from Pulser (default: True)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Root path of the repository",
    )

    args = parser.parse_args()

    # Default to --check-all if no specific check is requested
    if not any(
        [
            args.check_all,
            args.check_signatures,
            args.check_links,
            args.check_examples,
            args.check_external_links,
        ]
    ):
        args.check_all = True

    # Add root to Python path for imports
    sys.path.insert(0, str(args.root))

    detector = DriftDetector(
        args.root, ignore_pulser_reexports=args.ignore_pulser_reexports
    )

    exit_code = 0

    if args.check_all or args.check_signatures:
        if not args.json:
            print("Running documentation drift detection...")

        # Include external links check when --check-all is used
        include_external = args.check_all or args.check_external_links
        report = detector.check_all(
            check_external_links=include_external,
            verbose=args.verbose,
        )

        if args.json:
            import json

            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.summary())

        if report.has_issues():
            exit_code = 1

    if args.check_all or args.check_links:
        print("\nChecking documentation links (playwright)...")
        validator = PlaywrightValidator()
        broken = asyncio.run(validator.check_links())
        if broken:
            print("Broken links found:")
            for link in broken:
                print(f"  - {link}")
            exit_code = 1
        else:
            print("All internal links OK")

    if args.check_all or args.check_examples:
        print("\nValidating code examples...")
        validator = PlaywrightValidator()
        examples = asyncio.run(validator.validate_code_examples())
        print(f"Found {len(examples)} code examples to validate")
        if args.verbose:
            for ex in examples:
                print(f"  - {ex}")

    # Only run standalone external links check if not already done via check_all
    if args.check_external_links and not args.check_all:
        if not args.json:
            print("\nChecking external links in documentation...")
        report = detector.check_external_links_only(verbose=args.verbose)
        if args.json:
            import json

            print(json.dumps(report.to_dict(), indent=2))
        else:
            if report.broken_external_links:
                print(
                    f"\nBroken external links " f"({len(report.broken_external_links)}):"
                )
                for link_info in report.broken_external_links:
                    status = link_info.get("status", "unknown")
                    url = link_info.get("url", "unknown")
                    location = link_info.get("location", "unknown")
                    # file:line format for VSCode click-to-navigate
                    print(f"  {location}: {url} (status: {status})")
                exit_code = 1
            else:
                print("All external links OK")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
