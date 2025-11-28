#!/usr/bin/env python3
"""
Documentation Agent for Pasqal Emulators.

A CLI tool to detect documentation drift between code and docs.
Validates that public APIs are properly documented and that
documentation stays in sync with code changes.

Usage:
    python scripts/doc_agent.py --check-all
    python scripts/doc_agent.py --check-signatures
    python scripts/doc_agent.py --check-links
    python scripts/doc_agent.py --check-examples
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
class DriftReport:
    """Report of documentation drift issues."""

    missing_in_docs: list[str] = field(default_factory=list)
    signature_mismatches: list[dict[str, Any]] = field(default_factory=list)
    broken_references: list[str] = field(default_factory=list)
    undocumented_params: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def has_issues(self) -> bool:
        return bool(
            self.missing_in_docs
            or self.signature_mismatches
            or self.broken_references
            or self.undocumented_params
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON output."""
        return {
            "missing_in_docs": self.missing_in_docs,
            "signature_mismatches": self.signature_mismatches,
            "broken_references": self.broken_references,
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
    """Parses mkdocs documentation to find mkdocstrings references."""

    # Pattern for mkdocstrings references like "::: emu_mps.mps.MPS"
    MKDOCSTRINGS_PATTERN = re.compile(r"^:::?\s+([\w.]+)", re.MULTILINE)

    def __init__(self, docs_path: Path):
        self.docs_path = docs_path

    def find_all_references(self) -> list[DocReference]:
        """Find all mkdocstrings references in documentation files."""
        references: list[DocReference] = []

        for md_file in self.docs_path.rglob("*.md"):
            refs = self._parse_file(md_file)
            references.extend(refs)

        return references

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
        self.docs_parser = DocsParser(root_path / "docs")
        self.ignore_pulser_reexports = ignore_pulser_reexports

    def check_all(self) -> DriftReport:
        """Run all drift detection checks."""
        report = DriftReport()

        # Check signature coverage
        self._check_api_coverage(report)

        # Check for broken references
        self._check_references(report)

        # Check parameter documentation
        self._check_param_docs(report)

        return report

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
        [args.check_all, args.check_signatures, args.check_links, args.check_examples]
    ):
        args.check_all = True

    # Add root to Python path for imports
    sys.path.insert(0, str(args.root))

    detector = DriftDetector(
        args.root, ignore_pulser_reexports=args.ignore_pulser_reexports
    )

    if args.check_all or args.check_signatures:
        if not args.json:
            print("Running documentation drift detection...")

        report = detector.check_all()

        if args.json:
            import json

            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.summary())

        if report.has_issues():
            return 1

    if args.check_links:
        import asyncio

        print("\nChecking documentation links...")
        validator = PlaywrightValidator()
        broken = asyncio.run(validator.check_links())
        if broken:
            print("Broken links found:")
            for link in broken:
                print(f"  - {link}")
            return 1
        print("All links OK")

    if args.check_examples:
        import asyncio

        print("\nValidating code examples...")
        validator = PlaywrightValidator()
        examples = asyncio.run(validator.validate_code_examples())
        print(f"Found {len(examples)} code examples to validate")
        if args.verbose:
            for ex in examples:
                print(f"  - {ex}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
