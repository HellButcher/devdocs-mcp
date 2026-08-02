#!/usr/bin/env bash
# Install devdocs-mcp as a uv tool, using the versions pinned in uv.lock.
#
# Usage:
#   ./install.sh            # install with ML (semantic search) dependencies
#   ./install.sh --no-ml    # install without ML dependencies
#
# This works by exporting the project's uv.lock to a temporary requirements
# file and passing it to `uv tool install` as a constraints file, so the
# installed tool gets the exact locked dependency versions instead of a
# fresh, potentially different resolution.
#
# Note: this always passes --force to `uv tool install`, so re-running it
# (or running the other variant) will reinstall/overwrite any existing
# devdocs-mcp tool install, regardless of how it was previously installed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

extra="ml"
package_spec="${SCRIPT_DIR}[ml]"

for arg in "$@"; do
    case "$arg" in
        --no-ml)
            extra=""
            package_spec="${SCRIPT_DIR}"
            ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Usage: $0 [--no-ml]" >&2
            exit 1
            ;;
    esac
done

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: 'uv' is not installed. See https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

tmp_requirements="$(mktemp "${TMPDIR:-/tmp}/devdocs-mcp-requirements.XXXXXX")"
trap 'rm -f "$tmp_requirements"' EXIT

echo "Exporting locked dependencies from uv.lock..."
if [ -n "$extra" ]; then
    uv export --project "$SCRIPT_DIR" --format requirements.txt --extra "$extra" --no-emit-project -o "$tmp_requirements"
else
    uv export --project "$SCRIPT_DIR" --format requirements.txt --no-emit-project -o "$tmp_requirements"
fi

echo "Installing devdocs-mcp as a uv tool..."
uv tool install --constraints "$tmp_requirements" --force "$package_spec"

echo "Done. Run 'devdocs-mcp' to start the server."
