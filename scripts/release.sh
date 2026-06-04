#!/usr/bin/env bash
# Builds, uploads to PyPI, and tags the release.
# Should only be run by project maintainers.
set -euo pipefail

VENV=".venv/bin"

# Run pre-commit checks
bash tests/check.sh

# Must be on main with a clean working tree
current_branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$current_branch" != "main" ]]; then
    echo "Error: must be on main branch (currently on '$current_branch')"
    exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Error: working tree is not clean"
    exit 1
fi

# Read version from package
version=$("$VENV/python" -c "import seqscore; print(seqscore.__version__)")
tag="v$version"

# Abort if tag already exists
if git rev-parse "$tag" >/dev/null 2>&1; then
    echo "Error: tag $tag already exists. Update __version__ in seqscore/__init__.py."
    exit 1
fi

echo "Releasing $tag"

# Build
rm -rf dist/
"$VENV/python" -m build

# Tag and push
git tag "$tag"
git push origin "$tag"

# Prompt to verify tag before uploading
echo ""
echo "Tag $tag pushed. Check the release on GitHub before uploading to PyPI:"
echo "  https://github.com/bltlab/seqscore/releases/tag/$tag"
echo ""
read -r -p "Upload to PyPI? [y/N] " confirm
if [[ "${confirm,,}" != "y" ]]; then
    echo "Aborted. Re-run this script to retry the upload."
    exit 1
fi

# Upload to PyPI
"$VENV/twine" upload dist/*

echo "Done: $tag released and pushed"
