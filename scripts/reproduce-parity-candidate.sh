#!/usr/bin/env bash
# Reproduce the archived component comparison in an isolated source tree.
set -euo pipefail

if [[ $# != 1 ]]; then
    printf 'Usage: %s /absolute/path/to/validation-venv/bin/python\n' "$0" >&2
    exit 2
fi
python_bin=$1
if [[ $python_bin != /* || ! -x $python_bin ]]; then
    printf 'Pass an absolute path to the validation environment Python executable.\n' >&2
    exit 2
fi

cd "$(dirname "$0")/.."
candidate=7e7da4716ff5caf20c6dc212d0a31318cfd0f961
fortran=9f7e9caadbd6e8584e0c8fc9bf58536869b5480c
snapshot=fe4a5563fec7eacaba86537e82157f866917dfd8

for commit in "$candidate" "$snapshot"; do
    if ! git cat-file -e "${commit}^{commit}"; then
        printf 'Missing JAX commit %s. Fetch the AyaLahlou/clm-ml-jax history and retry.\n' "$commit" >&2
        exit 1
    fi
done
[[ $(git rev-parse "${candidate}:clm-ml-fortran") == "$fortran" ]]
printf 'Historical JAX candidate %s\nFortran gitlink %s\nFixture-carrying JAX snapshot %s\n' \
    "$candidate" "$fortran" "$snapshot"

scratch=$(mktemp -d "${TMPDIR:-/tmp}/clm-parity.XXXXXX")
# Only remove the isolated archive made by this invocation.
trap 'rm -rf -- "$scratch"' EXIT
mkdir "$scratch/candidate" "$scratch/snapshot"
git archive "$candidate" | tar -xf - -C "$scratch/candidate"
git archive "$snapshot" | tar -xf - -C "$scratch/snapshot"
mkdir -p "$scratch/candidate/clm-ml-fortran"
cp -R "$scratch/snapshot/tests/fortran_validation/golden_IO" \
    "$scratch/candidate/clm-ml-fortran/"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
for tree in candidate snapshot; do
    cd "$scratch/$tree"
    printf '\nRunning archived component cases on %s\n' "$tree"
    "$python_bin" -m pytest tests/fortran_validation -q --junitxml=component-results.xml
    "$python_bin" - <<'PY'
from xml.etree import ElementTree

cases = ElementTree.parse("component-results.xml").findall(".//testcase")
uncompleted = ("skipped", "failure", "error")
if len(cases) != 160 or any(
    case.find(tag) is not None for case in cases for tag in uncompleted
):
    raise SystemExit("Expected 160 completed component cases with zero skips or failures.")
PY
done
