#!/usr/bin/env bash
set -euo pipefail

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$release_root"

failed=0

mapfile -d '' tracked_files < <(find . -type f \
  -not -path './.git/*' \
  -not -path './data/*' \
  -not -path './outputs/*' \
  -not -path './.venv/*' \
  -not -path './.pytest_cache/*' \
  -not -path './.ruff_cache/*' \
  -not -path './src/*.egg-info/*' \
  -not -path '*/__pycache__/*' \
  -not -path './scripts/release_audit.sh' \
  -not -name '*.pyc' -print0)

mapfile -d '' text_files < <(find . -type f \
  -not -path './.git/*' \
  -not -path './data/*' \
  -not -path './outputs/*' \
  -not -path './.venv/*' \
  -not -path './.pytest_cache/*' \
  -not -path './.ruff_cache/*' \
  -not -path './src/*.egg-info/*' \
  -not -path '*/__pycache__/*' \
  -not -path './scripts/release_audit.sh' \
  -not -name '*.pyc' \
  -not -name '*.pdf' \
  -not -name '*.png' \
  -not -name '*.jpg' \
  -not -name '*.jpeg' -print0)

scan_text() {
  local pattern="$1"
  shift
  if command -v rg >/dev/null 2>&1; then
    rg -n -i "$pattern" "$@"
  else
    grep -Eni "$pattern" "$@"
  fi
}

for suffix in pt pth ckpt safetensors joblib pkl pickle xlsx xls fif edf set mat; do
  if find . -type f -name "*.${suffix}" -not -path './data/*' -not -path './outputs/*' | grep -q .; then
    echo "ERROR: disallowed artifact type found: *.${suffix}"
    failed=1
  fi
done

if find . -type f -size +5M -not -path './data/*' -not -path './outputs/*' | grep -q .; then
  echo "ERROR: file larger than 5 MiB found outside ignored output directories"
  find . -type f -size +5M -not -path './data/*' -not -path './outputs/*'
  failed=1
fi

path_pattern='/mnt/|/home/|C:\\Users\\'
if (( ${#text_files[@]} )) && scan_text "$path_pattern" "${text_files[@]}"; then
  echo "ERROR: local absolute path or user name found"
  failed=1
fi

secret_pattern='BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9_]{30,}|api[_-]?key[[:space:]]*=[[:space:]]*[A-Za-z0-9]'
if (( ${#text_files[@]} )) && scan_text "$secret_pattern" "${text_files[@]}"; then
  echo "ERROR: possible credential found"
  failed=1
fi

# These tokens are deliberately assembled in the audit script so the scanner
# can check the release tree without matching its own policy definitions.
identity_pattern='Yunchao Min|Nan Yan|Yi Guo|xuanyi_guo|nan\.yan|Shenzhen People.s Hospital|Chinese Academy of Sciences|University of Chinese Academy of Sciences'
funding_pattern='2018YFA0701405|U23B2018|62271477|82371471|JCYJ20220818|KQTD20200820113106007|2021A1515010983'
email_pattern='[[:alnum:]._%+-]+@[[:alnum:].-]+\.[[:alpha:]]{2,}'
if (( ${#text_files[@]} )) && scan_text "$identity_pattern|$funding_pattern|$email_pattern" "${text_files[@]}"; then
  echo "ERROR: personal, institutional, funding, or email information found"
  failed=1
fi

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

echo "Release audit passed. No blocked artifacts, identifying details, funding identifiers, local paths, or obvious credentials found."
