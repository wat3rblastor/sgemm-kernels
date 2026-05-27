#!/usr/bin/env bash

set -euo pipefail

OUTPUT_DIR="./test"
BINARY="./build/gemm"

mkdir -p "${OUTPUT_DIR}"
rm -f "${OUTPUT_DIR}"/test_kernel_*.txt

echo -n "test_kernel:"
while IFS= read -r i
do
  echo -n "${i}..."
  file_name="${OUTPUT_DIR}/test_kernel_${i}.txt"
  "${BINARY}" "${i}" > "${file_name}"
done < <("${BINARY}" --list-ids)
echo
