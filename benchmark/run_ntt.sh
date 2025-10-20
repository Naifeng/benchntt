#!/bin/bash

cd src

run_benchmark() {
    local impl=$1
    local c_file="ntt_${impl}.c"
    local exec_file="../bin/ntt_${impl}"

    echo -e "${impl} NTT:"
    printf "%-25s %-25s\n" "Size (N)" "Runtime per butterfly [ns]"

    for STAGES in {8..20}; do
        sed -i "s/#define stages [0-9]*/#define stages ${STAGES}/" "${c_file}"
        
        # compile and run
        ${CC:-gcc} ${CFLAGS:--march=native -O3 -w} -o "${exec_file}" "${c_file}" config.c && "${exec_file}"
    done

    # reset the stages back to a default value
    sed -i "s/#define stages [0-9]*/#define stages 10/" "${c_file}"
}

# run benchmarks for each implementation
for impl in scalar avx2 avx512; do
    run_benchmark $impl
done