# benchNTT

**benchNTT** provides high-performance CPU implementations of 128-bit cryptographic kernels, specifically number theoretic transforms (NTTs) and basic linear algebra subprograms (BLAS) operations. The kernels are implemented using scalar, AVX2, and AVX-512 instructions. For more details, please refer to our [paper](https://dl.acm.org/doi/10.1145/3725843.3756120).

Related GPU work is available in our [MoMA](https://github.com/Naifeng/moma) repository.

## Dependencies

One of the following compilers is required:

- **Intel:** [ICX](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html) >= 2024.2.0
- **AMD:** [AOCC](https://www.amd.com/en/developer/aocc.html) >= 16.0.3
- **Generic:** [GCC](https://gcc.gnu.org/)

## Usage

The build system supports multiple compilers and architectures:

### Intel CPUs

*ICX required for best performance.*

```bash
make blas-intel     # Build and run BLAS benchmarks
make ntt-intel      # Build and run NTT benchmarks
```

### AMD CPUs

*AOCC required for best performance.*

```bash
make blas-amd       # Build and run BLAS benchmarks
make ntt-amd        # Build and run NTT benchmarks
```

### Generic

*GCC required.*

```bash
make blas           # Build and run BLAS benchmarks
make ntt            # Build and run NTT benchmarks
```

> **Note I:** If your CPU does not support AVX2 or AVX-512, you can comment out the corresponding build targets (see `BLAS_TARGETS` and `NTT_TARGETS`) in the `Makefile` to build only the supported implementations.

> **Note II:** You can also use `make all` to build and run all targets with GCC. Always run `make clean` before switching compilers.

## Environment Requirements

For accurate benchmarking, please ensure that no other processes are competing for CPU resources. Running benchmarks on dedicated machines (e.g., via SSH) and taking multiple measurements is recommended to minimize noise.

## Directory Structure

```
benchntt (this repository)
│    README.md (this file)
│    Makefile
│    ...
│
└────benchmark
│    │    run_ntt.sh
│ 
└────data
│    │    x_1024.txt
│    │    ver_1024.txt
│    │    ...
│ 
└────src
     │    blas_avx512.c
     │    ntt_avx512.c
     |    ...
```

- `benchmark`: Bash script for benchmarking different NTT sizes.
- `data`: Verification data for NTT and BLAS operations.
- `src`: Scalar, AVX2, and AVX-512 implementations of NTT and BLAS operations.

## License

Distributed under the SPIRAL License. For more details, please refer to the `LICENSE` file.