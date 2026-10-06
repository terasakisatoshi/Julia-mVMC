#!/usr/bin/env bash
set -euo pipefail
# Optional developer command. No normal Julia/Rust test invokes this script.
native_dir=${1:?absolute path to original mVMC-1.3.0/src}
probe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
stage_dir=$(mktemp -d /tmp/mvmc-julia-fsz-complex101.XXXXXX)
printf 'stage=%s\n' "$stage_dir"
exec > >(tee "$stage_dir/generation.log") 2>&1
printf '%s  %s\n' 849488176375102fbc3bab7001e0dc27dfa7c8e049fd90ee1d6248ebfba10764 "$native_dir/mVMC/matrix.c" 2b43a47a85ec13833d453a010962334147024a6d6aae01b99bb8b2a0b80c15db "$native_dir/mVMC/vmcmake_fsz.c" | sha256sum -c -
sed -n '1,21p;77,179p' "$native_dir/mVMC/matrix.c" > "$stage_dir/matrix.inc"
sed -n '1,21p;435,517p' "$native_dir/mVMC/vmcmake_fsz.c" > "$stage_dir/initializer.inc"
sed -n '1,21p;58,152p' "$native_dir/mVMC/projection.c" > "$stage_dir/projection.inc"
cp "$probe_dir/complex_probe.c" "$stage_dir/"
{ uname -sm; cc --version; c++ --version; gfortran --version; } > "$stage_dir/environment.txt"
sha256sum "$probe_dir/generate_complex.sh" "$stage_dir/complex_probe.c" "$stage_dir"/*.inc "$native_dir"/mVMC/{matrix.c,vmcmake_fsz.c,projection.c} "$native_dir"/sfmt/{SFMT.c,SFMT.h,SFMT-params.h,SFMT-params19937.h,SFMT-sse2.h} > "$stage_dir/sources.sha256"
for name in zsktrf zsktf2 zlasktrf zskr2 zskr2k; do
    sha256sum "$native_dir/pfapack/fortran/$name.f" >> "$stage_dir/sources.sha256"
    gfortran -O0 -ffp-contract=off -c "$native_dir/pfapack/fortran/$name.f" -o "$stage_dir/$name.o"
done
for name in ltl2inv ilaenv_lauum; do
    c++ -std=c++11 -O0 -ffp-contract=off -DBLAS_EXTERNAL -I"$native_dir/common" -I"$native_dir/common/deps" -I"$native_dir/ltl2inv" -c "$native_dir/ltl2inv/$name.cc" -o "$stage_dir/$name.o"
done
gfortran -O0 -ffp-contract=off -J"$stage_dir" -c "$native_dir/ltl2inv/ilaenv_wrap.f90" -o "$stage_dir/ilaenv_wrap.o"
sha256sum "$native_dir"/ltl2inv/{ltl2inv.cc,ilaenv_lauum.cc,ilaenv_wrap.f90,invert.tcc,pfaffian.tcc,trmmt.tcc} "$native_dir"/common/{blalink.hh,blalink_fort.h,colmaj.hh} >> "$stage_dir/sources.sha256"
cc -std=c11 -O0 -ffp-contract=off -DMEXP=19937 -Wno-unknown-pragmas -I"$stage_dir" -I"$native_dir/sfmt" -c "$stage_dir/complex_probe.c" -o "$stage_dir/probe.o"
c++ "$stage_dir"/*.o -lopenblas -lgfortran -o "$stage_dir/probe"
ldd "$stage_dir/probe" > "$stage_dir/libraries.txt"
OPENBLAS_NUM_THREADS=1 "$stage_dir/probe" > "$stage_dir/results.txt"
sed -n '/^fixture_begin$/,$p' "$stage_dir/results.txt" | tail -n +2 > "$stage_dir/complex-initializer-101.txt"
sha256sum -c "$stage_dir/sources.sha256"
sha256sum "$stage_dir/probe" "$stage_dir/results.txt" "$stage_dir/complex-initializer-101.txt"
