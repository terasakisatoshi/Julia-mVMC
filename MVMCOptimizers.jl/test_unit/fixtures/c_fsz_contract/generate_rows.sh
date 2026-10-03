#!/usr/bin/env bash
set -euo pipefail
probe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
native_dir=${1:?absolute path to original mVMC-1.3.0/src}
stage_dir=$(mktemp -d /tmp/mvmc-issue176-fsz.XXXXXX)
printf 'stage=%s\n' "$stage_dir"
exec > >(tee "$stage_dir/generation.log") 2>&1
set -x
printf '%s  %s\n' 849488176375102fbc3bab7001e0dc27dfa7c8e049fd90ee1d6248ebfba10764 "$native_dir/mVMC/matrix.c" | sha256sum -c -
# Mechanical verbatim extraction, including the upstream license.
sed -n '1,21p;125,179p;223,278p' "$native_dir/mVMC/matrix.c" > "$stage_dir/children.inc"
cmp "$stage_dir/children.inc" "$probe_dir/children.inc"
c++ --version > "$stage_dir/environment.txt"
cc --version >> "$stage_dir/environment.txt"
gfortran --version >> "$stage_dir/environment.txt"
uname -sm >> "$stage_dir/environment.txt"
sha256sum "$probe_dir/rows_probe.c" "$probe_dir/generate_rows.sh" "$native_dir/mVMC/matrix.c" "$stage_dir/children.inc" > "$stage_dir/source-sha256.txt"
for family in d z; do
  for name in sktrf sktf2 lasktrf skr2 skr2k; do
    source="$native_dir/pfapack/fortran/${family}${name}.f"
    sha256sum "$source" >> "$stage_dir/source-sha256.txt"
    gfortran -O0 -ffp-contract=off -c "$source" -o "$stage_dir/${family}${name}.o"
  done
done
for name in ltl2inv ilaenv_lauum; do
  c++ -std=c++11 -O0 -ffp-contract=off -DBLAS_EXTERNAL -I"$native_dir/common" -I"$native_dir/common/deps" -I"$native_dir/ltl2inv" -c "$native_dir/ltl2inv/$name.cc" -o "$stage_dir/$name.o"
done
sha256sum "$native_dir"/ltl2inv/{ltl2inv.cc,invert.tcc,pfaffian.tcc,trmmt.tcc,ilaenv_lauum.cc,ilaenv_lauum.hh,ilaenv.h,ilaenv_wrap.f90} "$native_dir"/common/{blalink.hh,blalink_fort.h,colmaj.hh,deps/blis.h} >> "$stage_dir/source-sha256.txt"
gfortran -O0 -ffp-contract=off -J"$stage_dir" -c "$native_dir/ltl2inv/ilaenv_wrap.f90" -o "$stage_dir/ilaenv_wrap.o"
cc -std=c11 -O0 -ffp-contract=off -Wno-unknown-pragmas -I"$stage_dir" -c "$probe_dir/rows_probe.c" -o "$stage_dir/probe.o"
c++ "$stage_dir"/*.o -lopenblas -lgfortran -o "$stage_dir/probe"
ldd "$stage_dir/probe" > "$stage_dir/libraries.txt"
OPENBLAS_NUM_THREADS=1 "$stage_dir/probe" | tee "$stage_dir/rows.txt"
test "$(wc -l < "$stage_dir/rows.txt")" -eq 16
cmp <(head -14 "$stage_dir/rows.txt") "$probe_dir/rows.txt"
cmp <(tail -2 "$stage_dir/rows.txt") "$probe_dir/sum-overflow.txt"
sha256sum "$stage_dir/probe" "$stage_dir/rows.txt"
sha256sum -c "$stage_dir/source-sha256.txt"
