# Private scalar ports of GNU GCC13.3 libgcc2.c L_divdc3 and LLVM17
# Modified work: private Julia translation and platform dispatch, 2026-10-04.
# compiler-rt divdc3.c. GNU: GPL-3.0-or-later WITH GCC-exception-3.1;
# Copyright (C) 1989-2023 Free Software Foundation, Inc.
# LLVM: Apache-2.0 WITH LLVM-exception. Source-adjacent licenses/NOTICE
# retain origins and distinguish this Julia translation from Rust validation.
_ordinary_complex_uses_gnu(os, libc) = os == "linux" && libc == "glibc"

function _ordinary_complex_divide_gnu(z::ComplexF64, w::ComplexF64)
    a,b = reim(z); c,d = reim(w)
    big = floatmax(Float64)/2.0
    small = floatmin(Float64)
    threshold = eps(Float64)
    scale_up = 1.0/threshold
    numerator_limit = big*threshold
    imaginary_dominant = abs(c) < abs(d)
    largest = imaginary_dominant ? abs(d) : abs(c)
    if largest >= big
        a /= 2.0; b /= 2.0; c /= 2.0; d /= 2.0
    end
    largest = imaginary_dominant ? abs(d) : abs(c)
    if largest < threshold ||
        ((abs(a) < small && abs(b) < numerator_limit && largest < numerator_limit) ||
         (abs(b) < small && abs(a) < numerator_limit && largest < numerator_limit))
        a *= scale_up; b *= scale_up; c *= scale_up; d *= scale_up
    end
    result = if imaginary_dominant
        ratio = c/d
        denominator = c*ratio+d
        if abs(ratio) > small
            complex((a*ratio+b)/denominator, (b*ratio-a)/denominator)
        else
            complex((c*(a/d)+b)/denominator, (c*(b/d)-a)/denominator)
        end
    else
        ratio = d/c
        denominator = d*ratio+c
        if abs(ratio) > small
            complex((b*ratio+a)/denominator, (b-a*ratio)/denominator)
        else
            complex((a+d*(b/c))/denominator, (b-d*(a/c))/denominator)
        end
    end
    if isnan(real(result)) && isnan(imag(result))
        if c == 0.0 && d == 0.0 && (!isnan(a) || !isnan(b))
            infinity = copysign(Inf,c)
            result = complex(infinity*a,infinity*b)
        elseif (isinf(a) || isinf(b)) && isfinite(c) && isfinite(d)
            a = copysign(Float64(isinf(a)),a); b = copysign(Float64(isinf(b)),b)
            result = complex(Inf*(a*c+b*d),Inf*(b*c-a*d))
        elseif (isinf(c) || isinf(d)) && isfinite(a) && isfinite(b)
            c = copysign(Float64(isinf(c)),c); d = copysign(Float64(isinf(d)),d)
            result = complex(0.0*(a*c+b*d),0.0*(b*c-a*d))
        end
    end
    return result
end

function _ordinary_complex_scale(value::Float64, power::Int)
    # Multiplications preserve the reviewed finite/subnormal scaling order;
    # no math library, reciprocal, fastmath or new FFI is introduced.
    while power < -1022
        value *= reinterpret(Float64, UInt64(54)<<52) # 2^-969
        power += 969
    end
    while power > 1023
        value *= reinterpret(Float64, UInt64(2046)<<52) # 2^1023
        power -= 1023
    end
    return value*reinterpret(Float64,UInt64(power+1023)<<52)
end

function _ordinary_complex_divide_llvm(z::ComplexF64, w::ComplexF64; arm_fma::Bool=false)
    a,b = reim(z); c,d = reim(w)
    # C fmax/ Rust f64::max chooses the non-NaN operand when only one is NaN.
    magnitude = isnan(abs(c)) ? abs(d) : isnan(abs(d)) ? abs(c) : max(abs(c),abs(d))
    power = 0
    if isfinite(magnitude) && magnitude != 0.0
        bits = reinterpret(UInt64,magnitude)
        exponent_bits = Int((bits>>52)&0x7ff)
        power = exponent_bits == 0 ? 63-leading_zeros(bits)-1074 : exponent_bits-1023
        c = _ordinary_complex_scale(c,-power)
        d = _ordinary_complex_scale(d,-power)
    end
    product_sum(x,y,z) = arm_fma ? fma(x,y,z) : x*y+z
    denominator = product_sum(c,c,d*d)
    result = complex(
        _ordinary_complex_scale(product_sum(a,c,b*d)/denominator,-power),
        _ordinary_complex_scale(product_sum(b,c,-(a*d))/denominator,-power))
    if isnan(real(result)) && isnan(imag(result))
        if denominator == 0.0 && (!isnan(a) || !isnan(b))
            infinity = copysign(Inf,c)
            result = complex(infinity*a,infinity*b)
        elseif (isinf(a) || isinf(b)) && isfinite(c) && isfinite(d)
            a = copysign(Float64(isinf(a)),a); b = copysign(Float64(isinf(b)),b)
            result = complex(Inf*product_sum(a,c,b*d),Inf*product_sum(b,c,-(a*d)))
        elseif isinf(magnitude) && isfinite(a) && isfinite(b)
            c = copysign(Float64(isinf(c)),c); d = copysign(Float64(isinf(d)),d)
            result = complex(0.0*product_sum(a,c,b*d),0.0*product_sum(b,c,-(a*d)))
        end
    end
    return result
end

function _ordinary_complex_divide(z::ComplexF64, w::ComplexF64)
    # Linux-musl must not silently select libgcc's GNU/glibc contract.
    @static if _ordinary_complex_uses_gnu(
        Base.BinaryPlatforms.os(Base.BinaryPlatforms.HostPlatform()),
        Base.BinaryPlatforms.libc(Base.BinaryPlatforms.HostPlatform()))
        return _ordinary_complex_divide_gnu(z,w)
    else
        return _ordinary_complex_divide_llvm(z,w;
            arm_fma=Sys.isapple() && Sys.ARCH == :aarch64)
    end
end
