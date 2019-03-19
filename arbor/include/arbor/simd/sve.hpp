#pragma once

// SVE SIMD intrinsics implementation.

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#include <cmath>
#include <cstdint>

#include <iostream>
#include <arbor/simd/approx.hpp>
#include <arbor/simd/implbase.hpp>

namespace arb {
namespace simd {
namespace simd_detail {

//const unsigned sve_length = svlen_f64(svdup_f64(0.0));

struct sve_double;
struct sve_int;
struct sve_mask;

template <>
struct simd_traits<sve_double> {
    static constexpr unsigned width = 4;
    using scalar_type = double;
    using vector_type = std::array<double, width>;
    using mask_impl = sve_mask;  // int64x2_t?
};

template <>
struct simd_traits<sve_int> {
    static const unsigned width = 4;
    using scalar_type = int32_t;
    using vector_type = std::array<int32_t, width>;
    using mask_impl = sve_mask;  // int64x2_t
};

template <>
struct simd_traits<sve_mask> {
    static constexpr unsigned width = 4;
    using scalar_type = bool;
    using vector_type = std::array<bool, width>;
    using mask_impl = sve_mask;  // int64x2_t
};

struct sve_mask : implbase<sve_mask> {
    using array = std::array<bool, 4>;
    // Use default implementations for:
    //     element, set_element, div.
    
    static void copy_to(const array& v, bool* p) {
	    std::memcpy(p, &v, sizeof(v));
    }

    static array copy_from(const bool* p) {
	array res;
	std::memcpy(&res, p, sizeof(res));
	return res; }

    static void mask_copy_to(const array& m, bool* y) {
	std::copy(m.begin(), m.end(), y);
    }

    static array mask_copy_from(const bool* w) {
        array res;
        std::copy(w, w+4, res.data());
        return res;
    }

    static bool mask_element(const array& u, int i) {
        return static_cast<bool>(u[i]);
    }

    static void mask_set_element(array& u, int i, bool b) {
	u[i] = b;
    }
};

struct sve_int : implbase<sve_int> {
    using array = std::array<int32_t, 4>;
    using boolarray = std::array<bool, 4>;
    // Use default implementations for:
    //     element, set_element, div.

    static void copy_to(const array& v, int32_t* p) {
	    std::memcpy(p, &v, sizeof(v));
    }

    static array copy_from(const int32_t* p) {
	array res;
	std::memcpy(&res, p, sizeof(res));
	return res; }


    // Arithmetic operations
    static array negate(const array& a) {
	svbool_t half = svwhilelt_b32(0, 4);
	array res;
	svint32_t vec = svld1(half, a.data());
	vec = svneg_x(half, vec);
	svst1(half, res.data(),  vec);
        return res;
    }

    static array add(const array& a, const array& b) {
	svbool_t half = svwhilelt_b32(0, 4);
	array res;
	svint32_t veca = svld1(half, a.data());
	svint32_t vecb = svld1(half, b.data());
	svint32_t vec = svadd_z(half, veca, vecb);
	svst1(half, res.data(),  vec);
        return res;
    }

    static array sub(const array& a, const array& b) {
	svbool_t half = svwhilelt_b32(0, 4);
	array res;
	svint32_t veca = svld1(half, a.data());
	svint32_t vecb = svld1(half, b.data());
	svint32_t vec = svsub_z(half, veca, vecb);
	svst1(half, res.data(),  vec);
        return res;
    }

    static array mul(const array& a, const array& b) {
	svbool_t half = svwhilelt_b32(0, 4);
	array res;
	svint32_t veca = svld1(half, a.data());
	svint32_t vecb = svld1(half, b.data());
	svint32_t vec = svmul_z(half, veca, vecb);
	svst1(half, res.data(),  vec);
        return res;
    }

    static array div(const array& a, const array& b) {
	svbool_t half = svwhilelt_b32(0, 4);
	array res;
	svint32_t veca = svld1(half, a.data());
	svint32_t vecb = svld1(half, b.data());
	svint32_t vec = svdiv_z(half, veca, vecb);
	svst1(half, res.data(),  vec);
        return res;
    }

    static array fma(const array& a, const array& b, const array& c) {
	svbool_t half = svwhilelt_b32(0, 4);
	array res;
	svint32_t veca = svld1(half, a.data());
	svint32_t vecb = svld1(half, b.data());
	svint32_t vecc = svld1(half, c.data());
	svint32_t vec = svmad_z(half, veca, vecb, vecc);
	svst1(half, res.data(),  vec);
        return res;
    }

    // Mathematical functions
    static array abs(const array& a) {
	svbool_t half = svwhilelt_b32(0, 4);
	array res;
	svint32_t veca = svld1(half, a.data());
	svint32_t vec = svabs_z(half, veca);
	svst1(half, res.data(),  vec);
	return res;
    }

    static array min(const array& a, const array& b) {
	svbool_t half = svwhilelt_b32(0, 4);
	array res;
	svint32_t veca = svld1(half, a.data());
	svint32_t vecb = svld1(half, b.data());
	svint32_t vec = svmin_z(half, veca, vecb);
	svst1(half, res.data(),  vec);
	return res;
    }

    static array max(const array& a, const array& b) {
	svbool_t half = svwhilelt_b32(0, 4);
	array res;
	svint32_t veca = svld1(half, a.data());
	svint32_t vecb = svld1(half, b.data());
	svint32_t vec = svmax_z(svptrue_b32(), veca, vecb);
	svst1(half, res.data(),  vec);
	return res;
    }

};

struct sve_double : implbase<sve_double> {
    using array = std::array<double, 4>;
    // Use default implementations for:
    //     element, set_element, div.

    static void copy_to(const array& v, double* p) {
	    std::memcpy(p, &v, sizeof(v));
    }

    static array copy_from(const double* p) {
	array res;
	std::memcpy(&res, p, sizeof(res));
	return res; }

    // Arithmetic operations
    static array negate(const array& a) {
	array res;
	svfloat64_t vec = svld1(svptrue_b64(), a.data());
	vec = svneg_x(svptrue_b64(), vec);
	svst1(svptrue_b64(), res.data(),  vec);
        return res;
    }

    static array add(const array& a, const array& b) {
	array res;
	svfloat64_t veca = svld1(svptrue_b64(), a.data());
	svfloat64_t vecb = svld1(svptrue_b64(), b.data());
	svfloat64_t vec = svadd_z(svptrue_b64(), veca, vecb);
	svst1(svptrue_b64(), res.data(),  vec);
        return res;
    }

    static array sub(const array& a, const array& b) {
	array res;
	svfloat64_t veca = svld1(svptrue_b64(), a.data());
	svfloat64_t vecb = svld1(svptrue_b64(), b.data());
	svfloat64_t vec = svsub_z(svptrue_b64(), veca, vecb);
	svst1(svptrue_b64(), res.data(),  vec);
        return res;
    }

    static array mul(const array& a, const array& b) {
	array res;
	svfloat64_t veca = svld1(svptrue_b64(), a.data());
	svfloat64_t vecb = svld1(svptrue_b64(), b.data());
	svfloat64_t vec = svmul_z(svptrue_b64(), veca, vecb);
	svst1(svptrue_b64(), res.data(),  vec);
        return res;
    }

    static array div(const array& a, const array& b) {
	array res;
	svfloat64_t veca = svld1(svptrue_b64(), a.data());
	svfloat64_t vecb = svld1(svptrue_b64(), b.data());
	svfloat64_t vec = svdiv_z(svptrue_b64(), veca, vecb);
	svst1(svptrue_b64(), res.data(),  vec);
        return res;
    }

    static array fma(const array& a, const array& b, const array& c) {
	array res;
	svfloat64_t veca = svld1(svptrue_b64(), a.data());
	svfloat64_t vecb = svld1(svptrue_b64(), b.data());
	svfloat64_t vecc = svld1(svptrue_b64(), c.data());
	svfloat64_t vec = svmad_z(svptrue_b64(), veca, vecb, vecc);
	svst1(svptrue_b64(), res.data(),  vec);
        return res;
    }

    static double reduce_add(const array& a) {
	svfloat64_t veca = svld1(svptrue_b64(), a.data());
	return svaddv(svptrue_b64(), veca);
    }



};

}  // namespace simd_detail

namespace simd_abi {
template <typename T, unsigned N>
struct sve;

template <>
struct sve<double, 4> {
    using type = simd_detail::sve_double;
};
template <>
struct sve<int, 4> {
    using type = simd_detail::sve_int;
};

}  // namespace simd_abi

}  // namespace simd
}  // namespace arb

#endif  // def __ARM_SVE__
