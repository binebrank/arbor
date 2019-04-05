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
namespace detail {

//const unsigned sve_length = svlen_f64(svdup_f64(0.0));

struct sve_double;
struct sve_int;
struct sve_mask;

template <>
struct simd_traits<sve_double> {
    static constexpr unsigned width = SVE_LENGTH;
    using scalar_type = double;
    using vector_type = std::array<double, width>;
    using mask_impl = sve_mask;  // int64x2_t?
};

template <>
struct simd_traits<sve_int> {
    static const unsigned width = SVE_LENGTH;
    using scalar_type = int32_t;
    using vector_type = std::array<int32_t, width>;
    using mask_impl = sve_mask;  // int64x2_t
};

template <>
struct simd_traits<sve_mask> {
    static constexpr unsigned width = SVE_LENGTH;
    using scalar_type = bool;
    using vector_type = std::array<bool, width>;
    using mask_impl = sve_mask;  // int64x2_t
};

struct sve_mask : implbase<sve_mask> {
    using array = std::array<bool, SVE_LENGTH>;
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
        std::copy(w, w+SVE_LENGTH, res.data());
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
    using array = std::array<int32_t, SVE_LENGTH>;
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
	svbool_t half = svwhilelt_b32(0, SVE_LENGTH);
	array res;
	svint32_t veca = svld1(half, a.data());
	svint32_t vec = svneg_z(half, veca);
	svst1(half, res.data(),  vec);
        return res;
    }


    static array add(const array& a, const array& b) {
	svbool_t half = svwhilelt_b32(0, SVE_LENGTH);
	array res;
	svint32_t veca = svld1(half, &a[0]);
	svint32_t vecb = svld1(half, &b[0]);
	svint32_t vec = svadd_z(half, veca, vecb);
	svst1(half, &res[0],  vec);
        return res;
    }

    static array sub(const array& a, const array& b) {
	svbool_t half = svwhilelt_b32(0, SVE_LENGTH);
	array res;
	svint32_t veca = svld1(half, a.data());
	svint32_t vecb = svld1(half, b.data());
	svint32_t vec = svsub_z(half, veca, vecb);
	svst1(half, res.data(),  vec);
        return res;
    }

    static array mul(const array& a, const array& b) {
	svbool_t half = svwhilelt_b32(0, SVE_LENGTH);
	array res;
	svint32_t veca = svld1(half, a.data());
	svint32_t vecb = svld1(half, b.data());
	svint32_t vec = svmul_z(half, veca, vecb);
	svst1(half, res.data(),  vec);
        return res;
    }

    static array div(const array& a, const array& b) {
	svbool_t half = svwhilelt_b32(0, SVE_LENGTH);
	array res;
	svint32_t veca = svld1(half, a.data());
	svint32_t vecb = svld1(half, b.data());
	svint32_t vec = svdiv_z(half, veca, vecb);
	svst1(half, res.data(),  vec);
        return res;
    }

    static array fma(const array& a, const array& b, const array& c) {
	svbool_t half = svwhilelt_b32(0, SVE_LENGTH);
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
	svbool_t half = svwhilelt_b32(0, SVE_LENGTH);
	array res;
	svint32_t veca = svld1(half, a.data());
	svint32_t vec = svabs_z(half, veca);
	svst1(half, res.data(),  vec);
	return res;
    }

    static array min(const array& a, const array& b) {
	svbool_t half = svwhilelt_b32(0, SVE_LENGTH);
	array res;
	svint32_t veca = svld1(half, a.data());
	svint32_t vecb = svld1(half, b.data());
	svint32_t vec = svmin_z(half, veca, vecb);
	svst1(half, res.data(),  vec);
	return res;
    }

    static array max(const array& a, const array& b) {
	svbool_t half = svwhilelt_b32(0, SVE_LENGTH);
	array res;
	svint32_t veca = svld1(half, a.data());
	svint32_t vecb = svld1(half, b.data());
	svint32_t vec = svmax_z(svptrue_b32(), veca, vecb);
	svst1(half, res.data(),  vec);
	return res;
    }


};

struct sve_double : implbase<sve_double> {
    using array = std::array<double, SVE_LENGTH>;
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


    static array exp(const array& x){
	svfloat64_t xvec = svld1_f64(svptrue_b64(), x.data());

	svbool_t large = svcmpgt(svptrue_b64(), xvec, svdup_f64_z(svptrue_b64(), exp_maxarg));
	svbool_t small = svcmplt(svptrue_b64(), xvec, svdup_f64_z(svptrue_b64(), exp_minarg));

	// Compute n and g
	svfloat64_t nvec = svadd_f64_z(svptrue_b64(), svmul_z(svptrue_b64(), xvec, ln2inv), svdup_f64_z(svptrue_b64(), 0.5));
	svint64_t n = svcvt_s64_z(svptrue_b64(), svrintm_z(svptrue_b64(), nvec)); 
	
	// Compute g
	svfloat64_t gvec = svsub_f64_z(svptrue_b64(), xvec, svmul_f64_z(svptrue_b64(), svcvt_f64_z(svptrue_b64(), n), svdup_f64_z(svptrue_b64(), ln2C1)));
	gvec = 	svsub_f64_z(svptrue_b64(), gvec, svmul_f64_z(svptrue_b64(), svcvt_f64_z(svptrue_b64(), n), svdup_f64_z(svptrue_b64(), ln2C2)));
	
	// Compute g*g
	svfloat64_t ggvec = svmul_f64_z(svptrue_b64(), gvec, gvec);

        // Compute the g*P(g^2) and Q(g^2).
	svfloat64_t even = svmul_z(svptrue_b64(), ggvec, Q3exp);
        svfloat64_t odd = svmul_z(svptrue_b64(), ggvec, P2exp);
	even = svadd_z(svptrue_b64(), even, Q2exp);
	odd = svadd_z(svptrue_b64(), odd, P1exp);
	even = svmul_z(svptrue_b64(), even, ggvec);
	odd = svmul_z(svptrue_b64(), odd, ggvec);
	even = svadd_z(svptrue_b64(), even, Q1exp);
	odd = svadd_z(svptrue_b64(), odd, P0exp);
	even = svmul_z(svptrue_b64(), even, ggvec);
	odd = svmul_z(svptrue_b64(), odd, gvec);
	even = svadd_z(svptrue_b64(), even, Q0exp);

        // Compute R(g)/R(-g) = 1 + 2*g*P(g^2) / (Q(g^2)-g*P(g^2))
	svfloat64_t expg = svadd_z(svptrue_b64(), svdup_f64_z(svptrue_b64(), 1.), 
			   svmul_f64_z(svptrue_b64(), svdup_f64_z(svptrue_b64(), 2.),
			   svdiv_f64_z(svptrue_b64(), odd,
			   svsub_f64_z(svptrue_b64(), even, odd))));	
	
        // Compute product with 2^n.
	svfloat64_t expx = svscale_f64_z(svptrue_b64(), expg, n);

	svfloat64_t res = svsel(large, svdup_f64_z(svptrue_b64(), HUGE_VAL),
				svsel(small, svdup_f64_z(svptrue_b64(), 0), expx));
					//check also for NaN
				      
	// Store the result
	array result;
	svst1(svptrue_b64(), result.data(), res);
	return result;
    }

    static array expm1(const array& x){
	svfloat64_t xvec = svld1_f64(svptrue_b64(), x.data());

	svbool_t large = svcmpgt(svptrue_b64(), xvec, svdup_f64_z(svptrue_b64(), exp_maxarg));
	svbool_t small = svcmplt(svptrue_b64(), xvec, svdup_f64_z(svptrue_b64(), exp_minarg));

	svfloat64_t half = svdup_f64_z(svptrue_b64(), 0.5);	
	svint64_t oneint = svdup_s64_z(svptrue_b64(), 1);
	svfloat64_t two = svdup_f64_z(svptrue_b64(), 2.);

	svbool_t nzero = svcmple(svptrue_b64(), svabs_z(svptrue_b64(), xvec), half);
	svfloat64_t nvec = svrintm_z(svptrue_b64(), svadd_f64_z(svptrue_b64(), svmul_z(svptrue_b64(), xvec, ln2inv), half));
	
	nvec = svsel(nzero, svdup_f64_z(svptrue_b64(), 0), nvec);
	svint64_t n = svcvt_s64_z(svptrue_b64(), nvec);

	svfloat64_t gvec = svsub_f64_z(svptrue_b64(), xvec, svmul_f64_z(svptrue_b64(), nvec, svdup_f64_z(svptrue_b64(), ln2C1)));
	gvec = 	svsub_f64_z(svptrue_b64(), gvec, svmul_f64_z(svptrue_b64(), nvec, svdup_f64_z(svptrue_b64(), ln2C2)));
	
	// Compute g*g
	svfloat64_t ggvec = svmul_f64_z(svptrue_b64(), gvec, gvec);
	
	svfloat64_t even = svmul_z(svptrue_b64(), ggvec, Q3exp);
        svfloat64_t odd = svmul_z(svptrue_b64(), ggvec, P2exp);
	even = svadd_z(svptrue_b64(), even, Q2exp);
	odd = svadd_z(svptrue_b64(), odd, P1exp);
	even = svmul_z(svptrue_b64(), even, ggvec);
	odd = svmul_z(svptrue_b64(), odd, ggvec);
	even = svadd_z(svptrue_b64(), even, Q1exp);
	odd = svadd_z(svptrue_b64(), odd, P0exp);
	even = svmul_z(svptrue_b64(), even, ggvec);
	odd = svmul_z(svptrue_b64(), odd, gvec);
	even = svadd_z(svptrue_b64(), even, Q0exp);

        // Compute exp(g)-1 = 2*g*P(g^2) / (Q(g^2)-g*P(g^2))
	svfloat64_t expgm1 = svdiv_f64_z(svptrue_b64(), svmul_z(svptrue_b64(), two, odd),
			   svsub_f64_z(svptrue_b64(), even, odd));	

	svint64_t nm1 = svsub_z(svptrue_b64(), n, oneint);
	

	svint64_t im = svlsl_z(svptrue_b64(), nm1, svdup_u64_z(svptrue_b64(), 52));
	im = svadd_s64_z(svptrue_b64(), im, svlsl_z(svptrue_b64(), svdup_s64_z(svptrue_b64(), 1023), svdup_u64_z(svptrue_b64(), 52)));
	svfloat64_t twoexpm1 = svreinterpret_f64(im);



	svfloat64_t expxm1 = svmul_z(svptrue_b64(), two, 
			 	svadd_z(svptrue_b64(), svscale_z(svptrue_b64(), expgm1, nm1),
					 svsub_z(svptrue_b64(), twoexpm1, half)));


	svfloat64_t res = svsel(large, svdup_f64_z(svptrue_b64(), HUGE_VAL),
				svsel(small, svdup_f64_z(svptrue_b64(), -1),
				      svsel(nzero, expgm1, expxm1)));

	array result;
	svst1(svptrue_b64(), result.data(), res);
	return result;
    }

    static array log(const array& x){
	svfloat64_t xvec = svld1_f64(svptrue_b64(), x.data());
    
	svbool_t large = svcmpge(svptrue_b64(), xvec, svdup_f64_z(svptrue_b64(), HUGE_VAL));
	svbool_t small = svcmplt(svptrue_b64(), xvec, svdup_f64_z(svptrue_b64(), log_minarg));
	svbool_t domainerr = svcmplt(svptrue_b64(), xvec, svdup_f64_z(svptrue_b64(), 0));
	
	std::array<uint64_t, SVE_LENGTH> checknan;
	for (int i = 0; i<SVE_LENGTH; i++)
       		checknan[i] = isnan(x[i]) == 0 ? 0 : 1;
	

	svuint64_t isnan = svld1_u64(svptrue_b64(), checknan.data());
	svbool_t nanan = svcmpeq_u64(svptrue_b64(), isnan, svdup_u64_z(svptrue_b64(), 1));
	domainerr = svorr_z(svptrue_b64(), nanan, domainerr);

	// Compute u
	svuint64_t emask = svdup_u64_z(svptrue_b64(), 0x800fffffffffffff);
	svuint64_t bias = svdup_u64_z(svptrue_b64(), 0x3ff0000000000000);
	svfloat64_t ux = svreinterpret_f64(svorr_z(svptrue_b64(), bias, svand_z(svptrue_b64(), emask, svreinterpret_u64(xvec)))); 
	
	// Compute g
	svuint64_t xwh = svreinterpret_u64(xvec);
        svuint64_t emasked = svdup_u64_z(svptrue_b64(), 0x7ff0000000000000);
	svuint64_t xwhr = svand_z(svptrue_b64(), xwh, emasked);
	xwh = svasr_z(svptrue_b64(), xwhr, svdup_u64_z(svptrue_b64(), 52));
	svfloat64_t gx = svcvt_f64_z(svptrue_b64(), svsub_z(svptrue_b64(), svreinterpret_s64(xwh), svdup_s64_z(svptrue_b64(), 1023)));

	svfloat64_t one = svdup_f64_z(svptrue_b64(), 1.);
	svfloat64_t half = svdup_f64_z(svptrue_b64(), 0.5);

	// Correct u and g
	svbool_t gtsqrt2 = svcmpge(svptrue_b64(), ux, sqrt2);
	svfloat64_t gp1 = svadd_z(svptrue_b64(), gx, one);
	svfloat64_t ud2 = svmul_z(svptrue_b64(), ux, half);
	svfloat64_t g = svsel(gtsqrt2, gp1, gx);
	svfloat64_t u = svsel(gtsqrt2, ud2, ux);

	svfloat64_t z = svsub_z(svptrue_b64(), u, one);

        // Compute the P(z) and Q(z).
	svfloat64_t pz = svmul_z(svptrue_b64(), z, P5log);
	svfloat64_t qz = svadd_z(svptrue_b64(), z, Q4log);
	pz = svadd_z(svptrue_b64(), pz, P4log);
	qz = svmul_z(svptrue_b64(), qz, z);
	pz = svmul_z(svptrue_b64(), pz, z);
	qz = svadd_z(svptrue_b64(), qz, Q3log);
	pz = svadd_z(svptrue_b64(), pz, P3log);
	qz = svmul_z(svptrue_b64(), qz, z);
	pz = svmul_z(svptrue_b64(), pz, z);
	qz = svadd_z(svptrue_b64(), qz, Q2log);
	pz = svadd_z(svptrue_b64(), pz, P2log);
	qz = svmul_z(svptrue_b64(), qz, z);
	pz = svmul_z(svptrue_b64(), pz, z);
	qz = svadd_z(svptrue_b64(), qz, Q1log);
	pz = svadd_z(svptrue_b64(), pz, P1log);
	qz = svmul_z(svptrue_b64(), qz, z);
	pz = svmul_z(svptrue_b64(), pz, z);
	pz = svadd_z(svptrue_b64(), pz, P0log);
	qz = svadd_z(svptrue_b64(), qz, Q0log);

	svfloat64_t z2 = svmul_f64_z(svptrue_b64(), z, z);
	svfloat64_t z3 = svmul_f64_z(svptrue_b64(), z2, z);

	//Compute log
	svfloat64_t log = svdiv_z(svptrue_b64(), svmul_z(svptrue_b64(), z3, pz), qz);
	log = svadd_z(svptrue_b64(), log, svmul_z(svptrue_b64(), g, svdup_f64_z(svptrue_b64(), ln2C4)));
	log = svsub_z(svptrue_b64(), log, svmul_z(svptrue_b64(), z2, half));
	log = svadd_z(svptrue_b64(), log, z);
	log = svadd_z(svptrue_b64(), log, svmul_z(svptrue_b64(), g, svdup_f64_z(svptrue_b64(), ln2C3)));


	svfloat64_t res = svsel(domainerr, svdup_f64_z(svptrue_b64(), NAN),
				svsel(large, svdup_f64_z(svptrue_b64(), HUGE_VAL),
				      svsel(small, svdup_f64_z(svptrue_b64(), -HUGE_VAL), log)));

	array result;
	svst1(svptrue_b64(), result.data(), res);
	return result;

	
    }

};

}  // namespace simd_detail

namespace simd_abi {
template <typename T, unsigned N>
struct sve;

template <>
struct sve<double, SVE_LENGTH> {
    using type = detail::sve_double;
};
template <>
struct sve<int, SVE_LENGTH> {
    using type = detail::sve_int;
};

}  // namespace simd_abi

}  // namespace simd
}  // namespace arb

#endif  // def __ARM_SVE__
