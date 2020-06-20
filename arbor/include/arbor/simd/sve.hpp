#pragma once

// Author: Bine Brank, JSC
// SVE SIMD intrinsics implementation.

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#include <cmath>
#include <cstdint>

#include <iostream>
#include <arbor/simd/approx.hpp>
#include <arbor/simd/implbase.hpp>

#if __ARM_FEATURE_SVE_BITS==SVE_SIZE
typedef svint64_t vec_int64 __attribute__((arm_sve_vector_bits(SVE_SIZE)));
typedef svint32_t vec_int32 __attribute__((arm_sve_vector_bits(SVE_SIZE)));
typedef svuint64_t vec_uint64 __attribute__((arm_sve_vector_bits(SVE_SIZE)));
typedef svfloat64_t vec_double __attribute__((arm_sve_vector_bits(SVE_SIZE)));
typedef svbool_t pred512 __attribute__((arm_sve_vector_bits(SVE_SIZE)));
#endif

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
    using vector_type = vec_double;
    using mask_impl = sve_mask;  // int64x2_t?
};
template <>
struct simd_traits<sve_int> {
    static constexpr unsigned width = SVE_LENGTH;
    using scalar_type = int;
    using vector_type = vec_int32;
    using mask_impl = sve_mask;  // int64x2_t?
};
template <>
struct simd_traits<sve_mask> {
    static constexpr unsigned width = SVE_LENGTH;
    using scalar_type = bool;
    using vector_type = pred512;
    using mask_impl = sve_mask;  // int64x2_t?
};

struct sve_mask : implbase<sve_mask> {
    static void copy_to(const pred512& v, bool* p){
        vec_int64 myvec = svsel_s64(v, svdup_s64(1), svdup_s64(0));
        svst1b_s64(svptrue_b64(), (int8_t *) p, myvec);
    }
    static pred512 copy_from(const bool* p){
        vec_int64 myvec = svld1sb_s64(svptrue_b64(), (int8_t *) p);
        return svcmpgt(svptrue_b64(), myvec, svdup_s64(0));
    } 
    static void mask_copy_to(const pred512& v, bool* p){
        vec_int64 myvec = svsel_s64(v, svdup_s64(1), svdup_s64(0));
        svst1b_s64(svptrue_b64(), (int8_t *) p, myvec);
    }
    static pred512 mask_copy_from(const bool* p){
        vec_int64 myvec = svld1sb_s64(svptrue_b64(), (int8_t *) p);
        return svcmpgt(svptrue_b64(), myvec, svdup_s64(0));
    } 
    static bool mask_element(const pred512& v, int i){
        // we have to conert to svint64_t because there is no dup lane on predicate 
        vec_int64 myvec = svsel_s64(v, svdup_s64(1), svdup_s64(0));
        myvec = svdup_lane_s64(myvec, i);
        pred512 mybool = svcmpgt(svptrue_b64(), myvec, svdup_s64(0));
        return svptest_any(svptrue_b64(), mybool);
    }
    static void mask_set_element(pred512& v, int i, bool x){
        char data[512];
        vec_int64 myvec = svsel_s64(v, svdup_s64(1), svdup_s64(0));
        svst1_s64(svptrue_b64(), (int64_t *) data, myvec);
        // here minus perhaps
        ((int64_t *)data)[i] = (int64_t)x;
        myvec = svld1_s64(svptrue_b64(), (int64_t *) data);
        v = svcmpgt(svptrue_b64(), myvec, svdup_s64(0));
    }
};

struct sve_int : implbase<sve_int> {
    static void copy_to(const vec_int32& v, int* p){
	    pred512 half = svwhilelt_b32(0, SVE_LENGTH);
        svst1_s32(half, p, v);
    }
    static vec_int32 copy_from(const int* p){
	    pred512 half = svwhilelt_b32(0, SVE_LENGTH);
        return svld1_s32(half, p);
    } 
    // Arithmetic operations
    static vec_int32 negate(const vec_int32& a) {
	    pred512 half = svwhilelt_b32(0, SVE_LENGTH);
        return svneg_z(half, a);
    }
    static vec_int32 add(const vec_int32& a, const vec_int32& b) {
	    pred512 half = svwhilelt_b32(0, SVE_LENGTH);
        return svadd_z(half, a, b);
    }
    static vec_int32 sub(const vec_int32& a, const vec_int32& b) {
	    pred512 half = svwhilelt_b32(0, SVE_LENGTH);
        return svsub_z(half, a, b);
    }
    static vec_int32 mul(const vec_int32& a, const vec_int32& b) {
	    pred512 half = svwhilelt_b32(0, SVE_LENGTH);
        return svmul_z(half, a, b);
    }
    static vec_int32 div(const vec_int32& a, const vec_int32& b) {
	    pred512 half = svwhilelt_b32(0, SVE_LENGTH);
        return svdiv_z(half, a, b);
    }
    static vec_int32 fma(const vec_int32& a, const vec_int32& b, const vec_int32& c) {
	    pred512 half = svwhilelt_b32(0, SVE_LENGTH);
        return svmad_z(half, a, b, c);
    }
    static int32_t reduce_add(const vec_int32& a) {
	    pred512 half = svwhilelt_b32(0, SVE_LENGTH);
        return svaddv(half, a);
    }
    static vec_int32 abs(const vec_int32& a) {
	    pred512 half = svwhilelt_b32(0, SVE_LENGTH);
        return svabs_z(half, a);
    }
    static vec_int32 min(const vec_int32& a, const vec_int32& b) {
	    pred512 half = svwhilelt_b32(0, SVE_LENGTH);
        return svmin_z(half, a, b);
    }
    static vec_int32 max(const vec_int32& a, const vec_int32& b) {
	    pred512 half = svwhilelt_b32(0, SVE_LENGTH);
        return svmax_z(half, a, b);
    }
};


struct sve_double : implbase<sve_double> {
    static void copy_to(const vec_double& v, double* p){
        svst1_f64(svptrue_b64(), p, v);
    }
    static void copy_to_masked(const vec_double& v, double* p, const pred512& mask){
        svst1_f64(mask, p, v);
    }
    static vec_double copy_from(const double* p){
        return svld1_f64(svptrue_b64(), p);
    } 
    static vec_double copy_from_masked(const double* p, const pred512& mask){
        return svld1_f64(mask, p);
    } 
    static vec_double copy_from_masked(const vec_double& a, const double* p, const pred512& mask){
        //vec_double b = svadd_z(svptrue_b64(), a, (const vec_double) svdup_f64(0.0)); 
        vec_double loaded = svld1_f64(svptrue_b64(), p);
        return svsel_f64(mask, loaded,  a);
    } 
    static vec_double broadcast(double v) {
        return svdup_f64(v);
    }
    static double element0(const vec_double& a){
        return svlasta(svptrue_b64(), a);
    }
    // Comparisons
    static pred512 cmp_eq(const vec_double& a, const vec_double& b){
        return svcmpeq_f64(svptrue_b64(), a, b);
    }
    static pred512 cmp_neq(const vec_double& a, const vec_double& b){
        return svcmpne_f64(svptrue_b64(), a, b);
    }
    static pred512 cmp_gt(const vec_double& a, const vec_double& b){
        return svcmpgt_f64(svptrue_b64(), a, b);
    }
    static pred512 cmp_geq(const vec_double& a, const vec_double& b){
        return svcmpge_f64(svptrue_b64(), a, b);
    }
    static pred512 cmp_lt(const vec_double& a, const vec_double& b){
        return svcmplt_f64(svptrue_b64(), a, b);
    }
    static pred512 cmp_leq(const vec_double& a, const vec_double& b){
        return svcmple_f64(svptrue_b64(), a, b);
    }
    static vec_double ifelse(const pred512& m, const vec_double& a, const vec_double& b){
        return svsel_f64(m, a, b);
    }
    // Arithmetic operations
    static vec_double negate(const vec_double& a) {
        return svneg_z(svptrue_b64(), a);
    }
    static vec_double add(const vec_double& a, const vec_double& b) {
        return svadd_z(svptrue_b64(), a, b);
    }
    static vec_double sub(const vec_double& a, const vec_double& b) {
        return svsub_z(svptrue_b64(), a, b);
    }
    static vec_double mul(const vec_double& a, const vec_double& b) {
        return svmul_z(svptrue_b64(), a, b);
    }
    static vec_double div(const vec_double& a, const vec_double& b) {
        return svdiv_z(svptrue_b64(), a, b);
    }
    static vec_double fma(const vec_double& a, const vec_double& b, const vec_double& c) {
        return svmad_z(svptrue_b64(), a, b, c);
    }
    static double reduce_add(const vec_double& a) {
        return svaddv(svptrue_b64(), a);
    }
    static vec_double abs(const vec_double& a) {
        return svabs_z(svptrue_b64(), a);
    }
    static vec_double min(const vec_double& a, const vec_double& b) {
        return svmin_z(svptrue_b64(), a, b);
    }
    static vec_double max(const vec_double& a, const vec_double& b) {
        return svmax_z(svptrue_b64(), a, b);
    }

    static vec_double exp(const vec_double& x){
	pred512 large = svcmpgt(svptrue_b64(), x, svdup_f64_z(svptrue_b64(), exp_maxarg));
	pred512 small = svcmplt(svptrue_b64(), x, svdup_f64_z(svptrue_b64(), exp_minarg));

	// Compute n and g
	vec_double nvec = svadd_f64_z(svptrue_b64(), svmul_z(svptrue_b64(), x, ln2inv), svdup_f64_z(svptrue_b64(), 0.5));
	vec_int64 n = svcvt_s64_z(svptrue_b64(), svrintm_z(svptrue_b64(), nvec)); 
	
	// Compute g
	vec_double gvec = svsub_f64_z(svptrue_b64(), x, svmul_f64_z(svptrue_b64(), svcvt_f64_z(svptrue_b64(), n), svdup_f64_z(svptrue_b64(), ln2C1)));
	gvec = 	svsub_f64_z(svptrue_b64(), gvec, svmul_f64_z(svptrue_b64(), svcvt_f64_z(svptrue_b64(), n), svdup_f64_z(svptrue_b64(), ln2C2)));
	
	// Compute g*g
	vec_double ggvec = svmul_f64_z(svptrue_b64(), gvec, gvec);

        // Compute the g*P(g^2) and Q(g^2).
	vec_double even = svmul_z(svptrue_b64(), ggvec, Q3exp);
    vec_double odd = svmul_z(svptrue_b64(), ggvec, P2exp);
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
    vec_double expg = svadd_z(svptrue_b64(), svdup_f64_z(svptrue_b64(), 1.), 
            svmul_f64_z(svptrue_b64(), svdup_f64_z(svptrue_b64(), 2.),
                svdiv_f64_z(svptrue_b64(), odd,
                    svsub_f64_z(svptrue_b64(), even, odd))));	

    // Compute product with 2^n.
	svfloat64_t expx = svscale_f64_z(svptrue_b64(), expg, n);

	return svsel(large, svdup_f64_z(svptrue_b64(), HUGE_VAL),
				svsel(small, svdup_f64_z(svptrue_b64(), 0), expx));
    }

    static vec_double expm1(const vec_double& x){
	pred512 large = svcmpgt(svptrue_b64(), x, svdup_f64_z(svptrue_b64(), exp_maxarg));
	pred512 small = svcmplt(svptrue_b64(), x, svdup_f64_z(svptrue_b64(), exp_minarg));

	vec_double half = svdup_f64_z(svptrue_b64(), 0.5);	
	vec_int64 oneint = svdup_s64_z(svptrue_b64(), 1);
	vec_double two = svdup_f64_z(svptrue_b64(), 2.);

	pred512 nzero = svcmple(svptrue_b64(), svabs_z(svptrue_b64(), x), half);
	vec_double nvec = svrintm_z(svptrue_b64(), svadd_f64_z(svptrue_b64(), svmul_z(svptrue_b64(), x, ln2inv), half));
	
	nvec = svsel(nzero, svdup_f64_z(svptrue_b64(), 0), nvec);
	vec_int64 n = svcvt_s64_z(svptrue_b64(), nvec);

	vec_double gvec = svsub_f64_z(svptrue_b64(), x, svmul_f64_z(svptrue_b64(), nvec, svdup_f64_z(svptrue_b64(), ln2C1)));
	gvec = 	svsub_f64_z(svptrue_b64(), gvec, svmul_f64_z(svptrue_b64(), nvec, svdup_f64_z(svptrue_b64(), ln2C2)));
	
	// Compute g*g
	vec_double ggvec = svmul_f64_z(svptrue_b64(), gvec, gvec);

    vec_double even = svmul_z(svptrue_b64(), ggvec, Q3exp);
    vec_double odd = svmul_z(svptrue_b64(), ggvec, P2exp);
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
	vec_double expgm1 = svdiv_f64_z(svptrue_b64(), svmul_z(svptrue_b64(), two, odd),
			   svsub_f64_z(svptrue_b64(), even, odd));	

	vec_int64 nm1 = svsub_z(svptrue_b64(), n, oneint);

	vec_int64 im = svlsl_z(svptrue_b64(), nm1, svdup_u64_z(svptrue_b64(), 52));
	im = svadd_s64_z(svptrue_b64(), im, svlsl_z(svptrue_b64(), svdup_s64_z(svptrue_b64(), 1023), svdup_u64_z(svptrue_b64(), 52)));
	vec_double twoexpm1 = svreinterpret_f64(im);

	vec_double expxm1 = svmul_z(svptrue_b64(), two, 
			 	svadd_z(svptrue_b64(), svscale_z(svptrue_b64(), expgm1, nm1), svsub_z(svptrue_b64(), twoexpm1, half)));

	return svsel(large, svdup_f64_z(svptrue_b64(), HUGE_VAL),
				svsel(small, svdup_f64_z(svptrue_b64(), -1),
				      svsel(nzero, expgm1, expxm1)));
    }

    static vec_double log(const vec_double& x){
	pred512 large = svcmpge(svptrue_b64(), x, svdup_f64_z(svptrue_b64(), HUGE_VAL));
	pred512 small = svcmplt(svptrue_b64(), x, svdup_f64_z(svptrue_b64(), log_minarg));
	pred512 domainerr = svcmplt(svptrue_b64(), x, svdup_f64_z(svptrue_b64(), 0));
	
    pred512 nanerr = svnot_z(svptrue_b64(), svcmpeq_f64(svptrue_b64(), x, x));
	/* std::array<uint64_t, 8> checknan; */
	/* for (int i = 0; i<8; i++) */
       		/* checknan[i] = isnan(x[i]) == 0 ? 0 : 1; */

	/* vec_uint64 isnan = svld1_u64(svptrue_b64(), checknan.data()); */
	/* pred512 nanan = svcmpeq_u64(svptrue_b64(), isnan, svdup_u64_z(svptrue_b64(), 1)); */
	domainerr = svorr_z(svptrue_b64(), nanerr, domainerr);

	// Compute u
	vec_uint64 emask = svdup_u64_z(svptrue_b64(), 0x800fffffffffffff);
	vec_uint64 bias = svdup_u64_z(svptrue_b64(), 0x3ff0000000000000);
	vec_double ux = svreinterpret_f64(svorr_z(svptrue_b64(), bias, svand_z(svptrue_b64(), emask, svreinterpret_u64(x)))); 
	
	// Compute g
    // here is the problem
	vec_int64 xwh = svreinterpret_s64(x);
        svint64_t emasked = svdup_s64_z(svptrue_b64(), 0x7ff0000000000000);
	vec_int64 xwh_r = svand_s64_z(svptrue_b64(), xwh, emasked);
	xwh = svasr_s64_z(svptrue_b64(), xwh_r, svdup_u64_z(svptrue_b64(), 52));
	vec_double gx = svcvt_f64_z(svptrue_b64(), svsub_z(svptrue_b64(), svreinterpret_s64(xwh), svdup_s64_z(svptrue_b64(), 1023)));

	vec_double one = svdup_f64_z(svptrue_b64(), 1.);
	vec_double half = svdup_f64_z(svptrue_b64(), 0.5);

	// Correct u and g
	pred512 gtsqrt2 = svcmpge(svptrue_b64(), ux, sqrt2);
	vec_double gp1 = svadd_z(svptrue_b64(), gx, one);
	vec_double ud2 = svmul_z(svptrue_b64(), ux, half);
	vec_double g = svsel(gtsqrt2, gp1, gx);
	vec_double u = svsel(gtsqrt2, ud2, ux);

	vec_double z = svsub_z(svptrue_b64(), u, one);

        // Compute the P(z) and Q(z).
	vec_double pz = svmul_z(svptrue_b64(), z, P5log);
	vec_double qz = svadd_z(svptrue_b64(), z, Q4log);
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

	vec_double z2 = svmul_f64_z(svptrue_b64(), z, z);
	vec_double z3 = svmul_f64_z(svptrue_b64(), z2, z);

	//Compute log
	vec_double log = svdiv_z(svptrue_b64(), svmul_z(svptrue_b64(), z3, pz), qz);
	log = svadd_z(svptrue_b64(), log, svmul_z(svptrue_b64(), g, svdup_f64_z(svptrue_b64(), ln2C4)));
	log = svsub_z(svptrue_b64(), log, svmul_z(svptrue_b64(), z2, half));
	log = svadd_z(svptrue_b64(), log, z);
	log = svadd_z(svptrue_b64(), log, svmul_z(svptrue_b64(), g, svdup_f64_z(svptrue_b64(), ln2C3)));


	return svsel(domainerr, svdup_f64_z(svptrue_b64(), NAN),
				svsel(large, svdup_f64_z(svptrue_b64(), HUGE_VAL),
				      svsel(small, svdup_f64_z(svptrue_b64(), -HUGE_VAL), log)));
    }


};

}  // namespace simd_detail

namespace simd_abi {
template <typename T, unsigned N> struct sve;
template <> struct sve<double, SVE_LENGTH> { using type = detail::sve_double; };
template <> struct sve<int, SVE_LENGTH> { using type = detail::sve_int; };

}  // namespace simd_abi

}  // namespace simd
}  // namespace arb

#endif  // def __ARM_SVE__
