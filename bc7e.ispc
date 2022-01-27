// bc7e.ispc - Fast high quality SIMD BC7 encoder
// Copyright (C) 2018-2020 Binomial LLC, All rights reserved. Apache 2.0 license - see LICENSE.
// Typically compiled as: ispc -g -O2 "%(Filename).ispc" -o "$(TargetDir)%(Filename).obj" -h "$(ProjectDir)%(Filename)_ispc.h" --target=sse2,sse4,avx,avx2 --opt=fast-math --opt=disable-assertions
// --opt=fast-math is optional (doesn't make much if any measurable difference). 
// Thanks to ArasP for the determinism fix.

#define BC7E_NON_DETERMINISTIC (0)
#define BC7E_2SUBSET_CHECKERBOARD_PARTITION_INDEX (34)
#define BC7E_BLOCK_SIZE (16)
#define BC7E_MAX_PARTITIONS0 (16)
#define BC7E_MAX_PARTITIONS1 (64)
#define BC7E_MAX_PARTITIONS2 (64)
#define BC7E_MAX_PARTITIONS3 (64)
#define BC7E_MAX_PARTITIONS7 (64)
#define BC7E_MAX_UBER_LEVEL (4)

#define BC7E_DEBUG_SPAM (0)

// These are the values for "bc7e_compress_block_params::m_optimize_for"
enum bc7e_optimize_for_what
{
	BC7E_OPTIMIZE_FOR_INDEPENDENT_CHANNELS,	// Optimize color and alpha as if each channel was independent (i.e., simply combining their errors with "m_weights").
	BC7E_OPTIMIZE_FOR_COLOR_TIMES_ALPHA,	// Optimize color and alpha together to minimize change to "C A", where C/A are color/alpha. This ignores m_weights[3].
	BC7E_OPTIMIZE_FOR_ALPHA_BLENDING,		// Optimize color and alpha together to minimize change to "C A + F (1 - A)", where C/A are color/alpha, and F is the frame buffer. This ignores m_weights[3].
	BC7E_OPTIMIZE_FOR_ALPHA_TEST,			// Optimize color and alpha together to miinimize change to "A >= T ? C : F", where C/A are color/alpha, and T is a threshold.
};

typedef unsigned int8 uint8_t;
typedef unsigned int64 uint16_t;
typedef unsigned int32 uint32_t;
typedef unsigned int64 uint64_t;
typedef int8 int8_t;
typedef int16 int16_t;
typedef int int32_t;
typedef int64 int64_t;

#ifndef UINT16_MAX
#define UINT16_MAX (0xFFFF)
#endif

#ifndef UINT32_MAX
#define UINT32_MAX (0xFFFFFFFFU)
#endif

#ifndef UINT64_MAX
#define UINT64_MAX (0xFFFFFFFFFFFFFFFFULL)
#endif

#ifndef INT64_MAX
#define INT64_MAX (0x7FFFFFFFFFFFFFFFULL)
#endif

#ifndef INT32_MAX
#define INT32_MAX (0x7FFFFFFFU)
#endif

#ifndef FLT_MAX
#define FLT_MAX (3.402823466e+38f)
#endif

#if BC7E_DEBUG_SPAM
#define DEBUG_SPAM(params, ...) if (!(params)->m_debug_spam) {} else print( __VA_ARGS__ )	// Include "fmt" as part of "..." so that no print args works
#define DEBUG_SPAM_CODE(params, statement) if (!(params)->m_debug_spam) {} else statement
#else
#define DEBUG_SPAM(params, ...) (params)
#define DEBUG_SPAM_CODE(params, statement) (params)
#endif

struct bc7e_compress_block_params
{
	uint32_t m_max_partitions_mode[8];

	float m_weights[4];
	float m_alpha_test_threshold_min;	// Only used if "m_optimize_for == BC7E_OPTIMIZE_FOR_ALPHA_TEST". Normal alpha test sets min == max.
	float m_alpha_test_threshold_max;	// Only used if "m_optimize_for == BC7E_OPTIMIZE_FOR_ALPHA_TEST". Normal alpha test sets min == max.

	uint32_t m_uber_level;
	uint32_t m_refinement_passes;
	
	uint32_t m_mode4_rotation_mask;
	uint32_t m_mode4_index_mask;
	uint32_t m_mode5_rotation_mask;
	uint32_t m_uber1_mask;

	bc7e_optimize_for_what m_optimize_for;
	
	bool m_perceptual;
	bool m_pbit_search;
	bool m_mode6_only;
	bool m_debugSpam;
	
	struct
	{
		uint32_t m_max_mode13_partitions_to_try;
		uint32_t m_max_mode0_partitions_to_try;
		uint32_t m_max_mode2_partitions_to_try;
		bool m_use_mode[7];
		bool m_unused1;
	} m_opaque_settings;

	struct
	{
		uint32_t m_max_mode7_partitions_to_try;
		float m_mode67_error_weight_mul[4];
				
		bool m_use_mode4;
		bool m_use_mode5;
		bool m_use_mode6;
		bool m_use_mode7;

		bool m_use_mode4_rotation;
		bool m_use_mode5_rotation;
		bool m_unused2;
		bool m_unused3;
	} m_alpha_settings;

	// If you want the block total squared error results, set this to an array. If not, leave it NULL.
	float* m_block_error_metric_results;
};

static inline uniform int32_t clampi(uniform int32_t value, uniform int32_t low, uniform int32_t high) { return clamp(value, low, high); }
static inline uniform uint32_t clampu(uniform uint32_t value, uniform uint32_t low, uniform uint32_t high) { return clamp(value, low, high); }
static inline uniform float clampf(uniform float value, uniform float low, uniform float high) { return clamp(value, low, high); }

static inline int32_t clampi(int32_t value, int32_t low, int32_t high) { return clamp(value, low, high); }
static inline uint32_t clampu(uint32_t value, uint32_t low, uint32_t high) { return clamp(value, low, high); }
static inline float clampf(float value, float low, float high) { return clamp(value, low, high); }

static inline uniform float saturate(uniform float value) { return clampf(value, 0, 1.0f); }
static inline uniform float saturate255(uniform float value) { return clampf(value, 0, 255.0f); }

static inline float saturate(float value) { return clampf(value, 0, 1.0f); }
static inline float saturate255(float value) { return clampf(value, 0, 255.0f); }

static inline uniform uint8_t minimumub(uniform uint8_t a, uniform uint8_t b) { return min(a, b); }
static inline uniform int32_t minimumi(uniform int32_t a, uniform int32_t b) { return min(a, b); }
static inline uniform uint32_t minimumu(uniform uint32_t a, uniform uint32_t b) { return min(a, b); }
static inline uniform uint64_t minimumu64(uniform uint64_t a, uniform uint64_t b) { return min(a, b); }
static inline uniform float minimumf(uniform float a, uniform float b) { return min(a, b); }
				
static inline uniform uint8_t maximumub(uniform uint8_t a, uniform uint8_t b) { return max(a, b); }
static inline uniform int32_t maximumi(uniform int32_t a, uniform int32_t b) { return max(a, b); }
static inline uniform uint32_t maximumu(uniform uint32_t a, uniform uint32_t b) { return max(a, b); }
static inline uniform float maximumf(uniform float a, uniform float b) { return max(a, b); }

static inline uint8_t minimumub(uint8_t a, uint8_t b) { return min(a, b); }
static inline int32_t minimumi(int32_t a, int32_t b) { return min(a, b); }
static inline uint32_t minimumu(uint32_t a, uint32_t b) { return min(a, b); }
static inline uint64_t minimumu64(uint64_t a, uint64_t b) { return min(a, b); }
static inline float minimumf(float a, float b) { return min(a, b); }
				
static inline uint8_t maximumub(uint8_t a, uint8_t b) { return max(a, b); }
static inline int32_t maximumi(int32_t a, int32_t b) { return max(a, b); }
static inline uint32_t maximumu(uint32_t a, uint32_t b) { return max(a, b); }
static inline float maximumf(float a, float b) { return max(a, b); }
				
static inline uniform int32_t iabs32(uniform int32_t v) { uniform uint32_t msk = v >> 31; return (v ^ msk) - msk; }
static inline int32_t iabs32(int32_t v) { uint32_t msk = v >> 31; return (v ^ msk) - msk; }

static inline void swapub(uniform uint8_t * uniform a, uniform uint8_t * uniform b) { uniform uint8_t t = *a; *a = *b; *b = t; }
static inline void swapu(uniform uint32_t * uniform a, uniform uint32_t * uniform b) { uniform uint32_t t = *a; *a = *b; *b = t; }
static inline void swapi(uniform int32_t * uniform a, uniform int32_t * uniform b) { uniform int32_t t = *a; *a = *b; *b = t; }
static inline void swapf(uniform float * uniform a, uniform float * uniform b) { uniform float t = *a; *a = *b; *b = t; }

static inline void swapub(varying uint8_t *uniform a, varying uint8_t *uniform b) { uint8_t t = *a; *a = *b; *b = t; }
static inline void swapi(varying int32_t *uniform a, varying int32_t *uniform b) { int32_t t = *a; *a = *b; *b = t; }
static inline void swapu(varying uint32_t *uniform a, varying uint32_t *uniform b) { uint32_t t = *a; *a = *b; *b = t; }
static inline void swapf(varying float *uniform a, varying float *uniform b) { float t = *a; *a = *b; *b = t; }

static inline float square(float s) { return s * s; }
static inline int square(int s) { return s * s; }

struct color_quad_u8
{
	uint8_t m_c[4];
};

struct color_quad_i
{
	int32_t m_c[4];
};

struct color_quad_f
{
	float m_c[4];
};

static inline color_quad_i component_min_rgb(const varying color_quad_i * uniform pA, const varying color_quad_i * uniform pB) 
{ 
	color_quad_i res; 
	res.m_c[0] = minimumi(pA->m_c[0], pB->m_c[0]);
	res.m_c[1] = minimumi(pA->m_c[1], pB->m_c[1]);
	res.m_c[2] = minimumi(pA->m_c[2], pB->m_c[2]);
	res.m_c[3] = 255;
	return res;
}

static inline color_quad_i component_max_rgb(const varying color_quad_i * uniform pA, const varying color_quad_i * uniform pB) 
{ 
	color_quad_i res;
	res.m_c[0] = maximumi(pA->m_c[0], pB->m_c[0]);
	res.m_c[1] = maximumi(pA->m_c[1], pB->m_c[1]);
	res.m_c[2] = maximumi(pA->m_c[2], pB->m_c[2]);
	res.m_c[3] = 255;
	return res;
}

static inline varying color_quad_i *color_quad_i_set_clamped(varying color_quad_i * uniform pRes, varying int32_t r, varying int32_t g, varying int32_t b, varying int32_t a)
{
	pRes->m_c[0] = clampi(r, 0, 255);
	pRes->m_c[1] = clampi(g, 0, 255);
	pRes->m_c[2] = clampi(b, 0, 255);
	pRes->m_c[3] = clampi(a, 0, 255);
	return pRes;
}

static inline varying color_quad_i *color_quad_i_set(varying color_quad_i * uniform pRes, varying int32_t r, varying int32_t g, varying int32_t b, varying int32_t a)
{
	pRes->m_c[0] = r;
	pRes->m_c[1] = g;
	pRes->m_c[2] = b;
	pRes->m_c[3] = a;
	return pRes;
}

static inline varying color_quad_i *color_quad_i_set(varying color_quad_i * uniform pRes, uniform int32_t r, uniform int32_t g, uniform int32_t b, uniform int32_t a)
{
	pRes->m_c[0] = r;
	pRes->m_c[1] = g;
	pRes->m_c[2] = b;
	pRes->m_c[3] = a;
	return pRes;
}

static inline bool color_quad_i_equals(const varying color_quad_i * uniform pLHS, const varying color_quad_i * uniform pRHS)
{
	return (pLHS->m_c[0] == pRHS->m_c[0]) && (pLHS->m_c[1] == pRHS->m_c[1]) && (pLHS->m_c[2] == pRHS->m_c[2]) && (pLHS->m_c[3] == pRHS->m_c[3]);
}

static inline bool color_quad_i_notequals(const varying color_quad_i * uniform pLHS, const varying color_quad_i * uniform pRHS)
{
	return !color_quad_i_equals(pLHS, pRHS);
}

static inline float copysign(float value, float sign)
{
	return floatbits(intbits(value) | (intbits(sign) & 0x80000000));
}

struct vec3F
{
	float m_c[3];
};

struct vec4F
{
	float m_c[4];
};

static inline void vec3F_set_scalar(uniform vec3F * uniform pV, uniform float x)
{
	pV->m_c[0] = x;
	pV->m_c[1] = x;
	pV->m_c[2] = x;
}

static inline varying vec3F * uniform vec3F_set(varying vec3F * uniform pV, float x, float y, float z)
{
	pV->m_c[0] = x;
	pV->m_c[1] = y;
	pV->m_c[2] = z;
	return pV;
}

static inline void vec3F_accum(varying vec3F * uniform pLHS, const varying vec3F * uniform pRHS)
{
	pLHS->m_c[0] += pRHS->m_c[0];
	pLHS->m_c[1] += pRHS->m_c[1];
	pLHS->m_c[2] += pRHS->m_c[2];
}

static inline float vec3F_dot(const uniform vec3F * uniform pLHS, const varying vec3F * uniform pRHS)
{
	return pLHS->m_c[0] * pRHS->m_c[0] + pLHS->m_c[1] * pRHS->m_c[1] + pLHS->m_c[2] * pRHS->m_c[2];
}

static inline float vec3F_dot(const varying vec3F * uniform pLHS, const varying vec3F * uniform pRHS)
{
	return pLHS->m_c[0] * pRHS->m_c[0] + pLHS->m_c[1] * pRHS->m_c[1] + pLHS->m_c[2] * pRHS->m_c[2];
}

static inline vec3F vec3F_mul(const uniform vec3F * uniform pLHS, float s)
{
	vec3F res;
	vec3F_set(&res, pLHS->m_c[0] * s, pLHS->m_c[1] * s, pLHS->m_c[2] * s);
	return res;
}

static inline vec3F vec3F_mul(const varying vec3F * uniform pLHS, float s)
{
	vec3F res;
	vec3F_set(&res, pLHS->m_c[0] * s, pLHS->m_c[1] * s, pLHS->m_c[2] * s);
	return res;
}

static inline vec3F vec3F_mul(const varying vec3F * uniform pLHS, const uniform vec3F * uniform pRHS)
{
	vec3F res;
	vec3F_set(&res, pLHS->m_c[0] * pRHS->m_c[0], pLHS->m_c[1] * pRHS->m_c[1], pLHS->m_c[2] * pRHS->m_c[2]);
	return res;
}

static inline varying vec4F * uniform vec4F_set_scalar(varying vec4F * uniform pV, float x)
{
	pV->m_c[0] = x;
	pV->m_c[1] = x;
	pV->m_c[2] = x;
	pV->m_c[3] = x;
	return pV;
}

static inline varying vec4F * uniform vec4F_set(varying vec4F * uniform pV, float x, float y, float z, float w)
{
	pV->m_c[0] = x;
	pV->m_c[1] = y;
	pV->m_c[2] = z;
	pV->m_c[3] = w;
	return pV;
}

static inline varying vec4F * uniform vec4F_saturate_in_place(varying vec4F * uniform pV)
{
	pV->m_c[0] = saturate(pV->m_c[0]);
	pV->m_c[1] = saturate(pV->m_c[1]);
	pV->m_c[2] = saturate(pV->m_c[2]);
	pV->m_c[3] = saturate(pV->m_c[3]);
	return pV;
}

static inline vec4F vec4F_saturate(const varying vec4F * uniform pV)
{
	vec4F res;
	res.m_c[0] = saturate(pV->m_c[0]);
	res.m_c[1] = saturate(pV->m_c[1]);
	res.m_c[2] = saturate(pV->m_c[2]);
	res.m_c[3] = saturate(pV->m_c[3]);
	return res;
}

static inline vec4F vec4F_from_color(const varying color_quad_i * uniform pC)
{
	vec4F res;
	vec4F_set(&res, pC->m_c[0], pC->m_c[1], pC->m_c[2], pC->m_c[3]);
	return res;
}

static inline vec4F vec4F_add(const varying vec4F * uniform pLHS, const varying vec4F * uniform pRHS)
{
	vec4F res;
	vec4F_set(&res, pLHS->m_c[0] + pRHS->m_c[0], pLHS->m_c[1] + pRHS->m_c[1], pLHS->m_c[2] + pRHS->m_c[2], pLHS->m_c[3] + pRHS->m_c[3]);
	return res;
}

static inline vec4F vec4F_sub(const varying vec4F * uniform pLHS, const varying vec4F *uniform pRHS)
{
	vec4F res;
	vec4F_set(&res, pLHS->m_c[0] - pRHS->m_c[0], pLHS->m_c[1] - pRHS->m_c[1], pLHS->m_c[2] - pRHS->m_c[2], pLHS->m_c[3] - pRHS->m_c[3]);
	return res;
}

static inline float vec4F_dot(const varying vec4F * uniform pLHS, const varying vec4F * uniform pRHS)
{
	return pLHS->m_c[0] * pRHS->m_c[0] + pLHS->m_c[1] * pRHS->m_c[1] + pLHS->m_c[2] * pRHS->m_c[2] + pLHS->m_c[3] * pRHS->m_c[3];
}

static inline vec4F vec4F_mul(const varying vec4F * uniform pLHS, float s)
{
	vec4F res;
	vec4F_set(&res, pLHS->m_c[0] * s, pLHS->m_c[1] * s, pLHS->m_c[2] * s, pLHS->m_c[3] * s);
	return res;
}

static inline varying vec4F *vec4F_normalize_in_place(varying vec4F * uniform pV)
{
	float s = pV->m_c[0] * pV->m_c[0] + pV->m_c[1] * pV->m_c[1] + pV->m_c[2] * pV->m_c[2] + pV->m_c[3] * pV->m_c[3];
				
	if (s != 0.0f)
	{
		s = 1.0f / sqrt(s);

		pV->m_c[0] *= s;
		pV->m_c[1] *= s;
		pV->m_c[2] *= s;
		pV->m_c[3] *= s;
	}

	return pV;
}

// This "pseudo-normalizes" the vector. The results are machine independent and are "near" unit length while being faster than normalizing.
static inline varying vec4F *vec4F_pseudo_normalize_in_place(varying vec4F * uniform pV)
{
	float peak = max(max(max(abs(pV->m_c[0]), abs(pV->m_c[1])), abs(pV->m_c[2])), abs(pV->m_c[3]));
	float s = 1.0f / max(peak, 1.0e-20f);
	pV->m_c[0] *= s;
	pV->m_c[1] *= s;
	pV->m_c[2] *= s;
	pV->m_c[3] *= s;

	return pV;
}

static const uniform uint32_t g_bc7_weights2[4] = { 0, 21, 43, 64 };
static const uniform uint32_t g_bc7_weights3[8] = { 0, 9, 18, 27, 37, 46, 55, 64 };
static const uniform uint32_t g_bc7_weights4[16] = { 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 };

// Precomputed weight constants used during least fit determination. For each entry in g_bc7_weights[]: w * w, (1.0f - w) * w, (1.0f - w) * (1.0f - w), w
// Using a macro to initialize this table is easier to understand and more accurate than using 6-digit floats. 6-digit floats have 2 ULPs error for most
// of these constants, making them use 23 bits as a float, whereas the exact constants need at most 12 bits. The exact constants should always give exact
// results for the determinant calculation in least squares solving, whereas 6-digit floats would have small truncation error. This *could* lead to cases
// where exact weights would have a zero determinant, but 6-digit weights would have an epsilon determinant. I didn't actually see any such cases, but I
// have seen enough rare problems from precision issues that I always like to maximize floating point precision.
#define LSQWTS(n)	n * n / 4096.0f, (64 - n) * n / 4096.0f, (64 - n) * (64 - n) / 4096.0f, n / 64.0f
static const uniform float g_bc7_weights2x[4 * 4] = { LSQWTS(0), LSQWTS(21), LSQWTS(43), LSQWTS(64) };
static const uniform float g_bc7_weights3x[8 * 4] = { LSQWTS(0), LSQWTS(9), LSQWTS(18), LSQWTS(27), LSQWTS(37), LSQWTS(46), LSQWTS(55), LSQWTS(64) };
static const uniform float g_bc7_weights4x[16 * 4] = { LSQWTS(0), LSQWTS(4), LSQWTS(9), LSQWTS(13), LSQWTS(17), LSQWTS(21), LSQWTS(26), LSQWTS(30), LSQWTS(34), LSQWTS(38), LSQWTS(43), LSQWTS(47), LSQWTS(51), LSQWTS(55), LSQWTS(60), LSQWTS(64) };

static const uniform int g_bc7_partition1[16] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };

static const uniform int g_bc7_partition2[64 * 16] =
{
	0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,		0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,		0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,		0,0,0,1,0,0,1,1,0,0,1,1,0,1,1,1,		0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1,		0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1,		0,0,0,1,0,0,1,1,0,1,1,1,1,1,1,1,		0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1,
	0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,		0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,		0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,		0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,		0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,		0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,		0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,		0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,
	0,0,0,0,1,0,0,0,1,1,1,0,1,1,1,1,		0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,		0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,		0,1,1,1,0,0,1,1,0,0,0,1,0,0,0,0,		0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,		0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,0,		0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,		0,1,1,1,0,0,1,1,0,0,1,1,0,0,0,1,
	0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,		0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,		0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,		0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,		0,0,0,1,0,1,1,1,1,1,1,0,1,0,0,0,		0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,		0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,		0,0,1,1,1,0,0,1,1,0,0,1,1,1,0,0,
	0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,		0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,		0,1,0,1,1,0,1,0,0,1,0,1,1,0,1,0,		0,0,1,1,0,0,1,1,1,1,0,0,1,1,0,0,		0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,		0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0,		0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,		0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,1,
	0,1,1,1,0,0,1,1,1,1,0,0,1,1,1,0,		0,0,0,1,0,0,1,1,1,1,0,0,1,0,0,0,		0,0,1,1,0,0,1,0,0,1,0,0,1,1,0,0,		0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0,		0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,		0,0,1,1,1,1,0,0,1,1,0,0,0,0,1,1,		0,1,1,0,0,1,1,0,1,0,0,1,1,0,0,1,		0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,
	0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0,		0,0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,		0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,		0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,0,		0,1,1,0,1,1,0,0,1,0,0,1,0,0,1,1,		0,0,1,1,0,1,1,0,1,1,0,0,1,0,0,1,		0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0,		0,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0,
	0,1,1,0,1,1,0,0,1,1,0,0,1,0,0,1,		0,1,1,0,0,0,1,1,0,0,1,1,1,0,0,1,		0,1,1,1,1,1,1,0,1,0,0,0,0,0,0,1,		0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,		0,0,0,0,1,1,1,1,0,0,1,1,0,0,1,1,		0,0,1,1,0,0,1,1,1,1,1,1,0,0,0,0,		0,0,1,0,0,0,1,0,1,1,1,0,1,1,1,0,		0,1,0,0,0,1,0,0,0,1,1,1,0,1,1,1
};

static const uniform int g_bc7_table_anchor_index_second_subset[64] =
{
	15,15,15,15,15,15,15,15,		15,15,15,15,15,15,15,15,		15, 2, 8, 2, 2, 8, 8,15,		2, 8, 2, 2, 8, 8, 2, 2,		15,15, 6, 8, 2, 8,15,15,		2, 8, 2, 2, 2,15,15, 6,		6, 2, 6, 8,15,15, 2, 2,		15,15,15,15,15, 2, 2,15
};

static const uniform int g_bc7_partition3[64 * 16] =
{
	0,0,1,1,0,0,1,1,0,2,2,1,2,2,2,2,		0,0,0,1,0,0,1,1,2,2,1,1,2,2,2,1,		0,0,0,0,2,0,0,1,2,2,1,1,2,2,1,1,		0,2,2,2,0,0,2,2,0,0,1,1,0,1,1,1,		0,0,0,0,0,0,0,0,1,1,2,2,1,1,2,2,		0,0,1,1,0,0,1,1,0,0,2,2,0,0,2,2,		0,0,2,2,0,0,2,2,1,1,1,1,1,1,1,1,		0,0,1,1,0,0,1,1,2,2,1,1,2,2,1,1,
	0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,		0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,		0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,2,		0,0,1,2,0,0,1,2,0,0,1,2,0,0,1,2,		0,1,1,2,0,1,1,2,0,1,1,2,0,1,1,2,		0,1,2,2,0,1,2,2,0,1,2,2,0,1,2,2,		0,0,1,1,0,1,1,2,1,1,2,2,1,2,2,2,		0,0,1,1,2,0,0,1,2,2,0,0,2,2,2,0,
	0,0,0,1,0,0,1,1,0,1,1,2,1,1,2,2,		0,1,1,1,0,0,1,1,2,0,0,1,2,2,0,0,		0,0,0,0,1,1,2,2,1,1,2,2,1,1,2,2,		0,0,2,2,0,0,2,2,0,0,2,2,1,1,1,1,		0,1,1,1,0,1,1,1,0,2,2,2,0,2,2,2,		0,0,0,1,0,0,0,1,2,2,2,1,2,2,2,1,		0,0,0,0,0,0,1,1,0,1,2,2,0,1,2,2,		0,0,0,0,1,1,0,0,2,2,1,0,2,2,1,0,
	0,1,2,2,0,1,2,2,0,0,1,1,0,0,0,0,		0,0,1,2,0,0,1,2,1,1,2,2,2,2,2,2,		0,1,1,0,1,2,2,1,1,2,2,1,0,1,1,0,		0,0,0,0,0,1,1,0,1,2,2,1,1,2,2,1,		0,0,2,2,1,1,0,2,1,1,0,2,0,0,2,2,		0,1,1,0,0,1,1,0,2,0,0,2,2,2,2,2,		0,0,1,1,0,1,2,2,0,1,2,2,0,0,1,1,		0,0,0,0,2,0,0,0,2,2,1,1,2,2,2,1,
	0,0,0,0,0,0,0,2,1,1,2,2,1,2,2,2,		0,2,2,2,0,0,2,2,0,0,1,2,0,0,1,1,		0,0,1,1,0,0,1,2,0,0,2,2,0,2,2,2,		0,1,2,0,0,1,2,0,0,1,2,0,0,1,2,0,		0,0,0,0,1,1,1,1,2,2,2,2,0,0,0,0,		0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,		0,1,2,0,2,0,1,2,1,2,0,1,0,1,2,0,		0,0,1,1,2,2,0,0,1,1,2,2,0,0,1,1,
	0,0,1,1,1,1,2,2,2,2,0,0,0,0,1,1,		0,1,0,1,0,1,0,1,2,2,2,2,2,2,2,2,		0,0,0,0,0,0,0,0,2,1,2,1,2,1,2,1,		0,0,2,2,1,1,2,2,0,0,2,2,1,1,2,2,		0,0,2,2,0,0,1,1,0,0,2,2,0,0,1,1,		0,2,2,0,1,2,2,1,0,2,2,0,1,2,2,1,		0,1,0,1,2,2,2,2,2,2,2,2,0,1,0,1,		0,0,0,0,2,1,2,1,2,1,2,1,2,1,2,1,
	0,1,0,1,0,1,0,1,0,1,0,1,2,2,2,2,		0,2,2,2,0,1,1,1,0,2,2,2,0,1,1,1,		0,0,0,2,1,1,1,2,0,0,0,2,1,1,1,2,		0,0,0,0,2,1,1,2,2,1,1,2,2,1,1,2,		0,2,2,2,0,1,1,1,0,1,1,1,0,2,2,2,		0,0,0,2,1,1,1,2,1,1,1,2,0,0,0,2,		0,1,1,0,0,1,1,0,0,1,1,0,2,2,2,2,		0,0,0,0,0,0,0,0,2,1,1,2,2,1,1,2,
	0,1,1,0,0,1,1,0,2,2,2,2,2,2,2,2,		0,0,2,2,0,0,1,1,0,0,1,1,0,0,2,2,		0,0,2,2,1,1,2,2,1,1,2,2,0,0,2,2,		0,0,0,0,0,0,0,0,0,0,0,0,2,1,1,2,		0,0,0,2,0,0,0,1,0,0,0,2,0,0,0,1,		0,2,2,2,1,2,2,2,0,2,2,2,1,2,2,2,		0,1,0,1,2,2,2,2,2,2,2,2,2,2,2,2,		0,1,1,1,2,0,1,1,2,2,0,1,2,2,2,0,
};

static const uniform int g_bc7_table_anchor_index_third_subset_1[64] =
{
	3, 3,15,15, 8, 3,15,15,		8, 8, 6, 6, 6, 5, 3, 3,		3, 3, 8,15, 3, 3, 6,10,		5, 8, 8, 6, 8, 5,15,15,		8,15, 3, 5, 6,10, 8,15,		15, 3,15, 5,15,15,15,15,		3,15, 5, 5, 5, 8, 5,10,		5,10, 8,13,15,12, 3, 3
};

static const uniform int g_bc7_table_anchor_index_third_subset_2[64] =
{
	15, 8, 8, 3,15,15, 3, 8,		15,15,15,15,15,15,15, 8,		15, 8,15, 3,15, 8,15, 8,		3,15, 6,10,15,15,10, 8,		15, 3,15,10,10, 8, 9,10,		6,15, 8,15, 3, 6, 6, 8,		15, 3,15,15,15,15,15,15,		15,15,15,15, 3,15,15, 8
};

static const uniform int g_bc7_num_subsets[8] = { 3, 2, 3, 2, 1, 1, 1, 2 };
static const uniform int g_bc7_partition_bits[8] = { 4, 6, 6, 6, 0, 0, 0, 6 };
static const uniform int g_bc7_rotation_bits[8] = { 0, 0, 0, 0, 2, 2, 0, 0 };
static const uniform int g_bc7_color_index_bitcount[8] = { 3, 3, 2, 2, 2, 2, 4, 2 };
static int uniform get_bc7_color_index_size(uniform int mode, uniform int index_selection_bit) { return g_bc7_color_index_bitcount[mode] + index_selection_bit; }
#pragma ignore warning(perf)
static int varying get_bc7_color_index_size(varying int mode, varying int index_selection_bit) { return g_bc7_color_index_bitcount[mode] + index_selection_bit; }
static uniform int g_bc7_alpha_index_bitcount[8] = { 0, 0, 0, 0, 3, 2, 4, 2 };
static uniform int get_bc7_alpha_index_size(uniform int mode, uniform int index_selection_bit) { return g_bc7_alpha_index_bitcount[mode] - index_selection_bit; }
#pragma ignore warning(perf)
static varying int get_bc7_alpha_index_size(varying int mode, varying int index_selection_bit) { return g_bc7_alpha_index_bitcount[mode] - index_selection_bit; }
static const uniform int g_bc7_mode_has_p_bits[8] = { 1, 1, 0, 1, 0, 0, 1, 1 };
static const uniform int g_bc7_mode_has_shared_p_bits[8] = { 0, 1, 0, 0, 0, 0, 0, 0 };
static const uniform int g_bc7_color_precision_table[8] = { 4, 6, 5, 7, 5, 7, 7, 5 };
static const uniform int g_bc7_color_precision_plus_pbit_table[8] = { 5, 7, 5, 8, 5, 7, 8, 6 };
static const uniform int g_bc7_alpha_precision_table[8] = { 0, 0, 0, 0, 6, 8, 7, 5 };
static const uniform int g_bc7_alpha_precision_plus_pbit_table[8] = { 0, 0, 0, 0, 6, 8, 8, 6 };
static uniform bool get_bc7_mode_has_seperate_alpha_selectors(uniform int mode) { return (mode == 4) || (mode == 5); }
static varying bool get_bc7_mode_has_seperate_alpha_selectors(varying int mode) { return (mode == 4) || (mode == 5); }

struct endpoint_err
{
	uint16_t m_error;
	uint8_t m_lo;
	uint8_t m_hi;
};

static uniform endpoint_err g_bc7_mode_1_optimal_endpoints[256][2]; // [c][pbit]
const uniform uint32_t BC7E_MODE_1_OPTIMAL_INDEX = 2;

static uniform endpoint_err g_bc7_mode_7_optimal_endpoints[256][2][2]; // [c][pbit][hp][lp]
const uniform uint32_t BC7E_MODE_7_OPTIMAL_INDEX = 1;

static uniform endpoint_err g_bc7_mode_6_optimal_endpoints[256][2][2]; // [c][hp][lp]
const uniform uint32_t BC7E_MODE_6_OPTIMAL_INDEX = 5;

static uniform uint32_t g_bc7_mode_4_optimal_endpoints3[256]; // [c]
static uniform uint32_t g_bc7_mode_4_optimal_endpoints2[256]; // [c]
const uniform uint32_t BC7E_MODE_4_OPTIMAL_INDEX3 = 2;
const uniform uint32_t BC7E_MODE_4_OPTIMAL_INDEX2 = 1;

static uniform uint32_t g_bc7_mode_5_optimal_endpoints[256]; // [c]
const uniform uint32_t BC7E_MODE_5_OPTIMAL_INDEX = 1;

static uniform endpoint_err g_bc7_mode_0_optimal_endpoints[256][2][2]; // [c][hp][lp]
const uniform uint32_t BC7E_MODE_0_OPTIMAL_INDEX = 2;

static uniform bool g_codec_initialized;

export void bc7e_compress_block_init()
{
	if (g_codec_initialized)
		return;

	// Mode 0: 444.1
	for (uniform int c = 0; c < 256; c++)
	{
		for (uniform uint32_t hp = 0; hp < 2; hp++)
		{
			for (uniform uint32_t lp = 0; lp < 2; lp++)
			{
				uniform endpoint_err best;
				best.m_error = (uint16_t)UINT16_MAX;

				for (uniform uint32_t l = 0; l < 16; l++)
				{
					uniform uint32_t low = ((l << 1) | lp) << 3;
					low |= (low >> 5);

					for (uniform uint32_t h = 0; h < 16; h++)
					{
						uniform uint32_t high = ((h << 1) | hp) << 3;
						high |= (high >> 5);

						const uniform int k = (low * (64 - g_bc7_weights3[BC7E_MODE_0_OPTIMAL_INDEX]) + high * g_bc7_weights3[BC7E_MODE_0_OPTIMAL_INDEX] + 32) >> 6;

						const uniform int err = (k - c) * (k - c);
						if (err < best.m_error)
						{
							best.m_error = (uint16_t)err;
							best.m_lo = (uint8_t)l;
							best.m_hi = (uint8_t)h;
						}
					} // h
				} // l

				g_bc7_mode_0_optimal_endpoints[c][hp][lp] = best;
			} // lp
		} // hp
	} // c

	// Mode 1: 666.1
	for (uniform int c = 0; c < 256; c++)
	{
		for (uniform uint32_t lp = 0; lp < 2; lp++)
		{
			uniform endpoint_err best;
			best.m_error = (uint16_t)UINT16_MAX;

			for (uniform uint32_t l = 0; l < 64; l++)
			{
				uniform uint32_t low = ((l << 1) | lp) << 1;
				low |= (low >> 7);

				for (uniform uint32_t h = 0; h < 64; h++)
				{
					uniform uint32_t high = ((h << 1) | lp) << 1;
					high |= (high >> 7);

					const uniform int k = (low * (64 - g_bc7_weights3[BC7E_MODE_1_OPTIMAL_INDEX]) + high * g_bc7_weights3[BC7E_MODE_1_OPTIMAL_INDEX] + 32) >> 6;

					const uniform int err = (k - c) * (k - c);
					if (err < best.m_error)
					{
						best.m_error = (uint16_t)err;
						best.m_lo = (uint8_t)l;
						best.m_hi = (uint8_t)h;
					}
				} // h
			} // l

			g_bc7_mode_1_optimal_endpoints[c][lp] = best;
		} // lp
	} // c

	// Mode 6: 777.1 4-bit indices
	for (uniform int c = 0; c < 256; c++)
	{
		for (uniform uint32_t hp = 0; hp < 2; hp++)
		{
			for (uniform uint32_t lp = 0; lp < 2; lp++)
			{
				uniform endpoint_err best;
				best.m_error = (uint16_t)UINT16_MAX;

				for (uniform uint32_t l = 0; l < 128; l++)
				{
					uniform uint32_t low = (l << 1) | lp;
				
					for (uniform uint32_t h = 0; h < 128; h++)
					{
						uniform uint32_t high = (h << 1) | hp;
					
						const uniform int k = (low * (64 - g_bc7_weights4[BC7E_MODE_6_OPTIMAL_INDEX]) + high * g_bc7_weights4[BC7E_MODE_6_OPTIMAL_INDEX] + 32) >> 6;

						const uniform int err = (k - c) * (k - c);
						if (err < best.m_error)
						{
							best.m_error = (uint16_t)err;
							best.m_lo = (uint8_t)l;
							best.m_hi = (uint8_t)h;
						}
					} // h
				} // l

				g_bc7_mode_6_optimal_endpoints[c][hp][lp] = best;
			} // lp
		} // hp
	} // c

	// Mode 5: 777 2-bit indices
	for (uniform int c = 0; c < 256; c++)
	{
		uniform endpoint_err best;
		best.m_error = (uint16_t)UINT16_MAX;
		best.m_lo = 0;
		best.m_hi = 0;

		for (uniform uint32_t l = 0; l < 128; l++)
		{
			uniform uint32_t low = l << 1;
			low |= (low >> 7);

			for (uniform uint32_t h = 0; h < 128; h++)
			{
				uniform uint32_t high = h << 1;
				high |= (high >> 7);

				const uniform int k = (low * (64 - g_bc7_weights2[BC7E_MODE_5_OPTIMAL_INDEX]) + high * g_bc7_weights2[BC7E_MODE_5_OPTIMAL_INDEX] + 32) >> 6;

				const uniform int err = (k - c) * (k - c);
				if (err < best.m_error)
				{
					best.m_error = (uint16_t)err;
					best.m_lo = (uint8_t)l;
					best.m_hi = (uint8_t)h;
				}
			} // h
		} // l

		g_bc7_mode_5_optimal_endpoints[c] = (uint32_t)best.m_lo | (((uint32_t)best.m_hi) << 8);

	} // c


	//Mode 4: 555 3-bit indices
	for (uniform int c = 0; c < 256; c++)
	{
		uniform endpoint_err best;
		best.m_error = (uint16_t)UINT16_MAX;
		best.m_lo = 0;
		best.m_hi = 0;

		for (uniform uint32_t l = 0; l < 32; l++)
		{
			uniform uint32_t low = l << 3;
			low |= (low >> 5);

			for (uniform uint32_t h = 0; h < 32; h++)
			{
				uniform uint32_t high = h << 3;
				high |= (high >> 5);

				const uniform int k = (low * (64 - g_bc7_weights3[BC7E_MODE_4_OPTIMAL_INDEX3]) + high * g_bc7_weights3[BC7E_MODE_4_OPTIMAL_INDEX3] + 32) >> 6;

				const uniform int err = (k - c) * (k - c);
				if (err < best.m_error)
				{
					best.m_error = (uint16_t)err;
					best.m_lo = (uint8_t)l;
					best.m_hi = (uint8_t)h;
				}
			} // h
		} // l

		g_bc7_mode_4_optimal_endpoints3[c] = (uint32_t)best.m_lo | (((uint32_t)best.m_hi) << 8);

	} // c
	
	// Mode 4: 555 2-bit indices
	for (uniform int c = 0; c < 256; c++)
	{
		uniform endpoint_err best;
		best.m_error = (uint16_t)UINT16_MAX;
		best.m_lo = 0;
		best.m_hi = 0;

		for (uniform uint32_t l = 0; l < 32; l++)
		{
			uniform uint32_t low = l << 3;
			low |= (low >> 5);

			for (uniform uint32_t h = 0; h < 32; h++)
			{
				uniform uint32_t high = h << 3;
				high |= (high >> 5);

				const uniform int k = (low * (64 - g_bc7_weights2[BC7E_MODE_4_OPTIMAL_INDEX2]) + high * g_bc7_weights2[BC7E_MODE_4_OPTIMAL_INDEX2] + 32) >> 6;

				const uniform int err = (k - c) * (k - c);
				if (err < best.m_error)
				{
					best.m_error = (uint16_t)err;
					best.m_lo = (uint8_t)l;
					best.m_hi = (uint8_t)h;
				}
			} // h
		} // l

		g_bc7_mode_4_optimal_endpoints2[c] = (uint32_t)best.m_lo | (((uint32_t)best.m_hi) << 8);

	} // c

	// Mode 7: 555.1 2-bit indices 
	for (uniform int c = 0; c < 256; c++)
	{
		uniform endpoint_err best;
		best.m_error = (uint16_t)UINT16_MAX;
		best.m_lo = 0;
		best.m_hi = 0;

		for (uniform uint32_t hp = 0; hp < 2; hp++)
		{
			for (uniform uint32_t lp = 0; lp < 2; lp++)	
			{
				for (uniform uint32_t l = 0; l < 32; l++)
				{
					uniform uint32_t low = ((l << 1) | lp) << 2;
					low |= (low >> 6);

					for (uniform uint32_t h = 0; h < 32; h++)
					{
						uniform uint32_t high = ((h << 1) | hp) << 2;
						high |= (high >> 6);

						const uniform int k = (low * (64 - g_bc7_weights2[BC7E_MODE_7_OPTIMAL_INDEX]) + high * g_bc7_weights2[BC7E_MODE_7_OPTIMAL_INDEX] + 32) >> 6;

						const uniform int err = (k - c) * (k - c);
						if (err < best.m_error)
						{
							best.m_error = (uint16_t)err;
							best.m_lo = (uint8_t)l;
							best.m_hi = (uint8_t)h;
						}
					} // h
				} // l

				g_bc7_mode_7_optimal_endpoints[c][hp][lp] = best;
			
			} // hp

		} // lp

	} // c

	g_codec_initialized = true;
}

static void compute_least_squares_endpoints_rgba(uint32_t N, const varying int *uniform pSelectors, const uniform vec4F *uniform pSelector_weights, varying vec4F *uniform pXl, varying vec4F *uniform pXh, const varying color_quad_i * uniform pColors)
{
	// Least squares using normal equations: http://www.cs.cornell.edu/~bindel/class/cs3220-s12/notes/lec10.pdf 
	// I did this in matrix form first, expanded out all the ops, then optimized it a bit.
	float z00 = 0.0f, z01 = 0.0f, z10 = 0.0f, z11 = 0.0f;
	float q00_r = 0.0f, q10_r = 0.0f, t_r = 0.0f;
	float q00_g = 0.0f, q10_g = 0.0f, t_g = 0.0f;
	float q00_b = 0.0f, q10_b = 0.0f, t_b = 0.0f;
	float q00_a = 0.0f, q10_a = 0.0f, t_a = 0.0f;
	for (uniform uint32_t i = 0; i < N; i++)
	{
		const uint32_t sel = pSelectors[i];

#pragma ignore warning(perf)
		z00 += pSelector_weights[sel].m_c[0];
#pragma ignore warning(perf)
		z10 += pSelector_weights[sel].m_c[1];
#pragma ignore warning(perf)
		z11 += pSelector_weights[sel].m_c[2];

#pragma ignore warning(perf)
		float w = pSelector_weights[sel].m_c[3];

		q00_r += w * (int)pColors[i].m_c[0]; t_r += (int)pColors[i].m_c[0];
		q00_g += w * (int)pColors[i].m_c[1]; t_g += (int)pColors[i].m_c[1];
		q00_b += w * (int)pColors[i].m_c[2]; t_b += (int)pColors[i].m_c[2];
		q00_a += w * (int)pColors[i].m_c[3]; t_a += (int)pColors[i].m_c[3];
	}

	q10_r = t_r - q00_r;
	q10_g = t_g - q00_g;
	q10_b = t_b - q00_b;
	q10_a = t_a - q00_a;

	z01 = z10;

	float det = z00 * z11 - z01 * z10;
	if (det != 0.0f)
		det = 1.0f / det;

	float iz00, iz01, iz10, iz11;
	iz00 = z11 * det;
	iz01 = -z01 * det;
	iz10 = -z10 * det;
	iz11 = z00 * det;

	pXl->m_c[0] = (float)(iz00 * q00_r + iz01 * q10_r); pXh->m_c[0] = (float)(iz10 * q00_r + iz11 * q10_r);
	pXl->m_c[1] = (float)(iz00 * q00_g + iz01 * q10_g); pXh->m_c[1] = (float)(iz10 * q00_g + iz11 * q10_g);
	pXl->m_c[2] = (float)(iz00 * q00_b + iz01 * q10_b); pXh->m_c[2] = (float)(iz10 * q00_b + iz11 * q10_b);
	pXl->m_c[3] = (float)(iz00 * q00_a + iz01 * q10_a); pXh->m_c[3] = (float)(iz10 * q00_a + iz11 * q10_a);
}

static void compute_least_squares_endpoints_rgb(uint32_t N, const varying int *uniform pSelectors, const uniform vec4F *uniform pSelector_weights, varying vec4F *uniform pXl, varying vec4F *uniform pXh, const varying color_quad_i *uniform pColors)
{
	// Least squares using normal equations: http://www.cs.cornell.edu/~bindel/class/cs3220-s12/notes/lec10.pdf 
	// I did this in matrix form first, expanded out all the ops, then optimized it a bit.
	float z00 = 0.0f, z01 = 0.0f, z10 = 0.0f, z11 = 0.0f;
	float q00_r = 0.0f, q10_r = 0.0f, t_r = 0.0f;
	float q00_g = 0.0f, q10_g = 0.0f, t_g = 0.0f;
	float q00_b = 0.0f, q10_b = 0.0f, t_b = 0.0f;
	for (uniform uint32_t i = 0; i < N; i++)
	{
		const uint32_t sel = pSelectors[i];

#pragma ignore warning(perf)
		z00 += pSelector_weights[sel].m_c[0];
#pragma ignore warning(perf)
		z10 += pSelector_weights[sel].m_c[1];
#pragma ignore warning(perf)
		z11 += pSelector_weights[sel].m_c[2];
#pragma ignore warning(perf)
		float w = pSelector_weights[sel].m_c[3];

		q00_r += w * (int)pColors[i].m_c[0]; t_r += (int)pColors[i].m_c[0];
		q00_g += w * (int)pColors[i].m_c[1]; t_g += (int)pColors[i].m_c[1];
		q00_b += w * (int)pColors[i].m_c[2]; t_b += (int)pColors[i].m_c[2];
	}

	q10_r = t_r - q00_r;
	q10_g = t_g - q00_g;
	q10_b = t_b - q00_b;

	z01 = z10;

	float det = z00 * z11 - z01 * z10;
	if (det != 0.0f)
		det = 1.0f / det;

	float iz00, iz01, iz10, iz11;
	iz00 = z11 * det;
	iz01 = -z01 * det;
	iz10 = -z10 * det;
	iz11 = z00 * det;

	pXl->m_c[0] = (float)(iz00 * q00_r + iz01 * q10_r); pXh->m_c[0] = (float)(iz10 * q00_r + iz11 * q10_r);
	pXl->m_c[1] = (float)(iz00 * q00_g + iz01 * q10_g); pXh->m_c[1] = (float)(iz10 * q00_g + iz11 * q10_g);
	pXl->m_c[2] = (float)(iz00 * q00_b + iz01 * q10_b); pXh->m_c[2] = (float)(iz10 * q00_b + iz11 * q10_b);
}

static void compute_least_squares_endpoints_a(uint32_t N, const varying int *uniform pSelectors, const vec4F *uniform pSelector_weights, varying float *uniform pXl, varying float *uniform pXh, const varying color_quad_i *uniform pColors)
{
	// Least squares using normal equations: http://www.cs.cornell.edu/~bindel/class/cs3220-s12/notes/lec10.pdf 
	// I did this in matrix form first, expanded out all the ops, then optimized it a bit.
	float z00 = 0.0f, z01 = 0.0f, z10 = 0.0f, z11 = 0.0f;
	float q00_a = 0.0f, q10_a = 0.0f, t_a = 0.0f;
	for (uniform uint32_t i = 0; i < N; i++)
	{
		const uint32_t sel = pSelectors[i];

#pragma ignore warning(perf)
		z00 += pSelector_weights[sel].m_c[0];
#pragma ignore warning(perf)
		z10 += pSelector_weights[sel].m_c[1];
#pragma ignore warning(perf)
		z11 += pSelector_weights[sel].m_c[2];
#pragma ignore warning(perf)
		float w = pSelector_weights[sel].m_c[3];

		q00_a += w * (int)pColors[i].m_c[3]; t_a += (int)pColors[i].m_c[3];
	}

	q10_a = t_a - q00_a;

	z01 = z10;

	float det = z00 * z11 - z01 * z10;
	if (det != 0.0f)
		det = 1.0f / det;

	float iz00, iz01, iz10, iz11;
	iz00 = z11 * det;
	iz01 = -z01 * det;
	iz10 = -z10 * det;
	iz11 = z00 * det;

	*pXl = (float)(iz00 * q00_a + iz01 * q10_a); *pXh = (float)(iz10 * q00_a + iz11 * q10_a);
}

struct color_cell_compressor_params
{
	uniform uint32_t m_num_selector_weights;
	const uint32_t *uniform m_pSelector_weights;
	const vec4F *uniform m_pSelector_weightsx;
	uniform uint32_t m_comp_bits;
	uniform float m_weights[4];
	uniform float m_frame_buffer_other_value;	// If "m_optimize_for_color_times_alpha", this is 0 to optimize "C * A" (simple multiply) and 255 to optimize "C * A + F * (1 - A)" (alpha blend).
	uniform uint32_t m_rotation;				// If not zero, alpha was swapped with channel "m_rotation - 1" (weights are also swapped).
	uniform bool m_has_alpha;					// Only true for mode 6 or 7 in blocks with alpha. Opaque blocks set this to false for mode 6 and never use mode 7.
	uniform bool m_has_pbits;
	uniform bool m_endpoints_share_pbit;
	uniform bool m_perceptual;
	uniform bool m_optimize_for_color_times_alpha;	// Set for BC7E_OPTIMIZE_FOR_COLOR_TIMES_ALPHA, BC7E_OPTIMIZE_FOR_ALPHA_BLENDING, and for BC7E_OPTIMIZE_FOR_ALPHA_TEST.

	uniform bool m_debug_spam;

	// This is for "m_perceptual && m_optimize_for_color_times_alpha". In that case we want the worst error between two colors in YCbCr
	// *after* alpha blending with the frame buffer. We'll find this by first constructing a parallelepiped of all possible errors, and
	// choosing the furthest from the origin. One corner of the error parallelepiped is the difference of the alpha-weighted colors in
	// YCbCr. From there, we add the parallelepiped for all RGB colors, scaled by the delta in "1 - alpha". In other words, we subtract
	// the RGB parallelepiped scaled by alpha. Finally we scale each YCbCr axis by the square root of its weight, and the worst error is
	// the extremal point on the resulting shape.
	//
	// From this construction, we can see that the closest point can be a corner, edge, or face, but the furthest point is always at a
	// corner. If you are at an edge or a face, there is always at least one direction you can move that will increase distance. We want
	// the furthest point, so we only care about corners.
	//
	// Since it's a parallelepiped, we can describe it as the sum of a set of three basis vectors. In fact, we can trivially get those
	// vectors from the transform to YCbCr space and the YCbCr weights. The transform to YCbCr is:
	//    Y  = k_Y_R  * R + k_Y_G  * G + k_Y_B  * B
	//    Cb = k_Cb_R * R + k_Cr_G * G + k_Cr_B * B = (B - Y) * k_norm_Cb
	//    Cr = k_Cr_R * R + k_Cb_G * G + k_Cb_B * B = (R - Y) * k_norm_Cr
	// The parallelepiped vectors are thus:
	//    Vr = {k_Y_R * sqrt(wtY), k_Cb_R * sqrt(wtCb), k_Cr_R * sqrt(wtCr)} * 255
	//    Vg = {k_Y_G * sqrt(wtY), k_Cb_G * sqrt(wtCb), k_Cr_G * sqrt(wtCr)} * 255
	//    Vb = {k_Y_B * sqrt(wtY), k_Cb_B * sqrt(wtCb), k_Cr_B * sqrt(wtCr)} * 255
	// Of course, these vectors can be precalculated. If the base delta is D, then the eight possible corners C are:
	//    C = D + da (r Vr + g Vg + b Vb)
	// Where r, g, and b can independently be either 0 or 1. The question is, what is the maximum value of |C|^2?
	//
	// To find the furthest corner, we will move to the middle of the parallelepiped using:
	//    M = D + da * ((Vr + Vg + Vb) * 0.5)
	// "Vr + Vg + Vb" can be precalculated, and lots of terms cancel:
	//    (Vr + Vg + Vb).Y = (k_Y_R + k_Y_G + k_Y_B) * sqrt(wtY) * 255 = sqrt(wtY) * 255
	//    (Vr + Vg + Vb).Cb = (k_Cb_R + k_Cb_G + k_Cb_B) * sqrt(wtCb) * 255 = 0
	//    (Vr + Vg + Vb).Cr = (k_Cr_R + k_Cr_G + k_Cr_B) * sqrt(wtCb) * 255 = 0
	// The Cb and Cr terms must be 0 because they are scaled versions of (B - Y) and (R - Y), respectively. The Y weights sum to 1, and
	// the weight on B/R is 1, and 1 - 1 = 0. So we can instead do:
	//    M = D
	//    M.Y += da * sqrt(wtY) * 255 * 0.5
	//
	// From there, the furthest corner can be found one parallelepiped axis at a time. If V_ . M > 0, we add V_, otherwise we subtract
	// it. We can do this by copying the sign of "V_ . M" to "da" before scaling "V_" and accumulating. This takes us to the furthest
	// corner, so we just dot that vector with itself to get the maximum distance. In math:
	//    M += V_ * da * (V_ . M > 0 ? +1 : -1)
	//
	// NOTE: The code actually uses "Cr = R - Y" and "Cb = B - Y" for efficiency, and treats "k_norm_Cr" and "k_norm_Cb" as part of the
	// weights for Cr and Cb. So, m_sqrt_weights for Cr and Cb must include these norms.
	//
	// NOTE: This derivation is for minimizing error when alpha blending (BC7E_OPTIMIZE_FOR_ALPHA_BLENDING). If we have alpha times color
	// that is added to the frame buffer or replaces the frame buffer (BC7E_OPTIMIZE_FOR_COLOR_TIMES_ALPHA), then we just need to set Vr,
	// Vg, and Vb to 0, and we can use the same code.

	uniform vec3F m_sqrt_weights;
	uniform vec3F m_dycrcb_r;		// This is "Vr" in the derivation above
	uniform vec3F m_dycrcb_g;
	uniform vec3F m_dycrcb_b;
	uniform float m_dycrcb_mid;		// This is "(Vr + Vg + Vb) * 0.5" in the derivation above, which is 0 or sqrt(wtY) * 255 * 0.5

	// For simplicity, BC7E_OPTIMIZE_FOR_ALPHA_TEST uses the code path for BC7E_OPTIMIZE_FOR_ALPHA_BLENDING, except that alpha has a ramp
	// applied. Let the low threshold be L and the high threshold be H; for normal alpha teset, L == H. If the alpha is A, we can do:
	//   H' = ceil(H)
	//   L' = min(floor(L), H - 1)
	//   A' = clamp((A - L') / (H' - L') * 255, 0, 255)
	// The floor/ceil rounds the thresholds to an 8-bit representable value. The way we calculate L' guarantees the width is at least 1,
	// and that L == H is interpreted as "A >= H".
	//
	// For example, if L == H == 127.5, then:
	//   H' = 128
	//   L' = 127
	//   A' = clamp((A - 127) * 255, 0, 255)
	// This implies that A' = 0 if A <= 127, and A' = 1 if A >= 128. The calculation for L' and H' guarantees that any value in (127,128]
	// gives the same result.
	//
	// This can of course be precalculated:
	//   m_alpha_scale = (alpha testing) ? 255 / (H' - L') : 1
	//   m_alpha_bias = (alpha testing) ? L' : 0
	//   A' = clamp((A - m_alpha_bias) * m_alpha_scale, 0, 255)
	uniform float m_alpha_scale;
	uniform float m_alpha_bias;
};

struct partition_estimate_params
{
	uniform float m_max_selector;
	uniform float m_rcp_max_selector;
	uniform float m_weights[4];

	uniform bool m_debug_spam;
};

static inline void color_cell_compressor_params_clear(uniform color_cell_compressor_params *uniform p)
{
	p->m_num_selector_weights = 0;
	p->m_pSelector_weights = NULL;
	p->m_pSelector_weightsx = NULL;
	p->m_comp_bits = 0;
	p->m_perceptual = false;
	p->m_optimize_for_color_times_alpha = false;
	p->m_debug_spam = false;
	p->m_weights[0] = 1.0f;
	p->m_weights[1] = 1.0f;
	p->m_weights[2] = 1.0f;
	p->m_weights[3] = 1.0f;
	p->m_frame_buffer_other_value = 0.0f;
	p->m_rotation = 0;
	p->m_has_alpha = false;
	p->m_has_pbits = false;
	p->m_endpoints_share_pbit = false;
	vec3F_set_scalar(&p->m_sqrt_weights, 1);
	vec3F_set_scalar(&p->m_dycrcb_r, 0);
	vec3F_set_scalar(&p->m_dycrcb_g, 0);
	vec3F_set_scalar(&p->m_dycrcb_b, 0);
	p->m_dycrcb_mid = 0;
}

struct color_cell_compressor_results
{
	float m_best_overall_err;
	color_quad_i m_low_endpoint;
	color_quad_i m_high_endpoint;
	uint32_t m_pbits[2];
	varying int *uniform m_pSelectors;
	varying int *uniform m_pSelectors_temp;
};

static inline color_quad_i scale_color(const varying color_quad_i *uniform pC, const uniform color_cell_compressor_params *uniform pParams, uniform uint32_t num_channels)
{
	color_quad_i results;

	const uint32_t n = pParams->m_comp_bits + (pParams->m_has_pbits ? 1 : 0);
	assert((n >= 4) && (n <= 8));

	for (uniform uint32_t i = 0; i < num_channels; i++)
	{
		uint32_t v = pC->m_c[i] << (8 - n);
#pragma ignore warning(perf)
		v |= (v >> n);
		assert(v <= 255);
		results.m_c[i] = v;
	}

	return results;
}

#if 1
// These are the standard weights to turn nonlinear sRGB to nonlinear sYCC.
static const uniform float k_Y_R = 0.2990f;
static const uniform float k_Y_G = 0.5870f;
static const uniform float k_Y_B = 0.1140f;
static const uniform int k_Y_R_int = 29;	// 29/97 = 0.2990
static const uniform int k_Y_G_int = 57;	// 57/97 = 0.5876
static const uniform int k_Y_B_int = 11;	// 11/97 = 0.1134
#else
// These are the standard weights to turn linear RGB to linear YCbCr.
// The "best" luminance results would be to turn sRGB to linear RGB to get linear Y, and then convert that back to gamma space.
// A good and much faster approximation of this is to square RGB, convert to Y with these weights, then take the square root of that.
// This better accounts for the non-linearity of brightness perception, but it makes the compression math non-linear and slower.
static const uniform float k_Y_R = 0.2126f;
static const uniform float k_Y_G = 0.7152f;
static const uniform float k_Y_B = 0.0722f;
static const uniform int k_Y_R_int = 29;	// 29/137 = 0.2117
static const uniform int k_Y_G_int = 98;	// 98/137 = 0.7153
static const uniform int k_Y_B_int = 10;	// 10/137 = 0.0730
static const uniform int k_Y_R_int = 44;	//  44/207 = 0.2126
static const uniform int k_Y_G_int = 148;	// 148/207 = 0.7150
static const uniform int k_Y_B_int = 15;	//  15/207 = 0.0725
#endif

// Conveniently, for both these transforms, we have:
//   Cb = (B - Y) * 0.5 / (1 - k_Y_B)
//   Cr = (R - Y) * 0.5 / (1 - k_Y_R)

static const uniform float k_norm_Cb = 0.5f / (1.0f - k_Y_B);	//  0.5643  or   0.5389
static const uniform float k_norm_Cr = 0.5f / (1.0f - k_Y_R);	//  0.7133  or   0.6350

static const uniform float k_Cb_R = -k_Y_R * k_norm_Cb;			// -0.1687  or  -0.1146 
static const uniform float k_Cb_G = -k_Y_G * k_norm_Cb;			// -0.3313  or  -0.3854
static const uniform float k_Cb_B = 0.5f;						//  0.5 by design

static const uniform float k_Cr_R = 0.5f;						//  0.5 by design
static const uniform float k_Cr_G = -k_Y_G * k_norm_Cr;			// -0.4187  or  -0.4542
static const uniform float k_Cr_B = -k_Y_B * k_norm_Cr;			// -0.0813  or  -0.0458

static const uniform float pr_weight = k_norm_Cr * k_norm_Cr;	//  0.5087  or   0.4032
static const uniform float pb_weight = k_norm_Cb * k_norm_Cb;	//  0.3185  or   0.2904

// We choose the partition by estimating error using a super cheap color line fit to the colors in the different subsets. When using
// perceptual error, we need new weights (using the perceptual weights as if they were RGB weights will obviously give bad results).
// We could use the RGB weights for Y in YCbCr, but that ignores chroma. By blending in uniform weights, we give some small weight to
// chroma, but heavily prefer luminance. Experimentally, I found the sweet spot was to blend the Y channel's RGB weights and uniform
// weights in a 10:1 ratio. This worked best to minimize total squared error across a variety of settings for YCbCr channel error
// weights and across both definitions of the YCbCr transform. The difference in total squared error was 0.03% or less for various
// ratios, so the exact ratio doesn't matter much. The difference from uniform weights to YCbCr weights was about 2%, so using better
// weights does matter a lot.
//
// Incidentally, I tried pre-transforming the block to YCbCr space once and then doing the color estimation for every partition in
// that space. The color partition error estimate code didn't change, but now it worked directly in YCbCr space, so it minimized
// YCbCr error, and could use the given YCbCr weights. This had really bad results; total YCbCr error in the final image increased
// by about 3%! This was surprising, because the RGB to YCbCr transform is linear, so the color line in one space should map to the
// same color line in the other space, but now you could estimate error in the same space as you measure final error, so now you can
// accurately account for chroma. However, it was worse! I did experiment to see if the loss of precision in calculating covariance
// mattered. Covariance is exact in floating point math for the unmodified RGB values, but not if we've converted to another color
// space where terms now use more than 8 mantissa bits. If the conversion rounds the last mantissa bit, we've already lost precision.
// It turns out precision didn't matter; using doubles for covariance and an epsilon around zero for the sign check had about 0.01%
// impact on total error, but should have made the math give the same results as infinite precision for the vast majority of blocks.
// So the only other difference I see in running the same code in two different color spaces is the color line it chooses. In both
// spaces it connects opposite corners of the color bounding box. My theory is that the color line you get by connecting corners of
// the RGB bounding box is much better than the color line from connecting corners of the YCbCr bounding box at approximating the
// optimal color line, and the more accurate line matters more than the more accurate error estimate. 

static const uniform int k_Y_int_sum = k_Y_R_int + k_Y_G_int + k_Y_B_int;
static const uniform int est_ratio = 10;
static const uniform int k_est_wt_r = k_Y_R_int * est_ratio + k_Y_int_sum;
static const uniform int k_est_wt_g = k_Y_G_int * est_ratio + k_Y_int_sum;
static const uniform int k_est_wt_b = k_Y_B_int * est_ratio + k_Y_int_sum;
static const uniform int k_est_wt_a = ((k_est_wt_r + k_est_wt_g + k_est_wt_b) * 2) / 3;	// alpha uses about twice the average rgb weight

static inline float remap_alpha(const uniform color_cell_compressor_params *uniform pParams, const varying float alpha)
{
	return clamp((alpha - pParams->m_alpha_bias) * pParams->m_alpha_scale, 0.0f, 255.0f);
}

static float compute_color_distance_rgb(const uniform color_cell_compressor_params *uniform pParams, const varying color_quad_i * uniform pSolid, uint32_t num_pixels, const varying color_quad_i *uniform pPixels)
{
	if (pParams->m_perceptual)
	{
		const float sy = pSolid->m_c[0] * k_Y_R + pSolid->m_c[1] * k_Y_G + pSolid->m_c[2] * k_Y_B;
		const float scr = pSolid->m_c[0] - sy;
		const float scb = pSolid->m_c[2] - sy;

		float sum_dy_dy = 0.0f;
		float sum_dcr_dcr = 0.0f;
		float sum_dcb_dcb = 0.0f;
		for (uniform uint32_t i = 0; i < num_pixels; i++)
		{
			const float iy = pPixels[i].m_c[0] * k_Y_R + pPixels[i].m_c[1] * k_Y_G + pPixels[i].m_c[2] * k_Y_B;
			const float icr = pPixels[i].m_c[0] - iy;
			const float icb = pPixels[i].m_c[2] - iy;

			float dy = iy - sy;
			float dcr = icr - scr;
			float dcb = icb - scb;

			sum_dy_dy += dy * dy;
			sum_dcr_dcr += dcr * dcr;
			sum_dcb_dcb += dcb * dcb;
		}
		return pParams->m_weights[0] * sum_dy_dy + pParams->m_weights[1] * (pr_weight * sum_dcr_dcr) + pParams->m_weights[2] * (pb_weight * sum_dcb_dcb);
	}
	else
	{
		const int16_t sr = pSolid->m_c[0];
		const int16_t sg = pSolid->m_c[1];
		const int16_t sb = pSolid->m_c[2];

		int32_t sum_dr_dr = 0;
		int32_t sum_dg_dg = 0;
		int32_t sum_db_db = 0;
		for (uniform uint32_t i = 0; i < num_pixels; i++)
		{
			int16_t dr = pPixels[i].m_c[0] - sr;
			int16_t dg = pPixels[i].m_c[1] - sg;
			int16_t db = pPixels[i].m_c[2] - sb;

			uint16_t drSq = dr * dr;	// conversion to unsigned keeps full precision of a 16-bit multiply
			uint16_t dgSq = dg * dg;
			uint16_t dbSq = db * db;

			sum_dr_dr += drSq;
			sum_dg_dg += dgSq;
			sum_db_db += dbSq;
		}
		const float w0_dr = pParams->m_weights[0] * (float)sum_dr_dr;
		const float w1_dg = pParams->m_weights[1] * (float)sum_dg_dg;
		const float w2_db = pParams->m_weights[2] * (float)sum_db_db;
		return w0_dr + w1_dg + w2_db;
	}
}

static inline float accum_color_distance_rgba_perceptual_color_times_alpha(const uniform color_cell_compressor_params *uniform pParams, const varying color_quad_i * uniform pPixel, float sy, float scr, float scb, float sa)
{
	const float ir = pPixel->m_c[0];
	const float ig = pPixel->m_c[1];
	const float ib = pPixel->m_c[2];
	const float ia = remap_alpha(pParams, pPixel->m_c[3]);

	const float iy = ir * k_Y_R + ig * k_Y_G + ib * k_Y_B;
	const float icr = ir - iy;
	const float icb = ib - iy;
	const float da = ia - sa;

	const float dy = iy * ia - sy * sa;
	const float dcr = icr * ia - scr * sa;
	const float dcb = icb * ia - scb * sa;

	vec3F point;
	point.m_c[0] = dy * pParams->m_sqrt_weights.m_c[0] + da * pParams->m_dycrcb_mid;
	point.m_c[1] = dcr * pParams->m_sqrt_weights.m_c[1];
	point.m_c[2] = dcb * pParams->m_sqrt_weights.m_c[2];

	vec3F delta_r = vec3F_mul(&pParams->m_dycrcb_r, copysign(da, vec3F_dot(&pParams->m_dycrcb_r, &point)));
	vec3F delta_g = vec3F_mul(&pParams->m_dycrcb_g, copysign(da, vec3F_dot(&pParams->m_dycrcb_g, &point)));
	vec3F delta_b = vec3F_mul(&pParams->m_dycrcb_b, copysign(da, vec3F_dot(&pParams->m_dycrcb_b, &point)));

	vec3F_accum(&point, &delta_r);
	vec3F_accum(&point, &delta_g);
	vec3F_accum(&point, &delta_b);
					
	return vec3F_dot(&point, &point);
}

static float compute_color_distance_rgba(const uniform color_cell_compressor_params *uniform pParams, const varying color_quad_i * uniform pSolid, uint32_t num_pixels, const varying color_quad_i *uniform pPixels)
{
	if (pParams->m_perceptual)
	{
		if (pParams->m_optimize_for_color_times_alpha)
		{
			const float sr = pSolid->m_c[0];
			const float sg = pSolid->m_c[1];
			const float sb = pSolid->m_c[2];
			const float sa = remap_alpha(pParams, pSolid->m_c[3]);

			const float sy = sr * k_Y_R + sg * k_Y_G + sb * k_Y_B;
			const float scr = sr - sy;
			const float scb = sb - sy;

			float total_err = 0.0f;
			for (uniform uint32_t i = 0; i < num_pixels; i++)
			{
				total_err += accum_color_distance_rgba_perceptual_color_times_alpha(pParams, &pPixels[i], sy, scr, scb, sa);
			}
			total_err *= (1.0f / (255.0f * 255.0f));
			return total_err;
		}
		else
		{
			const float sl = pSolid->m_c[0] * k_Y_R + pSolid->m_c[1] * k_Y_G + pSolid->m_c[2] * k_Y_B;
			const float scr = pSolid->m_c[0] - sl;
			const float scb = pSolid->m_c[2] - sl;
			const int16_t sa = pSolid->m_c[3];

			float sum_dl_dl = 0.0f;
			float sum_dcr_dcr = 0.0f;
			float sum_dcb_dcb = 0.0f;
			int32_t sum_da_da = 0;
			for (uniform uint32_t i = 0; i < num_pixels; i++)
			{
				const float il = pPixels[i].m_c[0] * k_Y_R + pPixels[i].m_c[1] * k_Y_G + pPixels[i].m_c[2] * k_Y_B;
				const float icr = pPixels[i].m_c[0] - il;
				const float icb = pPixels[i].m_c[2] - il;
				const int16_t ia = pPixels[i].m_c[3];

				float dl = il - sl;
				float dcr = icr - scr;
				float dcb = icb - scb;
				int16_t da = ia - sa;

				sum_dl_dl += dl * dl;
				sum_dcr_dcr += dcr * dcr;
				sum_dcb_dcb += dcb * dcb;
				uint16_t daSq = da * da;	// conversion to unsigned keeps full precision of a 16-bit multiply
				sum_da_da += daSq;
			}

			return pParams->m_weights[0] * sum_dl_dl + pParams->m_weights[1] * (pr_weight * sum_dcr_dcr) + pParams->m_weights[2] * (pb_weight * sum_dcb_dcb) + pParams->m_weights[3] * (float)sum_da_da;
		}
	}
	else
	{
		if (pParams->m_optimize_for_color_times_alpha)
		{
			assert(pParams->m_rotation == 0);
			const float sa = remap_alpha(pParams, pSolid->m_c[3]);
			const float sr_sa = pSolid->m_c[0] * sa;
			const float sg_sa = pSolid->m_c[1] * sa;
			const float sb_sa = pSolid->m_c[2] * sa;

			float sum_dr_dr = 0;
			float sum_dg_dg = 0;
			float sum_db_db = 0;
			for (uniform uint32_t i = 0; i < num_pixels; i++)
			{
				float ir = pPixels[i].m_c[0];
				float ig = pPixels[i].m_c[1];
				float ib = pPixels[i].m_c[2];
				float ia = remap_alpha(pParams, pPixels[i].m_c[3]);

				float dr_0 = ir * ia - sr_sa;
				float dg_0 = ig * ia - sg_sa;
				float db_0 = ib * ia - sb_sa;
				float da_255 = pParams->m_frame_buffer_other_value * (ia - sa);

				float dr = max(abs(dr_0), abs(dr_0 - da_255));
				float dg = max(abs(dg_0), abs(dg_0 - da_255));
				float db = max(abs(db_0), abs(db_0 - da_255));

				sum_dr_dr += dr * dr;
				sum_dg_dg += dg * dg;
				sum_db_db += db * db;
			}
			const float total_err = (pParams->m_weights[0] * sum_dr_dr + pParams->m_weights[1] * sum_dg_dg + pParams->m_weights[2] * sum_db_db) * (1.0f / (255.0f * 255.0f));
			return total_err;
		}
		else
		{
			const int16_t sr = pSolid->m_c[0];
			const int16_t sg = pSolid->m_c[1];
			const int16_t sb = pSolid->m_c[2];
			const int16_t sa = pSolid->m_c[3];

			int32_t sum_dr_dr = 0;
			int32_t sum_dg_dg = 0;
			int32_t sum_db_db = 0;
			int32_t sum_da_da = 0;
			for (uniform uint32_t i = 0; i < num_pixels; i++)
			{
				int16_t dr = pPixels[i].m_c[0] - sr;
				int16_t dg = pPixels[i].m_c[1] - sg;
				int16_t db = pPixels[i].m_c[2] - sb;
				int16_t da = pPixels[i].m_c[3] - sa;

				uint16_t drSq = dr * dr;	// conversion to unsigned keeps full precision of a 16-bit multiply
				uint16_t dgSq = dg * dg;
				uint16_t dbSq = db * db;
				uint16_t daSq = da * da;

				sum_dr_dr += drSq;
				sum_dg_dg += dgSq;
				sum_db_db += dbSq;
				sum_da_da += daSq;
			}
		
			const float w0_dr = pParams->m_weights[0] * (float)sum_dr_dr;
			const float w1_dg = pParams->m_weights[1] * (float)sum_dg_dg;
			const float w2_db = pParams->m_weights[2] * (float)sum_db_db;
			const float w3_da = pParams->m_weights[3] * (float)sum_da_da;
			return w0_dr + w1_dg + w2_db + w3_da;
		}
	}
}

static inline float convert_color_distance_rgb_to_ycrcb(float sum_dr_dr, float sum_dg_dg, float sum_db_db, float sum_dr_dg, float sum_dg_db, float sum_db_dr, uniform float wy, uniform float wcr, uniform float wcb)
{
	const uniform float ky_rr = k_Y_R * k_Y_R;
	const uniform float ky_gg = k_Y_G * k_Y_G;
	const uniform float ky_bb = k_Y_B * k_Y_B;
	const uniform float ky_rg = k_Y_R * k_Y_G * 2;
	const uniform float ky_gb = k_Y_G * k_Y_B * 2;
	const uniform float ky_br = k_Y_B * k_Y_R * 2;

	float sum_dy_dy = sum_dr_dr * ky_rr + sum_dg_dg * ky_gg + sum_db_db * ky_bb + sum_dr_dg * ky_rg + sum_dg_db * ky_gb + sum_db_dr * ky_br;

	// sum_dcr_dcr is the same, except every constant is negated, and ky_r --> 1 - ky_r.
	// If every term is negated the sign errors cancel, so we can use the "ky_*" constants if we just do ky_r --> ky_r - 1.
	// sum_dcb_dcb is the same as sum_dcr_dcr, except we use ky_b --> ky_b - 1.
	const uniform float kcr_rr = 2 * k_Y_R - 1;
	const uniform float kcr_rg = 2 * k_Y_G;
	const uniform float kcr_br = 2 * k_Y_B;
	const uniform float kcb_bb = 2 * k_Y_B - 1;
	const uniform float kcb_gb = 2 * k_Y_G;
	const uniform float kcb_br = 2 * k_Y_R;

	float sum_dcr_dcr = sum_dy_dy - sum_dr_dr * kcr_rr - sum_dr_dg * kcr_rg - sum_db_dr * kcr_br;
	float sum_dcb_dcb = sum_dy_dy - sum_db_db * kcb_bb - sum_dg_db * kcb_gb - sum_db_dr * kcb_br;

	return wy * sum_dy_dy + wcr * (pr_weight * sum_dcr_dcr) + wcb * (pb_weight * sum_dcb_dcb);
}

static float compute_color_distance_rgba_perceptual_with_rotation(const uniform color_cell_compressor_params *uniform pParams, const varying color_quad_i * uniform pSolid, uint32_t num_pixels, const varying color_quad_i *uniform pPixels, const varying int32_t *uniform pCompAlpha)
{
	assert(!pParams->m_optimize_for_color_times_alpha);
	assert(pParams->m_perceptual);
	assert(pParams->m_rotation != 0);
	assert(pCompAlpha != NULL);
	// Mode 4 and 5 have rotation. If this is non-zero, alpha is swapped with a color channel by the GPU. To reuse
	// compression code, this is handled by swapping the channels in pPixels and the weights in pParams->m_weights.
	// For alpha-blending aware error metrics, we need to get the channels back in the original spot, which means
	// doing the same swap again.
	//
	// rotation 0:  RGB|A: no swap, handled above
	// rotation 1:  AGB|R: chan_a = 0, chan_0 = 1, chan_1 = 2
	// rotation 2:  RAB|G: chan_a = 1, chan_0 = 0, chan_1 = 2
	// rotation 3:  RGA|B: chan_a = 2, chan_0 = 0, chan_1 = 1

	uniform const int32_t chan_a = pParams->m_rotation - 1;
	uniform const int32_t chan_0 = (3 - pParams->m_rotation) >> 1;
	uniform const int32_t chan_1 = (6 - pParams->m_rotation) >> 1;

	// In our naming scheme, s = source, i = iterator, 0-2 = color channel indexes (some permutation of RGB)
	const float sa = pSolid->m_c[chan_a];
	const float s0 = pSolid->m_c[chan_0];
	const float s1 = pSolid->m_c[chan_1];

	float sum_d0_d0 = 0;
	float sum_d1_d1 = 0;
	float sum_d2_d2 = 0;
	float sum_d0_d1 = 0;
	float sum_d1_d2 = 0;
	float sum_d2_d0 = 0;
	float sum_da_da = 0;
	for (uniform uint32_t i = 0; i < num_pixels; i++)
	{
		float s2 = pCompAlpha[i];

		float ia = pPixels[i].m_c[chan_a];
		float i0 = pPixels[i].m_c[chan_0];
		float i1 = pPixels[i].m_c[chan_1];
		float i2 = pPixels[i].m_c[3];

		float d0 = i0 - s0;
		float d1 = i1 - s1;
		float d2 = i2 - s2;
		float da = ia - sa;

		sum_d0_d0 += d0 * d0;
		sum_d1_d1 += d1 * d1;
		sum_d2_d2 += d2 * d2;
		sum_d0_d1 += d0 * d1;
		sum_d1_d2 += d1 * d2;
		sum_d2_d0 += d2 * d0;
		sum_da_da += da * da;
	}

/*
	// This is included for reference. Doing the "ifs" will turn into masked selects between 3 options for each constant.
	// The equivalent "unrolled" form below can use masked selects between 2 options for most constants, but it is not so
	// easy to see that it is correct. The easiest way to see it's correct is to compare the "unrolled" form to the "if"
	// form, and to compare the "if" form to the rotation logic above.
	if (pParams->m_rotation == 1)
	{
		sum_dr_dr = sum_d2_d2;
		sum_dg_dg = sum_d0_d0;
		sum_db_db = sum_d1_d1;
		sum_dr_dg = sum_d2_d0;
		sum_dg_db = sum_d0_d1;
		sum_db_dr = sum_d1_d2;
		wy = pParams->weights[3];
		wcb = pParams->weights[2];
		wcr = pParams->weights[1];
		wa = pParams->weights[0];
	}
	else if (pParams->m_rotation == 2)
	{
		sum_dr_dr = sum_d0_d0;
		sum_dg_dg = sum_d2_d2;
		sum_db_db = sum_d1_d1;
		sum_dr_dg = sum_d2_d0;
		sum_dg_db = sum_d1_d2;
		sum_db_dr = sum_d0_d1;
		wy = pParams->weights[0];
		wcb = pParams->weights[2];
		wcr = pParams->weights[3];
		wa = pParams->weights[1];
	}
	else // pParams->m_rotation == 3
	{
		sum_dr_dr = sum_d0_d0;
		sum_dg_dg = sum_d1_d1;
		sum_db_db = sum_d2_d2;
		sum_dr_dg = sum_d0_d1;
		sum_dg_db = sum_d1_d2;
		sum_db_dr = sum_d2_d0;
		wy = pParams->weights[0];
		wcb = pParams->weights[3];
		wcr = pParams->weights[1];
		wa = pParams->weights[2];
	}
*/
	const float sum_dr_dr = (pParams->m_rotation == 1) ? sum_d2_d2 : sum_d0_d0;
	const float sum_dg_dg = (pParams->m_rotation == 1) ? sum_d0_d0 : (pParams->m_rotation == 2) ? sum_d2_d2 : sum_d1_d1;
	const float sum_db_db = (pParams->m_rotation == 3) ? sum_d2_d2 : sum_d1_d1;
	const float sum_dr_dg = (pParams->m_rotation == 3) ? sum_d0_d1 : sum_d2_d0;
	const float sum_dg_db = (pParams->m_rotation == 1) ? sum_d0_d1 : sum_d1_d2;
	const float sum_db_dr = (pParams->m_rotation == 1) ? sum_d1_d2 : (pParams->m_rotation == 2) ? sum_d0_d1 : sum_d2_d0;
	uniform const float wy = (pParams->m_rotation == 1) ? pParams->m_weights[3] : pParams->m_weights[0];
	uniform const float wcb = (pParams->m_rotation == 3) ? pParams->m_weights[3] : pParams->m_weights[2];
	uniform const float wcr = (pParams->m_rotation == 2) ? pParams->m_weights[3] : pParams->m_weights[1];
	uniform const float wa = pParams->m_weights[pParams->m_rotation - 1];

	return convert_color_distance_rgb_to_ycrcb(sum_dr_dr, sum_dg_dg, sum_db_db, sum_dr_dg, sum_dg_db, sum_db_dr, wy, wcr, wcb) + wa * sum_da_da;
}

static float compute_color_distance_rgba_color_times_alpha_with_rotation(const uniform color_cell_compressor_params *uniform pParams, const varying color_quad_i * uniform pSolid, uint32_t num_pixels, const varying color_quad_i *uniform pPixels, const varying int32_t *uniform pCompAlpha)
{
	assert(pParams->m_optimize_for_color_times_alpha);
	assert(pCompAlpha != NULL);

	if (pParams->m_perceptual)
	{
		float total_err = 0.0f;
		if (pParams->m_rotation == 0)
		{
			const float sr = pSolid->m_c[0];
			const float sg = pSolid->m_c[1];
			const float sb = pSolid->m_c[2];

			const float sy = sr * k_Y_R + sg * k_Y_G + sb * k_Y_B;
			const float scr = sr - sy;
			const float scb = sb - sy;

			for (uniform uint32_t i = 0; i < num_pixels; i++)
			{
				const float sa = remap_alpha(pParams, pCompAlpha[i]);
				total_err += accum_color_distance_rgba_perceptual_color_times_alpha(pParams, &pPixels[i], sy, scr, scb, sa);
			}
		}
		else if (pParams->m_rotation == 1)
		{
			const float sa = remap_alpha(pParams, pSolid->m_c[0]);
			const float sg = pSolid->m_c[1];
			const float sb = pSolid->m_c[2];
			const float sy_com = sg * k_Y_G + sb * k_Y_B;

			for (uniform uint32_t i = 0; i < num_pixels; i++)
			{
				const float sr = pCompAlpha[i];
				const float sy = sy_com + sr * k_Y_R;
				const float scr = sr - sy;
				const float scb = sb - sy;
				total_err += accum_color_distance_rgba_perceptual_color_times_alpha(pParams, &pPixels[i], sy, scr, scb, sa);
			}
		}
		else if (pParams->m_rotation == 2)
		{
			const float sr = pSolid->m_c[0];
			const float sa = remap_alpha(pParams, pSolid->m_c[1]);
			const float sb = pSolid->m_c[2];
			const float sy_com = sr * k_Y_R + sb * k_Y_B;

			for (uniform uint32_t i = 0; i < num_pixels; i++)
			{
				const float sg = pCompAlpha[i];
				const float sy = sy_com + sg * k_Y_G;
				const float scr = sr - sy;
				const float scb = sb - sy;
				total_err += accum_color_distance_rgba_perceptual_color_times_alpha(pParams, &pPixels[i], sy, scr, scb, sa);
			}
		}
		else // pParams->m_rotation == 3
		{
			const float sr = pSolid->m_c[0];
			const float sg = pSolid->m_c[1];
			const float sa = remap_alpha(pParams, pSolid->m_c[2]);
			const float sy_com = sr * k_Y_R + sg * k_Y_G;

			for (uniform uint32_t i = 0; i < num_pixels; i++)
			{
				const float sb = pCompAlpha[i];
				const float sy = sy_com + sb * k_Y_B;
				const float scr = sr - sy;
				const float scb = sb - sy;
				total_err += accum_color_distance_rgba_perceptual_color_times_alpha(pParams, &pPixels[i], sy, scr, scb, sa);
			}
		}
		total_err *= 1.0f / (255.0f * 255.0f);
		return total_err;
	}

	if (pParams->m_rotation == 0)
	{
		// In our naming scheme, s = source, i = iterator, 0-2 = color channel indexes (some permutation of RGB)
		const float sr = pSolid->m_c[0];
		const float sg = pSolid->m_c[1];
		const float sb = pSolid->m_c[2];

		float sum_dr_dr = 0;
		float sum_dg_dg = 0;
		float sum_db_db = 0;
		for (uniform uint32_t i = 0; i < num_pixels; i++)
		{
			float sa = remap_alpha(pParams, pCompAlpha[i]);

			float ir = pPixels[i].m_c[0];
			float ig = pPixels[i].m_c[1];
			float ib = pPixels[i].m_c[2];
			float ia = remap_alpha(pParams, pPixels[i].m_c[3]);

			float dr_0 = ir * ia - sr * sa;
			float dg_0 = ig * ia - sg * sa;
			float db_0 = ib * ia - sb * sa;
			float da_255 = pParams->m_frame_buffer_other_value * (ia - sa);

			float dr = max(abs(dr_0), abs(dr_0 - da_255));
			float dg = max(abs(dg_0), abs(dg_0 - da_255));
			float db = max(abs(db_0), abs(db_0 - da_255));

			sum_dr_dr += dr * dr;
			sum_dg_dg += dg * dg;
			sum_db_db += db * db;
		}

		const float total_err = (pParams->m_weights[0] * sum_dr_dr + pParams->m_weights[1] * sum_dg_dg + pParams->m_weights[2] * sum_db_db) * (1.0f / (255.0f * 255.0f));
		return total_err;
	}
	else
	{
		// Mode 4 and 5 have rotation. If this is non-zero, alpha is swapped with a color channel by the GPU. To reuse
		// compression code, this is handled by swapping the channels in pPixels and the weights in pParams->m_weights.
		// For alpha-blending aware error metrics, we need to get the channels back in the original spot, which means
		// doing the same swap again.
		//
		// rotation 0:  RGB|A: no swap, handled above
		// rotation 1:  AGB|R: chan_a = 0, chan_0 = 1, chan_1 = 2
		// rotation 2:  RAB|G: chan_a = 1, chan_0 = 0, chan_1 = 2
		// rotation 3:  RGA|B: chan_a = 2, chan_0 = 0, chan_1 = 1

		uniform const int32_t chan_a = pParams->m_rotation - 1;
		uniform const int32_t chan_0 = (3 - pParams->m_rotation) >> 1;
		uniform const int32_t chan_1 = (6 - pParams->m_rotation) >> 1;

		// In our naming scheme, s = source, i = iterator, 0-2 = color channel indexes (some permutation of RGB)
		const float sa = remap_alpha(pParams, pSolid->m_c[chan_a]);
		const float s0 = pSolid->m_c[chan_0];
		const float s1 = pSolid->m_c[chan_1];

		float sum_d0_d0 = 0;
		float sum_d1_d1 = 0;
		float sum_d2_d2 = 0;
		for (uniform uint32_t i = 0; i < num_pixels; i++)
		{
			float s2 = pCompAlpha[i];

			float ia = remap_alpha(pParams, pPixels[i].m_c[chan_a]);
			float i0 = pPixels[i].m_c[chan_0];
			float i1 = pPixels[i].m_c[chan_1];
			float i2 = pPixels[i].m_c[3];

			float d0_0 = i0 * ia - s0 * sa;
			float d1_0 = i1 * ia - s1 * sa;
			float d2_0 = i2 * ia - s2 * sa;
			float da_255 = pParams->m_frame_buffer_other_value * (ia - sa);

			float d0 = max(abs(d0_0), abs(d0_0 - da_255));
			float d1 = max(abs(d1_0), abs(d1_0 - da_255));
			float d2 = max(abs(d2_0), abs(d2_0 - da_255));

			sum_d0_d0 += d0 * d0;
			sum_d1_d1 += d1 * d1;
			sum_d2_d2 += d2 * d2;
		}
		float total_err = pParams->m_weights[chan_0] * sum_d0_d0 + pParams->m_weights[chan_1] * sum_d1_d1 + pParams->m_weights[3] * sum_d2_d2;
		total_err *= 1.0f / (255.0f * 255.0f);
		return total_err;
	}
}

static float pack_mode1_to_one_color(const uniform color_cell_compressor_params *uniform pParams, varying color_cell_compressor_results *uniform pResults, uint32_t r, uint32_t g, uint32_t b, 
	varying int *uniform pSelectors, uint32_t num_pixels, const varying color_quad_i *uniform pPixels)
{
	uint32_t best_err = UINT32_MAX;
	uint32_t best_p = 0;

	for (uniform uint32_t p = 0; p < 2; p++)
	{
#pragma ignore warning(perf)
		uint32_t err = g_bc7_mode_1_optimal_endpoints[r][p].m_error + g_bc7_mode_1_optimal_endpoints[g][p].m_error + g_bc7_mode_1_optimal_endpoints[b][p].m_error;
		if (err < best_err)
		{
			best_err = err;
			best_p = p;
		}
	}

	const endpoint_err *pEr = &g_bc7_mode_1_optimal_endpoints[r][best_p];
	const endpoint_err *pEg = &g_bc7_mode_1_optimal_endpoints[g][best_p];
	const endpoint_err *pEb = &g_bc7_mode_1_optimal_endpoints[b][best_p];

#pragma ignore warning(perf)
	color_quad_i_set(&pResults->m_low_endpoint, pEr->m_lo, pEg->m_lo, pEb->m_lo, 0);
#pragma ignore warning(perf)
	color_quad_i_set(&pResults->m_high_endpoint, pEr->m_hi, pEg->m_hi, pEb->m_hi, 0);
	pResults->m_pbits[0] = best_p;
	pResults->m_pbits[1] = 0;

	for (uniform uint32_t i = 0; i < num_pixels; i++)
		pSelectors[i] = BC7E_MODE_1_OPTIMAL_INDEX;

	color_quad_i p;
	
	for (uniform uint32_t i = 0; i < 3; i++)
	{
		uint32_t low = ((pResults->m_low_endpoint.m_c[i] << 1) | pResults->m_pbits[0]) << 1;
		low |= (low >> 7);

		uint32_t high = ((pResults->m_high_endpoint.m_c[i] << 1) | pResults->m_pbits[0]) << 1;
		high |= (high >> 7);

		p.m_c[i] = (low * (64 - g_bc7_weights3[BC7E_MODE_1_OPTIMAL_INDEX]) + high * g_bc7_weights3[BC7E_MODE_1_OPTIMAL_INDEX] + 32) >> 6;
	}

	p.m_c[3] = 255;

	const float total_err = compute_color_distance_rgb(pParams, &p, num_pixels, pPixels);

	pResults->m_best_overall_err = total_err;

	return total_err;
}

static float pack_mode24_to_one_color(const uniform color_cell_compressor_params *uniform pParams, varying color_cell_compressor_results *uniform pResults, uint32_t r, uint32_t g, uint32_t b, 
	varying int *uniform pSelectors, uint32_t num_pixels, const varying color_quad_i *uniform pPixels, const varying int32_t *uniform pCompAlpha)
{
	assert(pParams->m_rotation == 0 || pCompAlpha != NULL);

	uint32_t er, eg, eb;

	if (pParams->m_num_selector_weights == 8)
	{
		er = g_bc7_mode_4_optimal_endpoints3[r];
		eg = g_bc7_mode_4_optimal_endpoints3[g];
		eb = g_bc7_mode_4_optimal_endpoints3[b];
	}
	else
	{
		er = g_bc7_mode_4_optimal_endpoints2[r];
		eg = g_bc7_mode_4_optimal_endpoints2[g];
		eb = g_bc7_mode_4_optimal_endpoints2[b];
	}
	
	color_quad_i_set(&pResults->m_low_endpoint, er & 0xFF, eg & 0xFF, eb & 0xFF, 0);
	color_quad_i_set(&pResults->m_high_endpoint, er >> 8, eg >> 8, eb >> 8, 0);

	for (uniform uint32_t i = 0; i < num_pixels; i++)
		pSelectors[i] = (pParams->m_num_selector_weights == 8) ? BC7E_MODE_4_OPTIMAL_INDEX3 : BC7E_MODE_4_OPTIMAL_INDEX2;

	color_quad_i p;
	
	for (uniform uint32_t i = 0; i < 3; i++)
	{
		uint32_t low = pResults->m_low_endpoint.m_c[i] << 3;
		low |= (low >> 5);

		uint32_t high = pResults->m_high_endpoint.m_c[i] << 3;
		high |= (high >> 5);

		if (pParams->m_num_selector_weights == 8)
			p.m_c[i] = (low * (64 - g_bc7_weights3[BC7E_MODE_4_OPTIMAL_INDEX3]) + high * g_bc7_weights3[BC7E_MODE_4_OPTIMAL_INDEX3] + 32) >> 6;
		else
			p.m_c[i] = (low * (64 - g_bc7_weights2[BC7E_MODE_4_OPTIMAL_INDEX2]) + high * g_bc7_weights2[BC7E_MODE_4_OPTIMAL_INDEX2] + 32) >> 6;
	}
	
	p.m_c[3] = 255;

	float total_err;
	if (pCompAlpha != NULL && pParams->m_optimize_for_color_times_alpha)
		total_err = compute_color_distance_rgba_color_times_alpha_with_rotation(pParams, &p, num_pixels, pPixels, pCompAlpha);
	else if (pCompAlpha != NULL && pParams->m_rotation != 0 && pParams->m_perceptual)
		total_err = compute_color_distance_rgba_perceptual_with_rotation(pParams, &p, num_pixels, pPixels, pCompAlpha);
	else
		total_err = compute_color_distance_rgb(pParams, &p, num_pixels, pPixels);

	pResults->m_best_overall_err = total_err;

	return total_err;
}

static float pack_mode0_to_one_color(const uniform color_cell_compressor_params *uniform pParams, varying color_cell_compressor_results *uniform pResults, uint32_t r, uint32_t g, uint32_t b, 
	varying int *uniform pSelectors, uint32_t num_pixels, const varying color_quad_i *uniform pPixels)
{
	uint32_t best_err = UINT32_MAX;
	uint32_t best_p = 0;

	for (uniform uint32_t p = 0; p < 4; p++)
	{
#pragma ignore warning(perf)
		uint32_t err = g_bc7_mode_0_optimal_endpoints[r][p >> 1][p & 1].m_error + g_bc7_mode_0_optimal_endpoints[g][p >> 1][p & 1].m_error + g_bc7_mode_0_optimal_endpoints[b][p >> 1][p & 1].m_error;
		if (err < best_err)
		{
			best_err = err;
			best_p = p;
		}
	}

	const endpoint_err *pEr = &g_bc7_mode_0_optimal_endpoints[r][best_p >> 1][best_p & 1];
	const endpoint_err *pEg = &g_bc7_mode_0_optimal_endpoints[g][best_p >> 1][best_p & 1];
	const endpoint_err *pEb = &g_bc7_mode_0_optimal_endpoints[b][best_p >> 1][best_p & 1];

#pragma ignore warning(perf)
	color_quad_i_set(&pResults->m_low_endpoint, pEr->m_lo, pEg->m_lo, pEb->m_lo, 0);

#pragma ignore warning(perf)
	color_quad_i_set(&pResults->m_high_endpoint, pEr->m_hi, pEg->m_hi, pEb->m_hi, 0);

	pResults->m_pbits[0] = best_p & 1;
	pResults->m_pbits[1] = best_p >> 1;

	for (uniform uint32_t i = 0; i < num_pixels; i++)
		pSelectors[i] = BC7E_MODE_0_OPTIMAL_INDEX;

	color_quad_i p;
	
	for (uniform uint32_t i = 0; i < 3; i++)
	{
		uint32_t low = ((pResults->m_low_endpoint.m_c[i] << 1) | pResults->m_pbits[0]) << 3;
		low |= (low >> 5);

		uint32_t high = ((pResults->m_high_endpoint.m_c[i] << 1) | pResults->m_pbits[1]) << 3;
		high |= (high >> 5);

		p.m_c[i] = (low * (64 - g_bc7_weights3[BC7E_MODE_0_OPTIMAL_INDEX]) + high * g_bc7_weights3[BC7E_MODE_0_OPTIMAL_INDEX] + 32) >> 6;
	}
	
	p.m_c[3] = 255;

	const float total_err = compute_color_distance_rgb(pParams, &p, num_pixels, pPixels);

	pResults->m_best_overall_err = total_err;

	return total_err;
}

static float pack_mode6_to_one_color(const uniform color_cell_compressor_params *uniform pParams, varying color_cell_compressor_results *uniform pResults, uint32_t r, uint32_t g, uint32_t b, uint32_t a,
	varying int *uniform pSelectors, uint32_t num_pixels, const varying color_quad_i *uniform pPixels)
{
	uint32_t best_err = UINT32_MAX;
	uint32_t best_p = 0;

	for (uniform uint32_t p = 0; p < 4; p++)
	{
		uniform uint32_t hi_p = p >> 1;
		uniform uint32_t lo_p = p & 1;
#pragma ignore warning(perf)
		uint32_t err = g_bc7_mode_6_optimal_endpoints[r][hi_p][lo_p].m_error + g_bc7_mode_6_optimal_endpoints[g][hi_p][lo_p].m_error + g_bc7_mode_6_optimal_endpoints[b][hi_p][lo_p].m_error + g_bc7_mode_6_optimal_endpoints[a][hi_p][lo_p].m_error;
		if (err < best_err)
		{
			best_err = err;
			best_p = p;
		}
	}

	uint32_t best_hi_p = best_p >> 1;
	uint32_t best_lo_p = best_p & 1;

	const endpoint_err *pEr = &g_bc7_mode_6_optimal_endpoints[r][best_hi_p][best_lo_p];
	const endpoint_err *pEg = &g_bc7_mode_6_optimal_endpoints[g][best_hi_p][best_lo_p];
	const endpoint_err *pEb = &g_bc7_mode_6_optimal_endpoints[b][best_hi_p][best_lo_p];
	const endpoint_err *pEa = &g_bc7_mode_6_optimal_endpoints[a][best_hi_p][best_lo_p];

#pragma ignore warning(perf)
	color_quad_i_set(&pResults->m_low_endpoint, pEr->m_lo, pEg->m_lo, pEb->m_lo, pEa->m_lo);

#pragma ignore warning(perf)
	color_quad_i_set(&pResults->m_high_endpoint, pEr->m_hi, pEg->m_hi, pEb->m_hi, pEa->m_hi);

	pResults->m_pbits[0] = best_lo_p;
	pResults->m_pbits[1] = best_hi_p;

	for (uniform uint32_t i = 0; i < num_pixels; i++)
		pSelectors[i] = BC7E_MODE_6_OPTIMAL_INDEX;

	color_quad_i p;
	
	for (uniform uint32_t i = 0; i < 4; i++)
	{
		uint32_t low = (pResults->m_low_endpoint.m_c[i] << 1) | pResults->m_pbits[0];
		uint32_t high = (pResults->m_high_endpoint.m_c[i] << 1) | pResults->m_pbits[1];
		
		p.m_c[i] = (low * (64 - g_bc7_weights4[BC7E_MODE_6_OPTIMAL_INDEX]) + high * g_bc7_weights4[BC7E_MODE_6_OPTIMAL_INDEX] + 32) >> 6;
	}

	const float total_err = compute_color_distance_rgba(pParams, &p, num_pixels, pPixels);

	pResults->m_best_overall_err = total_err;

	return total_err;
}

static float pack_mode7_to_one_color(const uniform color_cell_compressor_params *uniform pParams, varying color_cell_compressor_results *uniform pResults, uint32_t r, uint32_t g, uint32_t b, uint32_t a,
	varying int *uniform pSelectors, uint32_t num_pixels, const varying color_quad_i *uniform pPixels)
{
	uint32_t best_err = UINT32_MAX;
	uint32_t best_p = 0;

	for (uniform uint32_t p = 0; p < 4; p++)
	{
		uniform uint32_t hi_p = p >> 1;
		uniform uint32_t lo_p = p & 1;
#pragma ignore warning(perf)
		uint32_t err = g_bc7_mode_7_optimal_endpoints[r][hi_p][lo_p].m_error + g_bc7_mode_7_optimal_endpoints[g][hi_p][lo_p].m_error + g_bc7_mode_7_optimal_endpoints[b][hi_p][lo_p].m_error + g_bc7_mode_7_optimal_endpoints[a][hi_p][lo_p].m_error;
		if (err < best_err)
		{
			best_err = err;
			best_p = p;
		}
	}

	uint32_t best_hi_p = best_p >> 1;
	uint32_t best_lo_p = best_p & 1;

	const endpoint_err *pEr = &g_bc7_mode_7_optimal_endpoints[r][best_hi_p][best_lo_p];
	const endpoint_err *pEg = &g_bc7_mode_7_optimal_endpoints[g][best_hi_p][best_lo_p];
	const endpoint_err *pEb = &g_bc7_mode_7_optimal_endpoints[b][best_hi_p][best_lo_p];
	const endpoint_err *pEa = &g_bc7_mode_7_optimal_endpoints[a][best_hi_p][best_lo_p];

#pragma ignore warning(perf)
	color_quad_i_set(&pResults->m_low_endpoint, pEr->m_lo, pEg->m_lo, pEb->m_lo, pEa->m_lo);

#pragma ignore warning(perf)
	color_quad_i_set(&pResults->m_high_endpoint, pEr->m_hi, pEg->m_hi, pEb->m_hi, pEa->m_hi);

	pResults->m_pbits[0] = best_lo_p;
	pResults->m_pbits[1] = best_hi_p;

	for (uniform uint32_t i = 0; i < num_pixels; i++)
		pSelectors[i] = BC7E_MODE_7_OPTIMAL_INDEX;

	color_quad_i p;
	
	for (uniform uint32_t i = 0; i < 4; i++)
	{
		uint32_t low = (pResults->m_low_endpoint.m_c[i] << 1) | pResults->m_pbits[0];
		uint32_t high = (pResults->m_high_endpoint.m_c[i] << 1) | pResults->m_pbits[1];
		
		p.m_c[i] = (low * (64 - g_bc7_weights2[BC7E_MODE_7_OPTIMAL_INDEX]) + high * g_bc7_weights2[BC7E_MODE_7_OPTIMAL_INDEX] + 32) >> 6;
	}

	float total_err = compute_color_distance_rgba(pParams, &p, num_pixels, pPixels);

	pResults->m_best_overall_err = total_err;

	return total_err;
}

static inline float evaluate_error_orthogonal(varying float qr, varying float qg, varying float qb, varying float pr, varying float pg, varying float pb, uniform float wr, uniform float wg, uniform float wb)
{
	float dr = qr - pr;
	float dg = qg - pg;
	float db = qb - pb;
	return wr * dr * dr + wg * dg * dg + wb * db * db;
}

static inline float evaluate_error_orthogonal(varying float qr, varying float qg, varying float qb, varying float qa, varying float pr, varying float pg, varying float pb, varying float pa, uniform float wr, uniform float wg, uniform float wb, uniform float wa)
{
	float dr = qr - pr;
	float dg = qg - pg;
	float db = qb - pb;
	float da = qa - pa;
	return wr * dr * dr + wg * dg * dg + wb * db * db + wa * da * da;
}

static inline float evaluate_error_orthogonal(const varying color_quad_f *uniform paletteColor, varying float pr, varying float pg, varying float pb, varying float pa, uniform float wr, uniform float wg, uniform float wb, uniform float wa)
{
	return evaluate_error_orthogonal(paletteColor->m_c[0], paletteColor->m_c[1], paletteColor->m_c[2], paletteColor->m_c[3], pr, pg, pb, pa, wr, wg, wb, wa);
}

static inline float evaluate_error_orthogonal(const varying color_quad_f *varying paletteColor, varying float pr, varying float pg, varying float pb, varying float pa, uniform float wr, uniform float wg, uniform float wb, uniform float wa)
{
#pragma ignore warning(perf)
	float qr = paletteColor->m_c[0];
#pragma ignore warning(perf)
	float qg = paletteColor->m_c[1];
#pragma ignore warning(perf)
	float qb = paletteColor->m_c[2];
#pragma ignore warning(perf)
	float qa = paletteColor->m_c[3];

	return evaluate_error_orthogonal(qr, qg, qb, qa, pr, pg, pb, pa, wr, wg, wb, wa);
}

static inline float evaluate_error_alpha_blend(varying float qr, varying float qg, varying float qb, varying float qa, varying float pr, varying float pg, varying float pb, varying float pa, uniform float wr, uniform float wg, uniform float wb, uniform float wa, const uniform color_cell_compressor_params *uniform pParams)
{
	float da_255, dr_0, dg_0, db_0;
	if (pParams->m_rotation == 0)
	{
		da_255 = pParams->m_frame_buffer_other_value * (remap_alpha(pParams, qa) - remap_alpha(pParams, pa));
		dr_0 = qr * qa - pr * pa;
		dg_0 = qg * qa - pg * pa;
		db_0 = qb * qa - pb * pa;
	}
	else if (pParams->m_rotation == 1)
	{
		// r and a are swapped
		da_255 = pParams->m_frame_buffer_other_value * (remap_alpha(pParams, qr) - remap_alpha(pParams, pr));
		dr_0 = qa * qr - pa * pr;
		dg_0 = qg * qr - pg * pr;
		db_0 = qb * qr - pb * pr;
		wr = wa;
	}
	else if (pParams->m_rotation == 2)
	{
		// g and a are swapped
		da_255 = pParams->m_frame_buffer_other_value * (remap_alpha(pParams, qg) - remap_alpha(pParams, pg));
		dr_0 = qr * qg - pr * pg;
		dg_0 = qa * qg - pa * pg;
		db_0 = qb * qg - pb * pg;
		wg = wa;
	}
	else // pParams->m_rotation == 3
	{
		// b and a are swapped
		da_255 = pParams->m_frame_buffer_other_value * (remap_alpha(pParams, qb) - remap_alpha(pParams, pb));
		dr_0 = qr * qb - pr * pb;
		dg_0 = qg * qb - pg * pb;
		db_0 = qa * qb - pa * pb;
		wb = wa;
	}

	const float dr = max(abs(dr_0), abs(dr_0 - da_255));
	const float dg = max(abs(dg_0), abs(dg_0 - da_255));
	const float db = max(abs(db_0), abs(db_0 - da_255));
	return (wr * dr * dr + wg * dg * dg + wb * db * db) * (1.0f / (255.0f * 255.0f));
}

static inline float evaluate_error_alpha_blend(const varying color_quad_f *uniform paletteColor, varying float pr, varying float pg, varying float pb, varying float pa, uniform float wr, uniform float wg, uniform float wb, uniform float wa, const uniform color_cell_compressor_params *uniform pParams)
{
	return evaluate_error_alpha_blend(paletteColor->m_c[0], paletteColor->m_c[1], paletteColor->m_c[2], paletteColor->m_c[3], pr, pg, pb, pa, wr, wg, wb, wa, pParams);
}

static inline float evaluate_error_alpha_blend(const varying color_quad_f *varying paletteColor, varying float pr, varying float pg, varying float pb, varying float pa, uniform float wr, uniform float wg, uniform float wb, uniform float wa, const uniform color_cell_compressor_params *uniform pParams)
{
#pragma ignore warning(perf)
	float qr = paletteColor->m_c[0];
#pragma ignore warning(perf)
	float qg = paletteColor->m_c[1];
#pragma ignore warning(perf)
	float qb = paletteColor->m_c[2];
#pragma ignore warning(perf)
	float qa = paletteColor->m_c[3];

	return evaluate_error_alpha_blend(qr, qg, qb, qa, pr, pg, pb, pa, wr, wg, wb, wa, pParams);
}

static float evaluate_solution(const varying color_quad_i *uniform pLow, const varying color_quad_i *uniform pHigh, const varying uint32_t *uniform pbits, 
	const uniform color_cell_compressor_params *uniform pParams, varying color_cell_compressor_results *uniform pResults, uint32_t num_pixels, const varying color_quad_i *uniform pPixels, const varying int32_t *uniform pCompAlpha)
{
	color_quad_i quantMinColor = *pLow;
	color_quad_i quantMaxColor = *pHigh;

	if (pParams->m_has_pbits)
	{
		uint32_t minPBit, maxPBit;

		if (pParams->m_endpoints_share_pbit)
			maxPBit = minPBit = pbits[0];
		else
		{
			minPBit = pbits[0];
			maxPBit = pbits[1];
		}

		quantMinColor.m_c[0] = (pLow->m_c[0] << 1) | minPBit;
		quantMinColor.m_c[1] = (pLow->m_c[1] << 1) | minPBit;
		quantMinColor.m_c[2] = (pLow->m_c[2] << 1) | minPBit;
		quantMinColor.m_c[3] = (pLow->m_c[3] << 1) | minPBit;

		quantMaxColor.m_c[0] = (pHigh->m_c[0] << 1) | maxPBit;
		quantMaxColor.m_c[1] = (pHigh->m_c[1] << 1) | maxPBit;
		quantMaxColor.m_c[2] = (pHigh->m_c[2] << 1) | maxPBit;
		quantMaxColor.m_c[3] = (pHigh->m_c[3] << 1) | maxPBit;
	}

	const uniform uint32_t nc = pParams->m_has_alpha ? 4 : 3;
	color_quad_i actualMinColor = scale_color(&quantMinColor, pParams, nc);
	color_quad_i actualMaxColor = scale_color(&quantMaxColor, pParams, nc);

	const uniform uint32_t N = pParams->m_num_selector_weights;
	const uniform bool optimizeForColorTimesAlpha_SeparateAlpha = (pParams->m_optimize_for_color_times_alpha && pCompAlpha != NULL);

	float total_err = 0;

	uniform float wr = pParams->m_weights[0];
	uniform float wg = pParams->m_weights[1];
	uniform float wb = pParams->m_weights[2];
	uniform float wa = pParams->m_weights[3];

	color_quad_f weightedColors[16];
	weightedColors[0].m_c[0] = actualMinColor.m_c[0];
	weightedColors[0].m_c[1] = actualMinColor.m_c[1];
	weightedColors[0].m_c[2] = actualMinColor.m_c[2];
	weightedColors[0].m_c[3] = actualMinColor.m_c[3];

	weightedColors[N - 1].m_c[0] = actualMaxColor.m_c[0];
	weightedColors[N - 1].m_c[1] = actualMaxColor.m_c[1];
	weightedColors[N - 1].m_c[2] = actualMaxColor.m_c[2];
	weightedColors[N - 1].m_c[3] = actualMaxColor.m_c[3];
		
	for (uniform uint32_t i = 1; i < (N - 1); i++)
		for (uniform uint32_t j = 0; j < nc; j++)
			weightedColors[i].m_c[j] = round((weightedColors[0].m_c[j] * (64.0f - pParams->m_pSelector_weights[i]) + weightedColors[N - 1].m_c[j] * pParams->m_pSelector_weights[i]) * (1.0f / 64.0f));

	if (!pParams->m_perceptual)
	{
		if (!pParams->m_has_alpha)
		{
			if (N == 16)
			{
				float lr = actualMinColor.m_c[0];
				float lg = actualMinColor.m_c[1];
				float lb = actualMinColor.m_c[2];

				float dr = actualMaxColor.m_c[0] - lr;
				float dg = actualMaxColor.m_c[1] - lg;
				float db = actualMaxColor.m_c[2] - lb;
			
				const float f = N / (dr * dr + dg * dg + db * db);

				lr *= -dr;
				lg *= -dg;
				lb *= -db;

				cfor (uniform uint32_t i = 0; i < num_pixels; i++)
				{
					const varying color_quad_i *uniform pC = &pPixels[i];
					float r = pC->m_c[0];
					float g = pC->m_c[1];
					float b = pC->m_c[2];

					float best_sel = round(((r * dr + lr) + (g * dg + lg) + (b * db + lb)) * f);
					best_sel = clamp(best_sel, (float)1, (float)(N - 1));

					float best_sel0 = best_sel - 1;

					float err0, err1;
					if (optimizeForColorTimesAlpha_SeparateAlpha)
					{
						const float a = pC->m_c[3];
						const float qa = pCompAlpha[i];

#pragma ignore warning(perf)
						float qr0 = weightedColors[(int)best_sel0].m_c[0];
#pragma ignore warning(perf)
						float qg0 = weightedColors[(int)best_sel0].m_c[1];
#pragma ignore warning(perf)
						float qb0 = weightedColors[(int)best_sel0].m_c[2];

						err0 = evaluate_error_alpha_blend(qr0, qg0, qb0, qa, r, g, b, a, wr, wg, wb, wa, pParams);

#pragma ignore warning(perf)
						float qr1 = weightedColors[(int)best_sel].m_c[0];
#pragma ignore warning(perf)
						float qg1 = weightedColors[(int)best_sel].m_c[1];
#pragma ignore warning(perf)
						float qb1 = weightedColors[(int)best_sel].m_c[2];

						err1 = evaluate_error_alpha_blend(qr1, qg1, qb1, qa, r, g, b, a, wr, wg, wb, wa, pParams);
					}
					else
					{
#pragma ignore warning(perf)
						float qr0 = weightedColors[(int)best_sel0].m_c[0] - r;
#pragma ignore warning(perf)
						float qg0 = weightedColors[(int)best_sel0].m_c[1] - g;
#pragma ignore warning(perf)
						float qb0 = weightedColors[(int)best_sel0].m_c[2] - b;

						err0 = evaluate_error_orthogonal(qr0, qg0, qb0, r, g, b, wr, wg, wb);

#pragma ignore warning(perf)
						float qr1 = weightedColors[(int)best_sel].m_c[0];
#pragma ignore warning(perf)
						float qg1 = weightedColors[(int)best_sel].m_c[1];
#pragma ignore warning(perf)
						float qb1 = weightedColors[(int)best_sel].m_c[2];

						err1 = evaluate_error_orthogonal(qr1, qg1, qb1, r, g, b, wr, wg, wb);
					}

					float min_err = min(err0, err1);
					total_err += min_err;
					pResults->m_pSelectors_temp[i] = (int)select(min_err == err0, best_sel0, best_sel);
				}
			}
			else if (N == 8)
			{
				cfor (uniform uint32_t i = 0; i < num_pixels; i++)
				{
					float pr = (float)pPixels[i].m_c[0];
					float pg = (float)pPixels[i].m_c[1];
					float pb = (float)pPixels[i].m_c[2];
				
					float best_err;
					int best_sel;

					if (optimizeForColorTimesAlpha_SeparateAlpha)
					{
						float pa = (float)pPixels[i].m_c[3];
						float qa = pCompAlpha[i];
						{
							float qr0 = weightedColors[0].m_c[0];
							float qg0 = weightedColors[0].m_c[1];
							float qb0 = weightedColors[0].m_c[2];
							float err0 = evaluate_error_alpha_blend(qr0, qg0, qb0, qa, pr, pg, pb, pa, wr, wg, wb, wa, pParams);

							float qr1 = weightedColors[1].m_c[0];
							float qg1 = weightedColors[1].m_c[1];
							float qb1 = weightedColors[1].m_c[2];
							float err1 = evaluate_error_alpha_blend(qr1, qg1, qb1, qa, pr, pg, pb, pa, wr, wg, wb, wa, pParams);

							float qr2 = weightedColors[2].m_c[0];
							float qg2 = weightedColors[2].m_c[1];
							float qb2 = weightedColors[2].m_c[2];
							float err2 = evaluate_error_alpha_blend(qr2, qg2, qb2, qa, pr, pg, pb, pa, wr, wg, wb, wa, pParams);

							float qr3 = weightedColors[3].m_c[0];
							float qg3 = weightedColors[3].m_c[1];
							float qb3 = weightedColors[3].m_c[2];
							float err3 = evaluate_error_alpha_blend(qr3, qg3, qb3, qa, pr, pg, pb, pa, wr, wg, wb, wa, pParams);

							best_err = min(min(min(err0, err1), err2), err3);
									
							best_sel = select(best_err == err1, 1, 0);
							best_sel = select(best_err == err2, 2, best_sel);
							best_sel = select(best_err == err3, 3, best_sel);
						}

						{
							float qr4 = weightedColors[4].m_c[0];
							float qg4 = weightedColors[4].m_c[1];
							float qb4 = weightedColors[4].m_c[2];
							float err4 = evaluate_error_alpha_blend(qr4, qg4, qb4, qa, pr, pg, pb, pa, wr, wg, wb, wa, pParams);

							float qr5 = weightedColors[5].m_c[0];
							float qg5 = weightedColors[5].m_c[1];
							float qb5 = weightedColors[5].m_c[2];
							float err5 = evaluate_error_alpha_blend(qr5, qg5, qb5, qa, pr, pg, pb, pa, wr, wg, wb, wa, pParams);

							float qr6 = weightedColors[6].m_c[0];
							float qg6 = weightedColors[6].m_c[1];
							float qb6 = weightedColors[6].m_c[2];
							float err6 = evaluate_error_alpha_blend(qr6, qg6, qb6, qa, pr, pg, pb, pa, wr, wg, wb, wa, pParams);

							float qr7 = weightedColors[7].m_c[0];
							float qg7 = weightedColors[7].m_c[1];
							float qb7 = weightedColors[7].m_c[2];
							float err7 = evaluate_error_alpha_blend(qr7, qg7, qb7, qa, pr, pg, pb, pa, wr, wg, wb, wa, pParams);

							best_err = min(best_err, min(min(min(err4, err5), err6), err7));

							best_sel = select(best_err == err4, 4, best_sel);
							best_sel = select(best_err == err5, 5, best_sel);
							best_sel = select(best_err == err6, 6, best_sel);
							best_sel = select(best_err == err7, 7, best_sel);
						}
					}
					else
					{
						{
							float qr0 = weightedColors[0].m_c[0];
							float qg0 = weightedColors[0].m_c[1];
							float qb0 = weightedColors[0].m_c[2];
							float err0 = evaluate_error_orthogonal(qr0, qg0, qb0, pr, pg, pb, wr, wg, wb);

							float qr1 = weightedColors[1].m_c[0];
							float qg1 = weightedColors[1].m_c[1];
							float qb1 = weightedColors[1].m_c[2];
							float err1 = evaluate_error_orthogonal(qr1, qg1, qb1, pr, pg, pb, wr, wg, wb);

							float qr2 = weightedColors[2].m_c[0];
							float qg2 = weightedColors[2].m_c[1];
							float qb2 = weightedColors[2].m_c[2];
							float err2 = evaluate_error_orthogonal(qr2, qg2, qb2, pr, pg, pb, wr, wg, wb);

							float qr3 = weightedColors[3].m_c[0];
							float qg3 = weightedColors[3].m_c[1];
							float qb3 = weightedColors[3].m_c[2];
							float err3 = evaluate_error_orthogonal(qr3, qg3, qb3, pr, pg, pb, wr, wg, wb);

							best_err = min(min(min(err0, err1), err2), err3);
									
							best_sel = select(best_err == err1, 1, 0);
							best_sel = select(best_err == err2, 2, best_sel);
							best_sel = select(best_err == err3, 3, best_sel);
						}

						{
							float qr4 = weightedColors[4].m_c[0];
							float qg4 = weightedColors[4].m_c[1];
							float qb4 = weightedColors[4].m_c[2];
							float err4 = evaluate_error_orthogonal(qr4, qg4, qb4, pr, pg, pb, wr, wg, wb);

							float qr5 = weightedColors[5].m_c[0];
							float qg5 = weightedColors[5].m_c[1];
							float qb5 = weightedColors[5].m_c[2];
							float err5 = evaluate_error_orthogonal(qr5, qg5, qb5, pr, pg, pb, wr, wg, wb);

							float qr6 = weightedColors[6].m_c[0];
							float qg6 = weightedColors[6].m_c[1];
							float qb6 = weightedColors[6].m_c[2];
							float err6 = evaluate_error_orthogonal(qr6, qg6, qb6, pr, pg, pb, wr, wg, wb);

							float qr7 = weightedColors[7].m_c[0];
							float qg7 = weightedColors[7].m_c[1];
							float qb7 = weightedColors[7].m_c[2];
							float err7 = evaluate_error_orthogonal(qr7, qg7, qb7, pr, pg, pb, wr, wg, wb);

							best_err = min(best_err, min(min(min(err4, err5), err6), err7));

							best_sel = select(best_err == err4, 4, best_sel);
							best_sel = select(best_err == err5, 5, best_sel);
							best_sel = select(best_err == err6, 6, best_sel);
							best_sel = select(best_err == err7, 7, best_sel);
						}
					}
					total_err += best_err;

					pResults->m_pSelectors_temp[i] = best_sel;
				}
			}
			else // if (N == 4)
			{
				cfor (uniform uint32_t i = 0; i < num_pixels; i++)
				{
					float pr = (float)pPixels[i].m_c[0];
					float pg = (float)pPixels[i].m_c[1];
					float pb = (float)pPixels[i].m_c[2];

					float err0, err1, err2, err3;
					if (optimizeForColorTimesAlpha_SeparateAlpha)
					{
						float pa = (float)pPixels[i].m_c[3];
						float qa = pCompAlpha[i];

						float qr0 = weightedColors[0].m_c[0];
						float qg0 = weightedColors[0].m_c[1];
						float qb0 = weightedColors[0].m_c[2];
						err0 = evaluate_error_alpha_blend(qr0, qg0, qb0, qa, pr, pg, pb, pa, wr, wg, wb, wa, pParams);

						float qr1 = weightedColors[1].m_c[0];
						float qg1 = weightedColors[1].m_c[1];
						float qb1 = weightedColors[1].m_c[2];
						err1 = evaluate_error_alpha_blend(qr1, qg1, qb1, qa, pr, pg, pb, pa, wr, wg, wb, wa, pParams);

						float qr2 = weightedColors[2].m_c[0];
						float qg2 = weightedColors[2].m_c[1];
						float qb2 = weightedColors[2].m_c[2];
						err2 = evaluate_error_alpha_blend(qr2, qg2, qb2, qa, pr, pg, pb, pa, wr, wg, wb, wa, pParams);

						float qr3 = weightedColors[3].m_c[0];
						float qg3 = weightedColors[3].m_c[1];
						float qb3 = weightedColors[3].m_c[2];
						err3 = evaluate_error_alpha_blend(qr3, qg3, qb3, qa, pr, pg, pb, pa, wr, wg, wb, wa, pParams);
					}
					else
					{
						float qr0 = weightedColors[0].m_c[0];
						float qg0 = weightedColors[0].m_c[1];
						float qb0 = weightedColors[0].m_c[2];
						err0 = evaluate_error_orthogonal(qr0, qg0, qb0, pr, pg, pb, wr, wg, wb);

						float qr1 = weightedColors[1].m_c[0];
						float qg1 = weightedColors[1].m_c[1];
						float qb1 = weightedColors[1].m_c[2];
						err1 = evaluate_error_orthogonal(qr1, qg1, qb1, pr, pg, pb, wr, wg, wb);

						float qr2 = weightedColors[2].m_c[0];
						float qg2 = weightedColors[2].m_c[1];
						float qb2 = weightedColors[2].m_c[2];
						err2 = evaluate_error_orthogonal(qr2, qg2, qb2, pr, pg, pb, wr, wg, wb);

						float qr3 = weightedColors[3].m_c[0];
						float qg3 = weightedColors[3].m_c[1];
						float qb3 = weightedColors[3].m_c[2];
						err3 = evaluate_error_orthogonal(qr3, qg3, qb3, pr, pg, pb, wr, wg, wb);
					}

					float best_err = min(min(min(err0, err1), err2), err3);

					int best_sel = select(best_err == err1, 1, 0);
					best_sel = select(best_err == err2, 2, best_sel);
					best_sel = select(best_err == err3, 3, best_sel);
								
					total_err += best_err;

					pResults->m_pSelectors_temp[i] = best_sel;
				}
			}
		}
		else
		{
			// alpha
			if (N == 16)
			{
				float lr = actualMinColor.m_c[0];
				float lg = actualMinColor.m_c[1];
				float lb = actualMinColor.m_c[2];
				float la = actualMinColor.m_c[3];

				float dr = actualMaxColor.m_c[0] - lr;
				float dg = actualMaxColor.m_c[1] - lg;
				float db = actualMaxColor.m_c[2] - lb;
				float da = actualMaxColor.m_c[3] - la;
			
				const float f = N / (dr * dr + dg * dg + db * db + da * da);

				lr *= -dr;
				lg *= -dg;
				lb *= -db;
				la *= -da;

				cfor (uniform uint32_t i = 0; i < num_pixels; i++)
				{
					const varying color_quad_i *uniform pC = &pPixels[i];
					float r = pC->m_c[0];
					float g = pC->m_c[1];
					float b = pC->m_c[2];
					float a = pC->m_c[3];

					float best_sel = round(((r * dr + lr) + (g * dg + lg) + (b * db + lb) + (a * da + la)) * f);
					best_sel = clamp(best_sel, (float)1, (float)(N - 1));

					float best_sel0 = best_sel - 1;

					float err0, err1;
					if (pParams->m_optimize_for_color_times_alpha)
					{
						err0 = evaluate_error_alpha_blend(&weightedColors[(int)best_sel0], r, g, b, a, wr, wg, wb, wa, pParams);
						err1 = evaluate_error_alpha_blend(&weightedColors[(int)best_sel], r, g, b, a, wr, wg, wb, wa, pParams);
					}
					else
					{
						err0 = evaluate_error_orthogonal(&weightedColors[(int)best_sel0], r, g, b, a, wr, wg, wb, wa);
						err1 = evaluate_error_orthogonal(&weightedColors[(int)best_sel], r, g, b, a, wr, wg, wb, wa);
					}

					float min_err = min(err0, err1);
					total_err += min_err;
					pResults->m_pSelectors_temp[i] = (int)select(min_err == err0, best_sel0, best_sel);
				}
			}
			else if (N == 8)
			{
				cfor (uniform uint32_t i = 0; i < num_pixels; i++)
				{
					float pr = (float)pPixels[i].m_c[0];
					float pg = (float)pPixels[i].m_c[1];
					float pb = (float)pPixels[i].m_c[2];
					float pa = (float)pPixels[i].m_c[3];
				
					float best_err;
					int best_sel;

					if (pParams->m_optimize_for_color_times_alpha)
					{
						{
							float err0 = evaluate_error_alpha_blend(&weightedColors[0], pr, pg, pb, pa, wr, wg, wb, wa, pParams);
							float err1 = evaluate_error_alpha_blend(&weightedColors[1], pr, pg, pb, pa, wr, wg, wb, wa, pParams);
							float err2 = evaluate_error_alpha_blend(&weightedColors[2], pr, pg, pb, pa, wr, wg, wb, wa, pParams);
							float err3 = evaluate_error_alpha_blend(&weightedColors[3], pr, pg, pb, pa, wr, wg, wb, wa, pParams);

							best_err = min(min(min(err0, err1), err2), err3);
										
							best_sel = select(best_err == err1, 1, 0);
							best_sel = select(best_err == err2, 2, best_sel);
							best_sel = select(best_err == err3, 3, best_sel);
						}
						{
							float err4 = evaluate_error_alpha_blend(&weightedColors[4], pr, pg, pb, pa, wr, wg, wb, wa, pParams);
							float err5 = evaluate_error_alpha_blend(&weightedColors[5], pr, pg, pb, pa, wr, wg, wb, wa, pParams);
							float err6 = evaluate_error_alpha_blend(&weightedColors[6], pr, pg, pb, pa, wr, wg, wb, wa, pParams);
							float err7 = evaluate_error_alpha_blend(&weightedColors[7], pr, pg, pb, pa, wr, wg, wb, wa, pParams);

							best_err = min(best_err, min(min(min(err4, err5), err6), err7));

							best_sel = select(best_err == err4, 4, best_sel);
							best_sel = select(best_err == err5, 5, best_sel);
							best_sel = select(best_err == err6, 6, best_sel);
							best_sel = select(best_err == err7, 7, best_sel);
						}
					}
					else
					{
						{
							float err0 = evaluate_error_orthogonal(&weightedColors[0], pr, pg, pb, pa, wr, wg, wb, wa);
							float err1 = evaluate_error_orthogonal(&weightedColors[1], pr, pg, pb, pa, wr, wg, wb, wa);
							float err2 = evaluate_error_orthogonal(&weightedColors[2], pr, pg, pb, pa, wr, wg, wb, wa);
							float err3 = evaluate_error_orthogonal(&weightedColors[3], pr, pg, pb, pa, wr, wg, wb, wa);

							best_err = min(min(min(err0, err1), err2), err3);
										
							best_sel = select(best_err == err1, 1, 0);
							best_sel = select(best_err == err2, 2, best_sel);
							best_sel = select(best_err == err3, 3, best_sel);
						}
						{
							float err4 = evaluate_error_orthogonal(&weightedColors[4], pr, pg, pb, pa, wr, wg, wb, wa);
							float err5 = evaluate_error_orthogonal(&weightedColors[5], pr, pg, pb, pa, wr, wg, wb, wa);
							float err6 = evaluate_error_orthogonal(&weightedColors[6], pr, pg, pb, pa, wr, wg, wb, wa);
							float err7 = evaluate_error_orthogonal(&weightedColors[7], pr, pg, pb, pa, wr, wg, wb, wa);

							best_err = min(best_err, min(min(min(err4, err5), err6), err7));

							best_sel = select(best_err == err4, 4, best_sel);
							best_sel = select(best_err == err5, 5, best_sel);
							best_sel = select(best_err == err6, 6, best_sel);
							best_sel = select(best_err == err7, 7, best_sel);
						}
					}
				
					total_err += best_err;

					pResults->m_pSelectors_temp[i] = best_sel;
				}
			}
			else // if (N == 4)
			{
				cfor (uniform uint32_t i = 0; i < num_pixels; i++)
				{
					float pr = (float)pPixels[i].m_c[0];
					float pg = (float)pPixels[i].m_c[1];
					float pb = (float)pPixels[i].m_c[2];
					float pa = (float)pPixels[i].m_c[3];
				
					float best_err;
					int best_sel;

					if (pParams->m_optimize_for_color_times_alpha)
					{
						float err0 = evaluate_error_alpha_blend(&weightedColors[0], pr, pg, pb, pa, wr, wg, wb, wa, pParams);
						float err1 = evaluate_error_alpha_blend(&weightedColors[1], pr, pg, pb, pa, wr, wg, wb, wa, pParams);
						float err2 = evaluate_error_alpha_blend(&weightedColors[2], pr, pg, pb, pa, wr, wg, wb, wa, pParams);
						float err3 = evaluate_error_alpha_blend(&weightedColors[3], pr, pg, pb, pa, wr, wg, wb, wa, pParams);

						best_err = min(min(min(err0, err1), err2), err3);
										
						best_sel = select(best_err == err1, 1, 0);
						best_sel = select(best_err == err2, 2, best_sel);
						best_sel = select(best_err == err3, 3, best_sel);
					}
					else
					{
						float err0 = evaluate_error_orthogonal(&weightedColors[0], pr, pg, pb, pa, wr, wg, wb, wa);
						float err1 = evaluate_error_orthogonal(&weightedColors[1], pr, pg, pb, pa, wr, wg, wb, wa);
						float err2 = evaluate_error_orthogonal(&weightedColors[2], pr, pg, pb, pa, wr, wg, wb, wa);
						float err3 = evaluate_error_orthogonal(&weightedColors[3], pr, pg, pb, pa, wr, wg, wb, wa);

						best_err = min(min(min(err0, err1), err2), err3);
										
						best_sel = select(best_err == err1, 1, 0);
						best_sel = select(best_err == err2, 2, best_sel);
						best_sel = select(best_err == err3, 3, best_sel);
					}
				
					total_err += best_err;

					pResults->m_pSelectors_temp[i] = best_sel;
				}
			}
		}
	}
	else
	{
		wg *= pr_weight;
		wb *= pb_weight;

		float weightedColorsY[16], weightedColorsCr[16], weightedColorsCb[16];
		
		if (!pParams->m_optimize_for_color_times_alpha || (!pParams->m_has_alpha && pCompAlpha == NULL))
		{
			for (uniform uint32_t i = 0; i < N; i++)
			{
				const float r = weightedColors[i].m_c[0];
				const float g = weightedColors[i].m_c[1];
				const float b = weightedColors[i].m_c[2];

				const float y = r * k_Y_R + g * k_Y_G + b * k_Y_B;
									
				weightedColorsY[i] = y;
				weightedColorsCr[i] = r - y;
				weightedColorsCb[i] = b - y;
			}

			if (pParams->m_has_alpha)
			{
				for (uniform uint32_t i = 0; i < num_pixels; i++)
				{
					const float r = pPixels[i].m_c[0];
					const float g = pPixels[i].m_c[1];
					const float b = pPixels[i].m_c[2];
					const float a = pPixels[i].m_c[3];

					const float y = r * k_Y_R + g * k_Y_G + b * k_Y_B;
					const float cr = r - y;
					const float cb = b - y;

					float best_err = FLT_MAX;
					int32_t best_sel;
									
					for (uniform uint32_t j = 0; j < N; j++)
					{
						const float dy = y - weightedColorsY[j];
						const float dcr = cr - weightedColorsCr[j];
						const float dcb = cb - weightedColorsCb[j];
						const float da = a - weightedColors[j].m_c[3];

						const float err = (wr * dy * dy) + (wg * dcr * dcr) + (wb * dcb * dcb) + (wa * da * da);
						if (err < best_err)
						{
							best_err = err;
							best_sel = j;
						}
					}
					
					total_err += best_err;

					pResults->m_pSelectors_temp[i] = best_sel;
				}
			}
			else
			{
				for (uniform uint32_t i = 0; i < num_pixels; i++)
				{
					const float r = pPixels[i].m_c[0];
					const float g = pPixels[i].m_c[1];
					const float b = pPixels[i].m_c[2];

					const float y = r * k_Y_R + g * k_Y_G + b * k_Y_B;
					const float cr = r - y;
					const float cb = b - y;

					float best_err = FLT_MAX;
					int32_t best_sel;
								
					for (uniform uint32_t j = 0; j < N; j++)
					{
						const float dy = y - weightedColorsY[j];
						const float dcr = cr - weightedColorsCr[j];
						const float dcb = cb - weightedColorsCb[j];

						const float err = (wr * dy * dy) + (wg * dcr * dcr) + (wb * dcb * dcb);
						if (err < best_err)
						{
							best_err = err;
							best_sel = j;
						}
					}
				
					total_err += best_err;

					pResults->m_pSelectors_temp[i] = best_sel;
				}
			}
		}
		else if (pParams->m_has_alpha)
		{
			// As an optimization, the loop to find the best palette entry just minimizes |D|^2 + da^2. We just need to pick the
			// palette entry that minimizes peak error; this should track well with that, and is much cheaper to calculate than
			// the actual error. We do the expensive check for the actual error at the end so that we can compare candidate blocks
			// fairly.
			for (uniform uint32_t i = 0; i < N; i++)
			{
				float r = weightedColors[i].m_c[0];
				float g = weightedColors[i].m_c[1];
				float b = weightedColors[i].m_c[2];
				float a = weightedColors[i].m_c[3];

				float y = r * k_Y_R + g * k_Y_G + b * k_Y_B;
									
				weightedColorsY[i] = y * a;
				weightedColorsCr[i] = (r - y) * a;
				weightedColorsCb[i] = (b - y) * a;
			}

			for (uniform uint32_t i = 0; i < num_pixels; i++)
			{
				float r = pPixels[i].m_c[0];
				float g = pPixels[i].m_c[1];
				float b = pPixels[i].m_c[2];
				float a = pPixels[i].m_c[3];

				float y = r * k_Y_R + g * k_Y_G + b * k_Y_B;
				float cr = r - y;
				float cb = b - y;
				float y_a = y * a;
				float cr_a = cr * a;
				float cb_a = cb * a;

				// The worst case dy, dcr, dcb for a candidate is 255 * 255 (~2^16). This gets squared (~2^32). There are 16 pixels (~2^36).
				// So, we need to initialize this to a really big number; 10 billion is approximately 2^33 which is not enough.
				float best_err = FLT_MAX;
				uniform const bool doPrints = false;//pParams->m_debug_spam;
				int32_t best_sel;
				float best_da;
				vec3F D;
				if (doPrints)
				{
					print("consts:\n");
					print("  wr % wg % wb % wa %\n", wr, wg, wb, wa);
					print("  has_alpha %\n", pParams->m_has_alpha);
					print("pixel:\n");
					print("  R %\n", r);
					print("  G %\n", g);
					print("  B %\n", b);
					print("  A %\n", a);
					print("  Y %\n", y);
					print("  Cr %\n", cr);
					print("  Cb %\n", cb);
					print("  Y_a %\n", y_a);
					print("  Cr_a %\n", cr_a);
					print("  Cb_a %\n", cb_a);
				}

				for (uniform uint32_t j = 0; j < N; j++)
				{
					float dy = y_a - weightedColorsY[j];
					float dcr = cr_a - weightedColorsCr[j];
					float dcb = cb_a - weightedColorsCb[j];
					float da = a - weightedColors[j].m_c[3];

					float err = (wr * dy * dy) + (wg * dcr * dcr) + (wb * dcb * dcb) + (wa * da * da);
					if (doPrints)
					{
						print("index %:\n", j);
						print("  R %\n", weightedColors[j].m_c[0]);
						print("  G %\n", weightedColors[j].m_c[1]);
						print("  B %\n", weightedColors[j].m_c[2]);
						print("  A %\n", weightedColors[j].m_c[3]);
						print("  Y %\n", weightedColorsY[j]);
						print("  Cr %\n", weightedColorsCr[j]);
						print("  Cb %\n", weightedColorsCb[j]);
						print("  dy %\n", dy);
						print("  dcr %\n", dcr);
						print("  dcb %\n", dcb);
						print("  da %\n", da);
						print("  err %\n", err);
					}
					if (err < best_err)
					{
						best_err = err;
						best_sel = j;
						D.m_c[0] = dy;
						D.m_c[1] = dcr;
						D.m_c[2] = dcb;
						best_da = da;
					}
				}

				// now calculate actual error
				vec3F mid = vec3F_mul(&D, &pParams->m_sqrt_weights);
				mid.m_c[0] += best_da * pParams->m_dycrcb_mid;

				vec3F delta_r = vec3F_mul(&pParams->m_dycrcb_r, copysign(best_da, vec3F_dot(&pParams->m_dycrcb_r, &mid)));
				vec3F delta_g = vec3F_mul(&pParams->m_dycrcb_g, copysign(best_da, vec3F_dot(&pParams->m_dycrcb_g, &mid)));
				vec3F delta_b = vec3F_mul(&pParams->m_dycrcb_b, copysign(best_da, vec3F_dot(&pParams->m_dycrcb_b, &mid)));

				vec3F corner = mid;
				vec3F_accum(&corner, &delta_r);
				vec3F_accum(&corner, &delta_g);
				vec3F_accum(&corner, &delta_b);

				const float actual_err = vec3F_dot(&corner, &corner) * (1.0f / (255.0f * 255.0f));
				if (doPrints)
					print("actual err %\n", actual_err);
				total_err += actual_err;

				pResults->m_pSelectors_temp[i] = best_sel;
				assert(best_sel < pParams->m_num_selector_weights);
			}
		}
		else
		{
			assert(pCompAlpha);
			assert(optimizeForColorTimesAlpha_SeparateAlpha);

			// As an optimization, the loop to find the best palette entry just minimizes |D|^2 + da^2. We just need to pick the
			// palette entry that minimizes peak error; this should track well with that, and is much cheaper to calculate than
			// the actual error. We do the expensive check for the actual error at the end so that we can compare candidate blocks
			// fairly.
			if (pParams->m_rotation == 0)
			{
				for (uniform uint32_t i = 0; i < N; i++)
				{
					float r = weightedColors[i].m_c[0];
					float g = weightedColors[i].m_c[1];
					float b = weightedColors[i].m_c[2];

					float y = r * k_Y_R + g * k_Y_G + b * k_Y_B;
									
					weightedColorsY[i] = y;
					weightedColorsCr[i] = r - y;
					weightedColorsCb[i] = b - y;
				}

				for (uniform uint32_t i = 0; i < num_pixels; i++)
				{
					float r = pPixels[i].m_c[0];
					float g = pPixels[i].m_c[1];
					float b = pPixels[i].m_c[2];
					float a = pPixels[i].m_c[3];

					float y = r * k_Y_R + g * k_Y_G + b * k_Y_B;
					float cr = r - y;
					float cb = b - y;
					float y_a = y * a;
					float cr_a = cr * a;
					float cb_a = cb * a;

					// The worst case dy, dcr, dcb for a candidate is 255 * 255 (~2^16). This gets squared (~2^32). There are 16 pixels (~2^36).
					// So, we need to initialize this to a really big number; 10 billion is approximately 2^33 which is not enough.
					float best_err = FLT_MAX;
					int32_t best_sel;
					float best_da;
					vec3F D;

					const float sep_a = pCompAlpha[i];
					for (uniform uint32_t j = 0; j < N; j++)
					{
						float dy = y_a - weightedColorsY[j] * sep_a;
						float dcr = cr_a - weightedColorsCr[j] * sep_a;
						float dcb = cb_a - weightedColorsCb[j] * sep_a;
						float da = a - sep_a;

						float err = (wr * dy * dy) + (wg * dcr * dcr) + (wb * dcb * dcb) + (wa * da * da);
						if (err < best_err)
						{
							best_err = err;
							best_sel = j;
							D.m_c[0] = dy;
							D.m_c[1] = dcr;
							D.m_c[2] = dcb;
							best_da = da;
						}
					}

					// now calculate actual error
					vec3F mid = vec3F_mul(&D, &pParams->m_sqrt_weights);
					mid.m_c[0] += best_da * pParams->m_dycrcb_mid;

					vec3F delta_r = vec3F_mul(&pParams->m_dycrcb_r, copysign(best_da, vec3F_dot(&pParams->m_dycrcb_r, &mid)));
					vec3F delta_g = vec3F_mul(&pParams->m_dycrcb_g, copysign(best_da, vec3F_dot(&pParams->m_dycrcb_g, &mid)));
					vec3F delta_b = vec3F_mul(&pParams->m_dycrcb_b, copysign(best_da, vec3F_dot(&pParams->m_dycrcb_b, &mid)));

					vec3F corner = mid;
					vec3F_accum(&corner, &delta_r);
					vec3F_accum(&corner, &delta_g);
					vec3F_accum(&corner, &delta_b);

					total_err += vec3F_dot(&corner, &corner) * (1.0f / (255.0f * 255.0f));

					pResults->m_pSelectors_temp[i] = best_sel;
					assert(best_sel < pParams->m_num_selector_weights);
				}
			}
			else
			{
				uniform const uint32_t chan_r = (pParams->m_rotation == 1) ? 3 : 0;
				uniform const uint32_t chan_g = (pParams->m_rotation == 2) ? 3 : 1;
				uniform const uint32_t chan_b = (pParams->m_rotation == 3) ? 3 : 2;
				uniform const uint32_t chan_a = pParams->m_rotation - 1;
				for (uniform uint32_t i = 0; i < num_pixels; i++)
				{
					const float r = pPixels[i].m_c[chan_r];
					const float g = pPixels[i].m_c[chan_g];
					const float b = pPixels[i].m_c[chan_b];
					const float a = pPixels[i].m_c[chan_a];

					const float y = r * k_Y_R + g * k_Y_G + b * k_Y_B;
					const float cr = r - y;
					const float cb = b - y;
					const float y_a = y * a;
					const float cr_a = cr * a;
					const float cb_a = cb * a;

					// The worst case dy, dcr, dcb for a candidate is 255 * 255 (~2^16). This gets squared (~2^32). There are 16 pixels (~2^36).
					// So, we need to initialize this to a really big number; 10 billion is approximately 2^33 which is not enough.
					float best_err = FLT_MAX;
					int32_t best_sel;
					float best_da;
					vec3F D;

					const float sep_a = pCompAlpha[i];
					for (uniform uint32_t j = 0; j < N; j++)
					{
						const float pal_r = (pParams->m_rotation == 1) ? pCompAlpha[i] : weightedColors[j].m_c[0];
						const float pal_g = (pParams->m_rotation == 2) ? pCompAlpha[i] : weightedColors[j].m_c[1];
						const float pal_b = (pParams->m_rotation == 3) ? pCompAlpha[i] : weightedColors[j].m_c[2];
						const float pal_a = weightedColors[j].m_c[pParams->m_rotation - 1];

						const float pal_y = pal_r * k_Y_R + pal_g * k_Y_G + pal_b * k_Y_B;
						const float pal_cr = pal_r - pal_y;
						const float pal_cb = pal_b - pal_y;

						const float dy = y_a - pal_y * pal_a;
						const float dcr = cr_a - pal_cr * pal_a;
						const float dcb = cb_a - pal_cb * pal_a;
						const float da = a - pal_a;

						const float err = (wr * dy * dy) + (wg * dcr * dcr) + (wb * dcb * dcb) + (wa * da * da);
						if (err < best_err)
						{
							best_err = err;
							best_sel = j;
							D.m_c[0] = dy;
							D.m_c[1] = dcr;
							D.m_c[2] = dcb;
							best_da = da;
						}
					}

					// now calculate actual error
					vec3F mid = vec3F_mul(&D, &pParams->m_sqrt_weights);
					mid.m_c[0] += best_da * pParams->m_dycrcb_mid;

					vec3F delta_r = vec3F_mul(&pParams->m_dycrcb_r, copysign(best_da, vec3F_dot(&pParams->m_dycrcb_r, &mid)));
					vec3F delta_g = vec3F_mul(&pParams->m_dycrcb_g, copysign(best_da, vec3F_dot(&pParams->m_dycrcb_g, &mid)));
					vec3F delta_b = vec3F_mul(&pParams->m_dycrcb_b, copysign(best_da, vec3F_dot(&pParams->m_dycrcb_b, &mid)));

					vec3F corner = mid;
					vec3F_accum(&corner, &delta_r);
					vec3F_accum(&corner, &delta_g);
					vec3F_accum(&corner, &delta_b);

					total_err += vec3F_dot(&corner, &corner) * (1.0f / (255.0f * 255.0f));

					pResults->m_pSelectors_temp[i] = best_sel;
					assert(best_sel < pParams->m_num_selector_weights);
				}
			}
		}
	}

	if (total_err < pResults->m_best_overall_err)
	{
		pResults->m_best_overall_err = total_err;

		pResults->m_low_endpoint = *pLow;
		pResults->m_high_endpoint = *pHigh;

		pResults->m_pbits[0] = pbits[0];
		pResults->m_pbits[1] = pbits[1];

		for (uniform uint32_t i = 0; i < num_pixels; i++)
		{
			pResults->m_pSelectors[i] = pResults->m_pSelectors_temp[i];
			assert(pResults->m_pSelectors[i] < pParams->m_num_selector_weights);
		}
	}
				
	return total_err;
}

static void fixDegenerateEndpoints(uniform uint32_t mode, varying color_quad_i *uniform pTrialMinColor, varying color_quad_i *uniform pTrialMaxColor, const varying vec4F *uniform pXl, const varying vec4F *uniform pXh, uniform uint32_t iscale)
{
	if ((mode == 1) || (mode == 4)) // also mode 2
	{
		// fix degenerate case where the input collapses to a single colorspace voxel, and we lose all freedom (test with grayscale ramps)
		for (uniform uint32_t i = 0; i < 3; i++)
		{
			if (pTrialMinColor->m_c[i] == pTrialMaxColor->m_c[i])
			{
				if (abs(pXl->m_c[i] - pXh->m_c[i]) > 0.0f)
				{
					if (pTrialMinColor->m_c[i] > (iscale >> 1))
					{
						if (pTrialMinColor->m_c[i] > 0)
							pTrialMinColor->m_c[i]--;
						else
							if (pTrialMaxColor->m_c[i] < iscale)
								pTrialMaxColor->m_c[i]++;
					}
					else
					{
						if (pTrialMaxColor->m_c[i] < iscale)
							pTrialMaxColor->m_c[i]++;
						else if (pTrialMinColor->m_c[i] > 0)
							pTrialMinColor->m_c[i]--;
					}

					if (mode == 4)
					{
						if (pTrialMinColor->m_c[i] > (iscale >> 1))
						{
							if (pTrialMaxColor->m_c[i] < iscale)
								pTrialMaxColor->m_c[i]++;
							else if (pTrialMinColor->m_c[i] > 0)
								pTrialMinColor->m_c[i]--;
						}
						else
						{
							if (pTrialMinColor->m_c[i] > 0)
								pTrialMinColor->m_c[i]--;
							else if (pTrialMaxColor->m_c[i] < iscale)
								pTrialMaxColor->m_c[i]++;
						}
					}
				}
			}
		}
	}
}

static float find_optimal_solution(uniform uint32_t mode, varying vec4F *uniform pXl, varying vec4F *uniform pXh, const uniform color_cell_compressor_params *uniform pParams, varying color_cell_compressor_results *uniform pResults, 
	uniform bool pbit_search, uint32_t num_pixels, const varying color_quad_i *uniform pPixels, const varying int32_t *uniform pCompAlpha)
{
	vec4F xl = *pXl;
	vec4F xh = *pXh;

	vec4F_saturate_in_place(&xl);
	vec4F_saturate_in_place(&xh);
		
	if (pParams->m_has_pbits)
	{
		// Notes: The pbit controls which quantization intervals are selected.
		// total_levels=2^(comp_bits+1), where comp_bits=4 for mode 0, etc.
		// pbit 0: v=(b*2)/(total_levels-1), pbit 1: v=(b*2+1)/(total_levels-1) where b is the component bin from [0,total_levels/2-1] and v is the [0,1] component value
		// rearranging you get for pbit 0: b=round((v*(total_levels-1)-0)/2)
		// rearranging you get for pbit 1: b=round((v*(total_levels-1)-1)/2)
		// let N = total_levels - 1: b = (int)round((v * N - p) * 0.5)
		// v is clamped to be in [0,1], so "v * N - p" is in the range [-p,N-p]. When halved, this is [-p/2, (N-p)/2].
		// The valid range for b is [0, (N-1)/2], so b can be too small when p is 1 or too large when p is 0, so we just need to clamp.
		//   b = (int)round(clamp((v * N - p) * 0.5, 0, (N - 1) * 0.5)
		// We can also precalculate constants:
		//   b = (int)round(clamp(v * (N * 0.5) - (p * 0.5), 0, (N * 0.5 - 0.5)))

		const uniform int total_comp_bits = pParams->m_comp_bits + 1;
		const uniform int iscalep = (1 << total_comp_bits) - 1;
		const uniform float scalep = (float)iscalep;
		const uniform float half_N = scalep * 0.5f;
		const uniform float upper = half_N - 0.5f;

		const uniform int32_t totalComps = pParams->m_has_alpha ? 4 : 3;

		const uniform bool force_pbits_to_1 = (mode == 6 && !pParams->m_has_alpha);	// If we have an opaque block in mode 6, we need pbits 1 to keep alpha 255 instead of 254

		if (pbit_search)
		{
			// compensated rounding+pbit search
			color_quad_i lo[2], hi[2];
								
			for (uniform int p = 0; p < 2; p++)
			{
				const uniform float half_p = p * 0.5f;
				for (uniform uint32_t c = 0; c < 4; c++)
				{
					lo[p].m_c[c] = (int)round(clamp(xl.m_c[c] * half_N - half_p, 0.0f, upper));
					hi[p].m_c[c] = (int)round(clamp(xh.m_c[c] * half_N - half_p, 0.0f, upper));
				}
			}

			fixDegenerateEndpoints(mode, &lo[0], &hi[0], &xl, &xh, iscalep >> 1);
			fixDegenerateEndpoints(mode, &lo[1], &hi[1], &xl, &xh, iscalep >> 1);

			uint32_t pbits[2];

			// shared pbits
			pbits[0] = 1; pbits[1] = 1;
			evaluate_solution(&lo[1], &hi[1], pbits, pParams, pResults, num_pixels, pPixels, pCompAlpha);

			if (!force_pbits_to_1)
			{
				pbits[0] = 0; pbits[1] = 0;
				evaluate_solution(&lo[0], &hi[0], pbits, pParams, pResults, num_pixels, pPixels, pCompAlpha);

				if (!pParams->m_endpoints_share_pbit)
				{
					// different pbits
					pbits[0] = 0; pbits[1] = 1;
					evaluate_solution(&lo[0], &hi[1], pbits, pParams, pResults, num_pixels, pPixels, pCompAlpha);

					pbits[0] = 1; pbits[1] = 0;
					evaluate_solution(&lo[1], &hi[0], pbits, pParams, pResults, num_pixels, pPixels, pCompAlpha);
				}
			}
		}
		else
		{
			// compensated rounding
			uint32_t best_pbits[2];
			color_quad_i bestMinColor, bestMaxColor;

			const uniform int left_shift = 8 - total_comp_bits;
			const uniform int right_shift = total_comp_bits * 2 - 8;
			const uniform int min_p = force_pbits_to_1 ? 1 : 0;
			if (!pParams->m_endpoints_share_pbit)
			{
				float best_err0 = FLT_MAX;
				float best_err1 = FLT_MAX;
								
				for (uniform int p = min_p; p < 2; p++)
				{
					color_quad_i xMinColor, xMaxColor;

					const uniform float half_p = p * 0.5f;
					float err0 = 0.0f;
					float err1 = 0.0f;
					for (uniform uint32_t c = 0; c < totalComps; c++)
					{
						const int lo = (int)round(clamp(xl.m_c[c] * half_N - half_p, 0.0f, upper));
						const int hi = (int)round(clamp(xh.m_c[c] * half_N - half_p, 0.0f, upper));
						xMinColor.m_c[c] = lo;
						xMaxColor.m_c[c] = hi;

						const int lo_with_p = lo * 2 + p;
						const int hi_with_p = hi * 2 + p;

						const int lo_decoded = (lo_with_p << left_shift) | (lo_with_p >> right_shift);
						const int hi_decoded = (hi_with_p << left_shift) | (hi_with_p >> right_shift);
						assert(lo_decoded <= 255);
						assert(hi_decoded <= 255);

						err0 += square(lo_decoded - xl.m_c[c] * 255.0f);
						err1 += square(hi_decoded - xh.m_c[c] * 255.0f);
					}
					if (!pParams->m_has_alpha)
					{
						xMinColor.m_c[3] = (int)round(clamp(xl.m_c[3] * half_N - half_p, 0.0f, upper));
						xMaxColor.m_c[3] = (int)round(clamp(xh.m_c[3] * half_N - half_p, 0.0f, upper));
					}

					if (err0 < best_err0)
					{
						best_err0 = err0;
						best_pbits[0] = p;
						bestMinColor = xMinColor;
					}

					if (err1 < best_err1)
					{
						best_err1 = err1;
						best_pbits[1] = p;
						bestMaxColor = xMaxColor;
					}
				}
			}
			else
			{
				// Endpoints share pbits
				float best_err = FLT_MAX;
								
				for (uniform int p = min_p; p < 2; p++)
				{
					color_quad_i xMinColor, xMaxColor;

					const float half_p = p * 0.5f;
					float err = 0.0f;
					for (uniform uint32_t c = 0; c < totalComps; c++)
					{
						const int lo = (int)round(clamp(xl.m_c[c] * half_N - half_p, 0.0f, upper));
						const int hi = (int)round(clamp(xh.m_c[c] * half_N - half_p, 0.0f, upper));
						xMinColor.m_c[c] = lo;
						xMaxColor.m_c[c] = hi;

						const int lo_with_p = lo * 2 + p;
						const int hi_with_p = hi * 2 + p;

						const int lo_decoded = (lo_with_p << left_shift) | (lo_with_p >> right_shift);
						const int hi_decoded = (hi_with_p << left_shift) | (hi_with_p >> right_shift);
						assert(lo_decoded <= 255);
						assert(hi_decoded <= 255);

						err += square(lo_decoded - xl.m_c[c] * 255.0f);
						err += square(hi_decoded - xh.m_c[c] * 255.0f);
					}

					if (err < best_err)
					{
						best_err = err;
						best_pbits[0] = p;
						best_pbits[1] = p;
						bestMinColor = xMinColor;
						bestMaxColor = xMaxColor;
					}
				}
			}

			fixDegenerateEndpoints(mode, &bestMinColor, &bestMaxColor, &xl, &xh, iscalep >> 1);

			if ((pResults->m_best_overall_err == FLT_MAX) || color_quad_i_notequals(&bestMinColor, &pResults->m_low_endpoint) || color_quad_i_notequals(&bestMaxColor, &pResults->m_high_endpoint) || (best_pbits[0] != pResults->m_pbits[0]) || (best_pbits[1] != pResults->m_pbits[1]))
			{
				evaluate_solution(&bestMinColor, &bestMaxColor, best_pbits, pParams, pResults, num_pixels, pPixels, pCompAlpha);
			}
		}
	}
	else
	{
		const uniform int iscale = (1 << pParams->m_comp_bits) - 1;
		const uniform float scale = (float)iscale;

		color_quad_i trialMinColor, trialMaxColor;
		color_quad_i_set_clamped(&trialMinColor, (int)(xl.m_c[0] * scale + .5f), (int)(xl.m_c[1] * scale + .5f), (int)(xl.m_c[2] * scale + .5f), (int)(xl.m_c[3] * scale + .5f));
		color_quad_i_set_clamped(&trialMaxColor, (int)(xh.m_c[0] * scale + .5f), (int)(xh.m_c[1] * scale + .5f), (int)(xh.m_c[2] * scale + .5f), (int)(xh.m_c[3] * scale + .5f));

		fixDegenerateEndpoints(mode, &trialMinColor, &trialMaxColor, &xl, &xh, iscale);

		if ((pResults->m_best_overall_err == FLT_MAX) || color_quad_i_notequals(&trialMinColor, &pResults->m_low_endpoint) || color_quad_i_notequals(&trialMaxColor, &pResults->m_high_endpoint))
		{
			uint32_t pbits[2];
			pbits[0] = 0;
			pbits[1] = 0;

			evaluate_solution(&trialMinColor, &trialMaxColor, pbits, pParams, pResults, num_pixels, pPixels, pCompAlpha);
		}
	}

	return pResults->m_best_overall_err;
}

// Note: In mode 6, m_has_alpha will only be true for transparent blocks.
static float color_cell_compression(uniform uint32_t mode, const uniform color_cell_compressor_params *uniform pParams, varying color_cell_compressor_results *uniform pResults, 
	const uniform bc7e_compress_block_params *uniform pComp_params, uint32_t num_pixels, const varying color_quad_i *uniform pPixels, const varying int32_t *uniform pCompAlpha, uniform bool refinement)
{
	pResults->m_best_overall_err = FLT_MAX;

	if ((mode != 6) && (mode != 7))
		assert(!pParams->m_has_alpha);

	if ((mode != 4) && (mode != 5))
		assert(pCompAlpha == NULL);

	if ((mode <= 2) || (mode == 4) || (mode >= 6))
	{
		const uint32_t cr = pPixels[0].m_c[0];
		const uint32_t cg = pPixels[0].m_c[1];
		const uint32_t cb = pPixels[0].m_c[2];
		const uint32_t ca = pPixels[0].m_c[3];

		bool allSame = true;
		for (uniform uint32_t i = 1; i < num_pixels; i++)
		{
			if ((cr != pPixels[i].m_c[0]) || (cg != pPixels[i].m_c[1]) || (cb != pPixels[i].m_c[2]) || (ca != pPixels[i].m_c[3]))
			{
				allSame = false;
				break;
			}
		}

		cif (allSame)
		{
			if (mode == 0)
				return pack_mode0_to_one_color(pParams, pResults, cr, cg, cb, pResults->m_pSelectors, num_pixels, pPixels);
			if (mode == 1)
				return pack_mode1_to_one_color(pParams, pResults, cr, cg, cb, pResults->m_pSelectors, num_pixels, pPixels);
			else if (mode == 6)
				return pack_mode6_to_one_color(pParams, pResults, cr, cg, cb, ca, pResults->m_pSelectors, num_pixels, pPixels);
			else if (mode == 7)
				return pack_mode7_to_one_color(pParams, pResults, cr, cg, cb, ca, pResults->m_pSelectors, num_pixels, pPixels);
			else
				return pack_mode24_to_one_color(pParams, pResults, cr, cg, cb, pResults->m_pSelectors, num_pixels, pPixels, pCompAlpha);
		}
	}

	vec4F meanColor, axis;
	vec4F_set_scalar(&meanColor, 0.0f);

	for (uniform uint32_t i = 0; i < num_pixels; i++)
	{
		vec4F color = vec4F_from_color(&pPixels[i]);
		meanColor = vec4F_add(&meanColor, &color);
	}
				
	vec4F meanColorScaled = vec4F_mul(&meanColor, 1.0f / (float)((int)num_pixels));

	meanColor = vec4F_mul(&meanColor, 1.0f / (float)((int)num_pixels * 255.0f));
	vec4F_saturate_in_place(&meanColor);

	if (pParams->m_has_alpha)
	{
		vec4F v;
		vec4F_set_scalar(&v, 0.0f);
		cfor (uniform uint32_t i = 0; i < num_pixels; i++)
		{
			vec4F color = vec4F_from_color(&pPixels[i]);
			color = vec4F_sub(&color, &meanColorScaled);

			vec4F a = vec4F_mul(&color, color.m_c[0]);
			vec4F b = vec4F_mul(&color, color.m_c[1]);
			vec4F c = vec4F_mul(&color, color.m_c[2]);
			vec4F d = vec4F_mul(&color, color.m_c[3]);

			vec4F n = i ? v : color;
			vec4F_normalize_in_place(&n);

			v.m_c[0] += vec4F_dot(&a, &n);
			v.m_c[1] += vec4F_dot(&b, &n);
			v.m_c[2] += vec4F_dot(&c, &n);
			v.m_c[3] += vec4F_dot(&d, &n);
		}
		axis = v;
		vec4F_normalize_in_place(&axis);
	}
	else
	{
		float cov[6];
		cov[0] = 0; cov[1] = 0; cov[2] = 0; 
		cov[3] = 0; cov[4] = 0;	cov[5] = 0;

		cfor (uniform uint32_t i = 0; i < num_pixels; i++)
		{
			const varying color_quad_i *varying pV = &pPixels[i];

			float r = pV->m_c[0] - meanColorScaled.m_c[0];
			float g = pV->m_c[1] - meanColorScaled.m_c[1];
			float b = pV->m_c[2] - meanColorScaled.m_c[2];
				
			cov[0] += r*r;
			cov[1] += r*g;
			cov[2] += r*b;
			cov[3] += g*g;
			cov[4] += g*b;
			cov[5] += b*b;
		}

		float vfr, vfg, vfb;
		//vfr = hi[0] - lo[0];
		//vfg = hi[1] - lo[1];
		//vfb = hi[2] - lo[2];
		// This is more stable.
		vfr = .9f;
		vfg = 1.0f;
		vfb = .7f;

		for (uniform uint32_t iter = 0; iter < 3; iter++)
		{
			float r = vfr*cov[0] + vfg*cov[1] + vfb*cov[2];
			float g = vfr*cov[1] + vfg*cov[3] + vfb*cov[4];
			float b = vfr*cov[2] + vfg*cov[4] + vfb*cov[5];

			float m = maximumf(maximumf(abs(r), abs(g)), abs(b));
			if (m > 1e-10f)
			{
				m = 1.0f / m;
				r *= m;
				g *= m;
				b *= m;
			}

			//float delta = square(vfr - r) + square(vfg - g) + square(vfb - b);

			vfr = r;
			vfg = g;
			vfb = b;

			//if ((iter > 1) && (delta < 1e-8f))
			//	break;
		}

		float len = vfr*vfr + vfg*vfg + vfb*vfb;

		if (len < 1e-10f)
			vec4F_set_scalar(&axis, 0.0f);
		else
		{
			len = 1.0f / sqrt(len);
			vfr *= len;
			vfg *= len;
			vfb *= len;
			vec4F_set(&axis, vfr, vfg, vfb, 0);
		}
	}

	cif (vec4F_dot(&axis, &axis) < .5f)
	{
		if (pParams->m_perceptual)
			vec4F_set(&axis, k_Y_R, k_Y_G, k_Y_B, pParams->m_has_alpha ? .715f : 0);
		else
			vec4F_set(&axis, 1.0f, 1.0f, 1.0f, pParams->m_has_alpha ? 1.0f : 0);
		vec4F_normalize_in_place(&axis);
	}

	float lo = FLT_MAX, hi = -FLT_MAX;

	cfor (uniform uint32_t i = 0; i < num_pixels; i++)
	{
		const vec4F color = vec4F_from_color(&pPixels[i]);

		const vec4F q = vec4F_sub(&color, &meanColorScaled);
		const float d = vec4F_dot(&q, &axis);

		lo = minimumf(lo, d);
		hi = maximumf(hi, d);
	}

	lo *= (1.0f / 255.0f);
	hi *= (1.0f / 255.0f);

	vec4F b0 = vec4F_mul(&axis, lo);
	vec4F b1 = vec4F_mul(&axis, hi);
	vec4F c0 = vec4F_add(&meanColor, &b0);
	vec4F c1 = vec4F_add(&meanColor, &b1);
	vec4F minColor = vec4F_saturate(&c0);
	vec4F maxColor = vec4F_saturate(&c1);
				
	vec4F whiteVec;
	vec4F_set_scalar(&whiteVec, 1.0f);
	if (vec4F_dot(&minColor, &whiteVec) > vec4F_dot(&maxColor, &whiteVec))
	{
		vec4F temp = minColor;
		minColor = maxColor;
		maxColor = temp;
	}

	if (find_optimal_solution(mode, &minColor, &maxColor, pParams, pResults, pComp_params->m_pbit_search, num_pixels, pPixels, pCompAlpha) == 0.0f)
	{
		assert(pResults->m_best_overall_err == 0.0f);
		return 0.0f;
	}

	if (!refinement)
		return pResults->m_best_overall_err;
	
	for (uniform uint32_t i = 0; i < pComp_params->m_refinement_passes; i++)
	{
		vec4F xl, xh;
		vec4F_set_scalar(&xl, 0.0f);
		vec4F_set_scalar(&xh, 0.0f);
		if (pParams->m_has_alpha)
			compute_least_squares_endpoints_rgba(num_pixels, pResults->m_pSelectors, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
		else
		{
			compute_least_squares_endpoints_rgb(num_pixels, pResults->m_pSelectors, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);

			xl.m_c[3] = 255.0f;
			xh.m_c[3] = 255.0f;
		}

		xl = vec4F_mul(&xl, (1.0f / 255.0f));
		xh = vec4F_mul(&xh, (1.0f / 255.0f));

		if (find_optimal_solution(mode, &xl, &xh, pParams, pResults, pComp_params->m_pbit_search, num_pixels, pPixels, pCompAlpha) == 0.0f)
		{
			assert(pResults->m_best_overall_err == 0.0f);
			return 0.0f;
		}
	}

	if (pComp_params->m_uber_level > 0)
	{
		int selectors_temp[16], selectors_temp1[16];
		for (uniform uint32_t i = 0; i < num_pixels; i++)
			selectors_temp[i] = pResults->m_pSelectors[i];

		const uniform int max_selector = pParams->m_num_selector_weights - 1;

		uint32_t min_sel = 16;
		uint32_t max_sel = 0;
		for (uniform uint32_t i = 0; i < num_pixels; i++)
		{
			uint32_t sel = selectors_temp[i];
			min_sel = minimumu(min_sel, sel);
			max_sel = maximumu(max_sel, sel);
		}

		vec4F xl, xh;
		vec4F_set_scalar(&xl, 0.0f);
		vec4F_set_scalar(&xh, 0.0f);

		if (pComp_params->m_uber1_mask & 1)
		{
			for (uniform uint32_t i = 0; i < num_pixels; i++)
			{
				uint32_t sel = selectors_temp[i];
				if ((sel == min_sel) && (sel < (pParams->m_num_selector_weights - 1)))
					sel++;
				selectors_temp1[i] = sel;
			}
						
			if (pParams->m_has_alpha)
				compute_least_squares_endpoints_rgba(num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
			else
			{
				compute_least_squares_endpoints_rgb(num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
				xl.m_c[3] = 255.0f;
				xh.m_c[3] = 255.0f;
			}

			xl = vec4F_mul(&xl, (1.0f / 255.0f));
			xh = vec4F_mul(&xh, (1.0f / 255.0f));

			if (find_optimal_solution(mode, &xl, &xh, pParams, pResults, pComp_params->m_pbit_search, num_pixels, pPixels, pCompAlpha) == 0.0f)
			{
				assert(pResults->m_best_overall_err == 0.0f);
				return 0.0f;
			}
		}

		if (pComp_params->m_uber1_mask & 2)
		{
			for (uniform uint32_t i = 0; i < num_pixels; i++)
			{
				uint32_t sel = selectors_temp[i];
				if ((sel == max_sel) && (sel > 0))
					sel--;
				selectors_temp1[i] = sel;
			}

			if (pParams->m_has_alpha)
				compute_least_squares_endpoints_rgba(num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
			else
			{
				compute_least_squares_endpoints_rgb(num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
				xl.m_c[3] = 255.0f;
				xh.m_c[3] = 255.0f;
			}

			xl = vec4F_mul(&xl, (1.0f / 255.0f));
			xh = vec4F_mul(&xh, (1.0f / 255.0f));

			if (find_optimal_solution(mode, &xl, &xh, pParams, pResults, pComp_params->m_pbit_search, num_pixels, pPixels, pCompAlpha) == 0.0f)
			{
				assert(pResults->m_best_overall_err == 0.0f);
				return 0.0f;
			}
		}

		if (pComp_params->m_uber1_mask & 4)
		{
			for (uniform uint32_t i = 0; i < num_pixels; i++)
			{
				uint32_t sel = selectors_temp[i];
				if ((sel == min_sel) && (sel < (pParams->m_num_selector_weights - 1)))
					sel++;
				else if ((sel == max_sel) && (sel > 0))
					sel--;
				selectors_temp1[i] = sel;
			}

			if (pParams->m_has_alpha)
				compute_least_squares_endpoints_rgba(num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
			else
			{
				compute_least_squares_endpoints_rgb(num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
				xl.m_c[3] = 255.0f;
				xh.m_c[3] = 255.0f;
			}

			xl = vec4F_mul(&xl, (1.0f / 255.0f));
			xh = vec4F_mul(&xh, (1.0f / 255.0f));

			if (find_optimal_solution(mode, &xl, &xh, pParams, pResults, pComp_params->m_pbit_search, num_pixels, pPixels, pCompAlpha) == 0.0f)
			{
				assert(pResults->m_best_overall_err == 0.0f);
				return 0.0f;
			}
		}

		uniform float weightSum = pParams->m_weights[0] + pParams->m_weights[1] + pParams->m_weights[2];
		if (!pParams->m_optimize_for_color_times_alpha)
			weightSum += pParams->m_weights[3];
		const uniform float avgErrorSum = weightSum * 0.875f;
		const float uber_err_thresh = (float)(int)num_pixels * avgErrorSum;
		if ((pComp_params->m_uber_level >= 2) && (pResults->m_best_overall_err > uber_err_thresh))
		{
			const uniform int Q = (pComp_params->m_uber_level >= 4) ? (pComp_params->m_uber_level - 2) : 1;
			for (uniform int ly = -Q; ly <= 1; ly++)
			{
				for (uniform int hy = max_selector - 1; hy <= (max_selector + Q); hy++)
				{
					if ((ly == 0) && (hy == max_selector))
						continue;

					for (uniform uint32_t i = 0; i < num_pixels; i++)
						selectors_temp1[i] = (int)clampf(round((float)max_selector * ((float)(int)selectors_temp[i] - (float)ly) / ((float)hy - (float)ly)), 0, (float)max_selector);

					vec4F_set_scalar(&xl, 0.0f);
					vec4F_set_scalar(&xh, 0.0f);
					if (pParams->m_has_alpha)
						compute_least_squares_endpoints_rgba(num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
					else
					{
						compute_least_squares_endpoints_rgb(num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
						xl.m_c[3] = 255.0f;
						xh.m_c[3] = 255.0f;
					}

					xl = vec4F_mul(&xl, (1.0f / 255.0f));
					xh = vec4F_mul(&xh, (1.0f / 255.0f));

					if (find_optimal_solution(mode, &xl, &xh, pParams, pResults, pComp_params->m_pbit_search && (pComp_params->m_uber_level >= 2), num_pixels, pPixels, pCompAlpha) == 0.0f)
					{
						assert(pResults->m_best_overall_err == 0.0f);
						return 0.0f;
					}
				}
			}
		}
	}

	if ((mode <= 2) || (mode == 4) || (mode >= 6))
	{
		color_cell_compressor_results avg_results;
					
		avg_results.m_best_overall_err = pResults->m_best_overall_err;
		avg_results.m_pSelectors = pResults->m_pSelectors;
		avg_results.m_pSelectors_temp = pResults->m_pSelectors_temp;
								 
		const uint32_t r = (int)(.5f + meanColor.m_c[0] * 255.0f);
		const uint32_t g = (int)(.5f + meanColor.m_c[1] * 255.0f);
		const uint32_t b = (int)(.5f + meanColor.m_c[2] * 255.0f);
		const uint32_t a = (int)(.5f + meanColor.m_c[3] * 255.0f);

		float avg_err;
		if (mode == 0)
			avg_err = pack_mode0_to_one_color(pParams, &avg_results, r, g, b, pResults->m_pSelectors_temp, num_pixels, pPixels);
		else if (mode == 1)
			avg_err = pack_mode1_to_one_color(pParams, &avg_results, r, g, b, pResults->m_pSelectors_temp, num_pixels, pPixels);
		else if (mode == 6)
			avg_err = pack_mode6_to_one_color(pParams, &avg_results, r, g, b, a, pResults->m_pSelectors_temp, num_pixels, pPixels);
		else if (mode == 7)
			avg_err = pack_mode7_to_one_color(pParams, &avg_results, r, g, b, a, pResults->m_pSelectors_temp, num_pixels, pPixels);
		else
			avg_err = pack_mode24_to_one_color(pParams, &avg_results, r, g, b, pResults->m_pSelectors_temp, num_pixels, pPixels, pCompAlpha);

		if (avg_err < pResults->m_best_overall_err)
		{
			pResults->m_best_overall_err = avg_err;
			pResults->m_low_endpoint = avg_results.m_low_endpoint;
			pResults->m_high_endpoint = avg_results.m_high_endpoint;
			pResults->m_pbits[0] = avg_results.m_pbits[0];
			pResults->m_pbits[1] = avg_results.m_pbits[1];

			for (uniform uint32_t i = 0; i < num_pixels; i++)
				pResults->m_pSelectors[i] = pResults->m_pSelectors_temp[i];
		}
	}
					
	return pResults->m_best_overall_err;
}

// The improvements in the error estimation are:
//  - It detects channels that vary in opposite directions, instead of assuming positive correlation. (This is the more important improvement.)
//  - It accounts for non-uniform error weights when picking the closest index on the color line. (This is a minor improvement, but very cheap.)
//  - The perceptual error metric uses hard-coded non-uniform error weights. (This actually happens in the calling code.)
static float color_cell_compression_est(uniform uint32_t mode, const uniform partition_estimate_params *uniform pParams, uniform uint32_t num_pixels, const varying color_quad_i *uniform pPixels)
{
	// We want to use the covariance of the color channels to get the right signs. For example, if you have a generally red
	// area and a generally blue area meeting with a soft boundary, you'll get negative covariance in red and green.
	//
	// We can find means and covariance in a single pass. Assume two variables X[i],Y[i] with N samples that have means A,B.
	//   A = 1/N * sum[i=0..N-1]{X[i]}
	//   B = 1/N * sum[i=0..N-1]{Y[i]}
	//   V = sum[i=0..N-1]{(X[i] - A)(Y[i] - B)}
	// Lets drop [] for shorthand and simplify.
	//   V = sum{X Y - A Y - B X + A B}
	//     = sum{X Y} - A sum{Y} - B sum{X} + A B sum{1}
	//     = sum{X Y} - A B N - B A N + A B N
	//     = sum{X Y} - sum{X} sum{Y} / N
	// We only want the sign, so we can do:
	//   N V = N sum{X Y} - sum{X} sum{Y}
	// Each of these requires just a single pass. Also, note that X and Y can be the same to calculate variance.
	//
	// Normally, you wouldn't do this for numerical stability, but we're dealing with at most 16 8-bit integers.
	// The {X Y} sums are at most 16 copies of 8-bit products multiplied by at most 16, so it fits in 4 + 8 + 8 + 4 = 24 bits.
	// The {X} {Y} products multiply terms that sum up to 16 8-bit integers, so it fits in (4 + 8) * 2 = 24 bits.
	// 32-bit floats have 24-bit mantissas, so all these terms fit losslessly in a float. That means the subtraction
	// will also be exact, and this calculation is guaranteed to give the exact right result.
	//
	// As an optimization, we actually only calculate variance relative to green. This won't catch red and blue going in
	// opposite directions when green is constant.

	float sum_r = 0.0f;
	float sum_g = 0.0f;
	float sum_b = 0.0f;

	float mins_r = 255.0f;
	float mins_g = 255.0f;
	float mins_b = 255.0f;
	float maxs_r = 0.0f;
	float maxs_g = 0.0f;
	float maxs_b = 0.0f;

	float cov_gr = 0;
	float cov_gb = 0;

	for (uniform uint32_t i = 0; i < num_pixels; i++)
	{
		float r = (float)pPixels[i].m_c[0];
		float g = (float)pPixels[i].m_c[1];
		float b = (float)pPixels[i].m_c[2];

		mins_r = min(mins_r, r);
		mins_g = min(mins_g, g);
		mins_b = min(mins_b, b);

		maxs_r = max(maxs_r, r);
		maxs_g = max(maxs_g, g);
		maxs_b = max(maxs_b, b);

		sum_r += r;
		sum_g += g;
		sum_b += b;

		cov_gr += g * r;
		cov_gb += g * b;
	}

	float dr = maxs_r - mins_r;
	float dg = maxs_g - mins_g;
	float db = maxs_b - mins_b;
	float base_r = mins_r;
	float base_g = mins_g;
	float base_b = mins_b;

	if (dg > 4.0f)
	{
		if (cov_gr * num_pixels < sum_g * sum_r)
		{
			dr = -dr;
			base_r = maxs_r;
		}
		if (cov_gb * num_pixels < sum_g * sum_b)
		{
			db = -db;
			base_b = maxs_b;
		}
	}

	// Get the per-channel error weights.
	uniform const float wr = pParams->m_weights[0];
	uniform const float wg = pParams->m_weights[1];
	uniform const float wb = pParams->m_weights[2];

	// To get the palette entry:
	//   N = max index
	//   f = (color - base) . delta / |delta|^2
	//   i = clamp(round(f * N), 0, N)
	//   p = base + (i / N) * delta
	// It turns out that delta -> delta / N simplifies this. Divide delta by k:
	//   f' = (color - base) . delta / |delta|^2 * k^2 / k = f * k
	//   i = clamp(round(f * N), 0, N) = clamp(round(f' * N/k), 0, N)
	//   p = base + i * delta * N/k
	// So, if delta -> delta / N, the N/k constants become 1, and everything simplifies:
	//   f = (color - base) . delta / |delta|^2
	//   i = clamp(round(f), 0, N)
	//   p = base + i * delta
	// We want the delta, which is the length of:
	//   color - p = (color - base) - i * delta
	//
	// The error metric is a weighted sum of the squares of the per-channel errors. This is equivalent to an unweighted distance in
	// "error space". In this space, we just scale each color channel by the square root of its weight. The normal Cartesian distance
	// in this space exactly matches the error metric. This is important for non-uniform error weights. The dot product in the error
	// space and the dot product in the original color space can give different distances along the color line, which can cause it to
	// choose different interpolants, which can give different errors.
	//
	// Conveniently, we don't need to take the square roots to get this behavior. Consider 2D for brevity, with square root weights
	// "Wx" and "Wy" and color vectors "A" and "B".
	//    dot = (Wx Ax) (Wx Bx) + (Wy Ay) (Wy By)
	//        = Wx^2 (Ax Bx) + Wy^2 (Ay By)
	// So, the result is the same if we just scale the individual products in the dot product before summing them.

	dr *= pParams->m_rcp_max_selector;
	dg *= pParams->m_rcp_max_selector;
	db *= pParams->m_rcp_max_selector;
	float wdr = dr * wr;
	float wdg = dg * wg;
	float wdb = db * wb;
	float lenSq = dr * wdr + dg * wdg + db * wdb;
	float norm = 1.0f / max(1.0f / 64.0f, lenSq);	// The minimum non-zero value for lenSq is 1 / 7^2
	wdr *= norm;
	wdg *= norm;
	wdb *= norm;

	// Now we can quickly estimate error. This makes several simplifying approximations:
	//  - It assumes the best end points are the mins/maxs of the bounding box
	//  - It assumes each end point uses 8 independent bits
	//  - It assumes the interpolation fractions are 1/3 and 2/3 (they are actually 21/64 and 43/64)
	//  - It assumes the closest color to the projection on the color axis is the closest color to the pixel
	float err_est = 0.0f;
	for (uniform uint32_t i = 0; i < num_pixels; i++)
	{
		float rel_r = (float)pPixels[i].m_c[0] - base_r;
		float rel_g = (float)pPixels[i].m_c[1] - base_g;
		float rel_b = (float)pPixels[i].m_c[2] - base_b;
		float dot = rel_r * wdr + rel_g * wdg + rel_b * wdb;
		float t = clamp(round(dot), 0.0f, pParams->m_max_selector);
		rel_r -= t * dr;
		rel_g -= t * dg;
		rel_b -= t * db;
		err_est += wr * rel_r * rel_r + wg * rel_g * rel_g + wb * rel_b * rel_b;
	}
//	DEBUG_SPAM(pParams, "     result %\n", err_est);
	return err_est;
}

static float color_cell_compression_est_mode7(uniform uint32_t mode, const uniform partition_estimate_params *uniform pParams, uniform uint32_t num_pixels, const varying color_quad_i *uniform pPixels)
{
	// This is almost the same as 'color_cell_compression_est_hq' and uses the same derivations.
	// The only differences are that it includes alpha and assumes pParams->m_num_selector_weights == 4.
	float sum_r = 0.0f;
	float sum_g = 0.0f;
	float sum_b = 0.0f;
	float sum_a = 0.0f;

	float mins_r = 255.0f;
	float mins_g = 255.0f;
	float mins_b = 255.0f;
	float mins_a = 255.0f;
	float maxs_r = 0.0f;
	float maxs_g = 0.0f;
	float maxs_b = 0.0f;
	float maxs_a = 0.0f;

	float cov_gr = 0;
	float cov_gb = 0;
	float cov_ga = 0;

	for (uniform uint32_t i = 0; i < num_pixels; i++)
	{
		float r = (float)pPixels[i].m_c[0];
		float g = (float)pPixels[i].m_c[1];
		float b = (float)pPixels[i].m_c[2];
		float a = (float)pPixels[i].m_c[3];

		mins_r = min(mins_r, r);
		mins_g = min(mins_g, g);
		mins_b = min(mins_b, b);
		mins_a = min(mins_a, a);

		maxs_r = max(maxs_r, r);
		maxs_g = max(maxs_g, g);
		maxs_b = max(maxs_b, b);
		maxs_a = max(maxs_a, a);

		sum_r += r;
		sum_g += g;
		sum_b += b;
		sum_a += a;

		cov_gr += g * r;
		cov_gb += g * b;
		cov_ga += g * a;
	}

	float dr = maxs_r - mins_r;
	float dg = maxs_g - mins_g;
	float db = maxs_b - mins_b;
	float da = maxs_a - mins_a;
	float base_r = mins_r;
	float base_g = mins_g;
	float base_b = mins_b;
	float base_a = mins_a;

	if (dg > 4.0f)
	{
		if (cov_gr * num_pixels < sum_g * sum_r)
		{
			dr = -dr;
			base_r = maxs_r;
		}
		if (cov_gb * num_pixels < sum_g * sum_b)
		{
			db = -db;
			base_b = maxs_b;
		}
		if (cov_ga * num_pixels < sum_g * sum_a)
		{
			da = -da;
			base_a = maxs_a;
		}
	}

	// Get the per-channel error weights.
	uniform float wr = pParams->m_weights[0];
	uniform float wg = pParams->m_weights[1];
	uniform float wb = pParams->m_weights[2];
	uniform float wa = pParams->m_weights[3];

	// get lookup constants
	dr *= pParams->m_rcp_max_selector;
	dg *= pParams->m_rcp_max_selector;
	db *= pParams->m_rcp_max_selector;
	da *= pParams->m_rcp_max_selector;
	float wdr = dr * wr;
	float wdg = dg * wg;
	float wdb = db * wb;
	float wda = da * wa;
	float lenSq = dr * wdr + dg * wdg + db * wdb + da * wda;
	float norm = 1.0f / max(1.0f / 16.0f, lenSq);	// The minimum non-zero value for lenSq is 1 / 3^2
	wdr *= norm;
	wdg *= norm;
	wdb *= norm;
	wda *= norm;

	// estimate error
	// NOTE: Simple per-channel weighting actually gave lower error for color times alpha in my test cases than trying to
	// account for color times alpha in this error metric. It's also less work.
	float err_est = 0.0f;
	for (uniform uint32_t i = 0; i < num_pixels; i++)
	{
		float rel_r = (float)pPixels[i].m_c[0] - base_r;
		float rel_g = (float)pPixels[i].m_c[1] - base_g;
		float rel_b = (float)pPixels[i].m_c[2] - base_b;
		float rel_a = (float)pPixels[i].m_c[3] - base_a;
		float dot = rel_r * wdr + rel_g * wdg + rel_b * wdb + rel_a * wda;
		float t = clamp(round(dot), 0.0f, pParams->m_max_selector);
		rel_r -= t * dr;
		rel_g -= t * dg;
		rel_b -= t * db;
		rel_a -= t * da;
		err_est += wr * rel_r * rel_r + wg * rel_g * rel_g + wb * rel_b * rel_b + wa * rel_a * rel_a;
	}
	return err_est;
}

static inline void partition_estimate_params_init(uniform partition_estimate_params *uniform pParams, uniform uint32_t mode, const uniform bc7e_compress_block_params *uniform pComp_params)
{
	pParams->m_debug_spam = pComp_params->m_debugSpam;

	pParams->m_max_selector = (float)(1 << g_bc7_color_index_bitcount[mode]) - 1.0f;
	pParams->m_rcp_max_selector = 1.0f / pParams->m_max_selector;

	if (pComp_params->m_perceptual)
	{
		pParams->m_weights[0] = k_est_wt_r;
		pParams->m_weights[1] = k_est_wt_g;
		pParams->m_weights[2] = k_est_wt_b;
		pParams->m_weights[3] = k_est_wt_a;
	}
	else
	{
		pParams->m_weights[0] = pComp_params->m_weights[0];
		pParams->m_weights[1] = pComp_params->m_weights[1];
		pParams->m_weights[2] = pComp_params->m_weights[2];
		pParams->m_weights[3] = pComp_params->m_weights[3];
		if (mode >= 6)
		{
			pParams->m_weights[0] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[0];
			pParams->m_weights[1] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[1];
			pParams->m_weights[2] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[2];
			pParams->m_weights[3] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[3];
		}
	}

	// For performance, "m_optimize_for_color_times_alpha" is ignored for color error metrics in evaluating partitions.
	// However, we do use it to set the alpha weight reasonably.
	// (In fact, this simpler alpha weight had lower final error than trying to account for color times alpha.)
	const bool optimize_for_color_times_alpha = (pComp_params->m_optimize_for != BC7E_OPTIMIZE_FOR_INDEPENDENT_CHANNELS);
	if (optimize_for_color_times_alpha)
		pParams->m_weights[3] = pParams->m_weights[0] + pParams->m_weights[1] + pParams->m_weights[2];
}

static uint32_t estimate_partition(uniform uint32_t mode, const varying color_quad_i *uniform pPixels, const uniform bc7e_compress_block_params *uniform pComp_params)
{
	const uniform uint32_t total_subsets = g_bc7_num_subsets[mode];
	uniform uint32_t total_partitions = minimumu(pComp_params->m_max_partitions_mode[mode], 1U << g_bc7_partition_bits[mode]);

	if (total_partitions <= 1)
		return 0;

	float best_err = FLT_MAX;
	uint32_t best_partition = 0;

	uniform partition_estimate_params params;
	partition_estimate_params_init(&params, mode, pComp_params);

	for (uniform uint32_t partition = 0; partition < total_partitions; partition++)
	{
		const int *uniform pPartition = (total_subsets == 3) ? &g_bc7_partition3[partition * 16] : &g_bc7_partition2[partition * 16];

		varying color_quad_i subset_colors[3][16];
		uniform uint32_t subset_total_colors[3];
		subset_total_colors[0] = 0;
		subset_total_colors[1] = 0;
		subset_total_colors[2] = 0;

		for (uniform uint32_t index = 0; index < 16; index++)
		{
			const uniform uint32_t p = pPartition[index];

			subset_colors[p][subset_total_colors[p]] = pPixels[index];
			subset_total_colors[p]++;
		}

		float total_subset_err = 0;

		for (uniform uint32_t subset = 0; subset < total_subsets; subset++)
		{
			float err;
			if (mode == 7)
				err = color_cell_compression_est_mode7(mode, &params, subset_total_colors[subset], &subset_colors[subset][0]);
			else
				err = color_cell_compression_est(mode, &params, subset_total_colors[subset], &subset_colors[subset][0]);

			total_subset_err += err;

		} // subset

		if (total_subset_err < best_err)
		{
			best_err = total_subset_err;
			best_partition = partition;
			if (best_err == 0.0f)
				break;
		}

		if (total_subsets == 2)
		{
			if ((partition == BC7E_2SUBSET_CHECKERBOARD_PARTITION_INDEX) && (best_partition != BC7E_2SUBSET_CHECKERBOARD_PARTITION_INDEX))
				break;
		}

	} // partition

	return best_partition;
}

struct solution
{
	uint32_t m_index;
	float m_err;
};

static uniform uint32_t estimate_partition_list(uniform uint32_t mode, const varying color_quad_i *uniform pPixels, const uniform bc7e_compress_block_params *uniform pComp_params, 
	varying solution *uniform pSolutions, uniform int32_t max_solutions)
{
	const uniform int32_t orig_max_solutions = max_solutions;

	const uniform uint32_t total_subsets = g_bc7_num_subsets[mode];
	uniform uint32_t total_partitions = minimumu(pComp_params->m_max_partitions_mode[mode], 1U << g_bc7_partition_bits[mode]);

	if (total_partitions <= 1)
	{
		pSolutions[0].m_index = 0;
		pSolutions[0].m_err = 0.0f;			// this is a nonsense error, but it doesn't matter
		return 1;
	}
	else if (max_solutions >= total_partitions)
	{
		for (uniform int i = 0; i < total_partitions; i++)
		{
			pSolutions[i].m_index = i;
			pSolutions[i].m_err = (float)i;	// this is a nonsense error, but it doesn't matter and it keeps them sorted
		}
		return total_partitions;
	}

	const uniform int32_t HIGH_FREQUENCY_SORTED_PARTITION_THRESHOLD = 4;
	if (total_subsets == 2)
	{
		if (max_solutions < HIGH_FREQUENCY_SORTED_PARTITION_THRESHOLD)
			max_solutions = HIGH_FREQUENCY_SORTED_PARTITION_THRESHOLD;
	}

	uniform partition_estimate_params params;
	partition_estimate_params_init(&params, mode, pComp_params);

	uniform int32_t num_solutions = 0;

	for (uniform uint32_t partition = 0; partition < total_partitions; partition++)
	{
//		DEBUG_SPAM(&params, "   err est partition %\n", partition);
		const int *uniform pPartition = (total_subsets == 3) ? &g_bc7_partition3[partition * 16] : &g_bc7_partition2[partition * 16];

		varying color_quad_i subset_colors[3][16];
		uniform uint32_t subset_total_colors[3];
		subset_total_colors[0] = 0;
		subset_total_colors[1] = 0;
		subset_total_colors[2] = 0;

		for (uniform uint32_t index = 0; index < 16; index++)
		{
			const uniform uint32_t p = pPartition[index];

			subset_colors[p][subset_total_colors[p]] = pPixels[index];
			subset_total_colors[p]++;
		}
				
		float total_subset_err = 0.0f;

		for (uniform uint32_t subset = 0; subset < total_subsets; subset++)
		{
			float err;
			if (mode == 7)
				err = color_cell_compression_est_mode7(mode, &params, subset_total_colors[subset], &subset_colors[subset][0]);
			else
				err = color_cell_compression_est(mode, &params, subset_total_colors[subset], &subset_colors[subset][0]);

			total_subset_err += err;

//			DEBUG_SPAM(&params, "     subset % error %\n", subset, err);
		} // subset
//		DEBUG_SPAM(&params, "    total error %\n", total_subset_err);

		int32_t i;
		for (i = 0; i < num_solutions; i++)
		{
#pragma ignore warning(perf)
			if (total_subset_err < pSolutions[i].m_err)
				break;
		}
						
		if (i < num_solutions)
		{
			int32_t solutions_to_move = (max_solutions - 1) - i;
			int32_t num_elements_at_i = num_solutions - i;
			if (solutions_to_move > num_elements_at_i)
				solutions_to_move = num_elements_at_i;
																
			assert(((i + 1) + solutions_to_move) <= max_solutions);
			assert((i + solutions_to_move) <= num_solutions);
			
			for (int32_t j = solutions_to_move - 1; j >= 0; --j)
			{
#pragma ignore warning(perf)
				pSolutions[i + j + 1] = pSolutions[i + j];
			}
		}

		if (num_solutions < max_solutions)
			num_solutions++;

		if (i < num_solutions)
		{
#pragma ignore warning(perf)
			pSolutions[i].m_err = total_subset_err;

#pragma ignore warning(perf)
			pSolutions[i].m_index = partition;
		}

#if BC7E_NON_DETERMINISTIC
		if ((total_subsets == 2) && (partition == BC7E_2SUBSET_CHECKERBOARD_PARTITION_INDEX))
		{
			if (all(i >= HIGH_FREQUENCY_SORTED_PARTITION_THRESHOLD))
				break;
		}
#endif

	} // partition

#if 0
	for (uniform int i = 0; i < num_solutions; i++)
	{
		assert(pSolutions[i].m_index < total_partitions);
	}

	for (uniform int i = 0; i < (num_solutions - 1); i++)
	{
		assert(pSolutions[i].m_err <= pSolutions[i + 1].m_err);
	}
#endif

	return min(num_solutions, orig_max_solutions);
}

static inline void set_block_bits(uint8_t *pBytes, uint32_t val, uint32_t num_bits, varying uint32_t *uniform pCur_ofs)
{
	assert(num_bits < 32);
	uint32_t limit = 1U << num_bits;
	assert(val < limit);
		
	while (num_bits)
	{
		const uint32_t n = minimumu(8 - (*pCur_ofs & 7), num_bits);

#pragma ignore warning(perf)
		pBytes[*pCur_ofs >> 3] |= (uint8_t)(val << (*pCur_ofs & 7));

		val >>= n;
		num_bits -= n;
		*pCur_ofs += n;
	}

	assert(*pCur_ofs <= 128);
}

struct bc7_optimization_results
{
	uint32_t m_mode;
	uint32_t m_partition;
	int m_selectors[16];
	int m_alpha_selectors[16];
	color_quad_i m_low[3];
	color_quad_i m_high[3];
	uint32_t m_pbits[3][2];
	uint32_t m_rotation;
	uint32_t m_index_selector;
};

#if BC7E_DEBUG_SPAM
static void spam_bc7_block(varying bc7_optimization_results *uniform pResults)
{
	print("block summary:\n");
	const uint32_t mode = pResults->m_mode;
	print("  mode %\n", mode);
	if (mode == 4 || mode == 5)
		print("  rotation %\n", pResults->m_rotation);
	if (mode == 4)
		print("  index selector %\n", pResults->m_index_selector);

#pragma ignore warning(perf)
	const uint32_t total_subsets = g_bc7_num_subsets[mode];
	if (total_subsets > 1)
		print("  partition %\n", pResults->m_partition);
	for (uniform uint32_t s = 0; s < total_subsets; s++)
	{
		print("  s % r0 %\n", s, pResults->m_low [s].m_c[0]);
		print("  s % r1 %\n", s, pResults->m_high[s].m_c[0]);
		print("  s % g0 %\n", s, pResults->m_low [s].m_c[1]);
		print("  s % g1 %\n", s, pResults->m_high[s].m_c[1]);
		print("  s % b0 %\n", s, pResults->m_low [s].m_c[2]);
		print("  s % b1 %\n", s, pResults->m_high[s].m_c[2]);
		if (mode >= 4)
		{
			print("  s % a0 %\n", s, pResults->m_low [s].m_c[3]);
			print("  s % a1 %\n", s, pResults->m_high[s].m_c[3]);
		}
#pragma ignore warning(perf)
		if (g_bc7_mode_has_p_bits[mode])
		{
			print("  s % p0 %\n", s, pResults->m_pbits[s][0]);
#pragma ignore warning(perf)
			if (!g_bc7_mode_has_shared_p_bits[mode])
				print("  s % p1 %\n", s, pResults->m_pbits[s][1]);
		}
	}
	for (uniform uint32_t s = 0; s < 16; ++s)
		print("  c sel %,%: %\n", s & 3, s >> 2, pResults->m_selectors[s]);
	if (mode == 4 || mode == 5)
	{
		for (uniform uint32_t s = 0; s < 16; ++s)
			print("  a sel %,%: %\n", s & 3, s >> 2, pResults->m_alpha_selectors[s]);
	}
}
#endif // #if BC7E_DEBUG_SPAM

static void encode_bc7_block(void *pBlock, const varying bc7_optimization_results *uniform pResults)
{
	const uint32_t best_mode = pResults->m_mode;

#pragma ignore warning(perf)
	const uint32_t total_subsets = g_bc7_num_subsets[best_mode];

#pragma ignore warning(perf)
	const uint32_t total_partitions = 1 << g_bc7_partition_bits[best_mode];

	const int *pPartition;
	if (total_subsets == 1)
		pPartition = &g_bc7_partition1[0];
	else if (total_subsets == 2)
		pPartition = &g_bc7_partition2[pResults->m_partition * 16];
	else
		pPartition = &g_bc7_partition3[pResults->m_partition * 16];

	int color_selectors[16];
	for (uniform int i = 0; i < 16; i++)
		color_selectors[i] = pResults->m_selectors[i];

	int alpha_selectors[16];
	for (uniform int i = 0; i < 16; i++)
		alpha_selectors[i] = pResults->m_alpha_selectors[i];

	color_quad_i low[3], high[3];
	low[0] = pResults->m_low[0];
	low[1] = pResults->m_low[1];
	low[2] = pResults->m_low[2];

	high[0] = pResults->m_high[0];
	high[1] = pResults->m_high[1];
	high[2] = pResults->m_high[2];
	
	uint32_t pbits[3][2];
	for (uniform int i = 0; i < 3; i++)
	{
		pbits[i][0] = pResults->m_pbits[i][0];
		pbits[i][1] = pResults->m_pbits[i][1];
	}

	int anchor[3];
	anchor[0] = -1;
	anchor[1] = -1;
	anchor[2] = -1;

	for (uniform uint32_t k = 0; k < total_subsets; k++)
	{
		uint32_t anchor_index = 0;
		if (k)
		{
			if ((total_subsets == 3) && (k == 1))
			{
#pragma ignore warning(perf)
				anchor_index = g_bc7_table_anchor_index_third_subset_1[pResults->m_partition];
			}
			else if ((total_subsets == 3) && (k == 2))
			{
#pragma ignore warning(perf)
				anchor_index = g_bc7_table_anchor_index_third_subset_2[pResults->m_partition];
			}
			else
			{
#pragma ignore warning(perf)
				anchor_index = g_bc7_table_anchor_index_second_subset[pResults->m_partition];
			}
		}

		anchor[k] = anchor_index;

		const uint32_t color_index_bits = get_bc7_color_index_size(best_mode, pResults->m_index_selector);
		const uint32_t num_color_indices = 1 << color_index_bits;

#pragma ignore warning(perf)
		if (color_selectors[anchor_index] & (num_color_indices >> 1))
		{
			for (uniform uint32_t i = 0; i < 16; i++)
			{
#pragma ignore warning(perf)
				if (pPartition[i] == k)
					color_selectors[i] = (num_color_indices - 1) - color_selectors[i];
			}

			if (get_bc7_mode_has_seperate_alpha_selectors(best_mode))
			{
				for (uniform uint32_t q = 0; q < 3; q++)
				{
					int t = low[k].m_c[q];
					low[k].m_c[q] = high[k].m_c[q];
					high[k].m_c[q] = t;
				}
			}
			else
			{
				color_quad_i tmp = low[k];
				low[k] = high[k];
				high[k] = tmp;
			}

#pragma ignore warning(perf)
			if (!g_bc7_mode_has_shared_p_bits[best_mode])
			{
				uint32_t t = pbits[k][0];
				pbits[k][0] = pbits[k][1];
				pbits[k][1] = t;
			}
		}

		if (get_bc7_mode_has_seperate_alpha_selectors(best_mode))
		{
			const uint32_t alpha_index_bits = get_bc7_alpha_index_size(best_mode, pResults->m_index_selector);
			const uint32_t num_alpha_indices = 1 << alpha_index_bits;

#pragma ignore warning(perf)
			if (alpha_selectors[anchor_index] & (num_alpha_indices >> 1))
			{
				for (uniform uint32_t i = 0; i < 16; i++)
				{
#pragma ignore warning(perf)
					if (pPartition[i] == k)
						alpha_selectors[i] = (num_alpha_indices - 1) - alpha_selectors[i];
				}

				int t = low[k].m_c[3];
				low[k].m_c[3] = high[k].m_c[3];
				high[k].m_c[3] = t;
			}
		}
	}

	uint8_t *pBlock_bytes = (uint8_t *)(pBlock);
	memset(pBlock_bytes, 0, BC7E_BLOCK_SIZE);

	uint32_t cur_bit_ofs = 0;
		
	set_block_bits(pBlock_bytes, 1 << best_mode, best_mode + 1, &cur_bit_ofs);

	if ((best_mode == 4) || (best_mode == 5))
		set_block_bits(pBlock_bytes, pResults->m_rotation, 2, &cur_bit_ofs);

	if (best_mode == 4)
		set_block_bits(pBlock_bytes, pResults->m_index_selector, 1, &cur_bit_ofs);

	if (total_partitions > 1)
		set_block_bits(pBlock_bytes, pResults->m_partition, (total_partitions == 64) ? 6 : 4, &cur_bit_ofs);

	const uint32_t total_comps = (best_mode >= 4) ? 4 : 3;
	for (uniform uint32_t comp = 0; comp < total_comps; comp++)
	{
		for (uniform uint32_t subset = 0; subset < total_subsets; subset++)
		{
			set_block_bits(pBlock_bytes, low[subset].m_c[comp], (comp == 3) ? g_bc7_alpha_precision_table[best_mode] : g_bc7_color_precision_table[best_mode], &cur_bit_ofs);
			set_block_bits(pBlock_bytes, high[subset].m_c[comp], (comp == 3) ? g_bc7_alpha_precision_table[best_mode] : g_bc7_color_precision_table[best_mode], &cur_bit_ofs);
		}
	}

#pragma ignore warning(perf)
	if (g_bc7_mode_has_p_bits[best_mode])
	{
		for (uniform uint32_t subset = 0; subset < total_subsets; subset++)
		{
			set_block_bits(pBlock_bytes, pbits[subset][0], 1, &cur_bit_ofs);
#pragma ignore warning(perf)
			if (!g_bc7_mode_has_shared_p_bits[best_mode])
				set_block_bits(pBlock_bytes, pbits[subset][1], 1, &cur_bit_ofs);
		}
	}

	for (uniform uint32_t y = 0; y < 4; y++)
	{
		for (uniform uint32_t x = 0; x < 4; x++)
		{
			uniform int idx = x + y * 4;

			uint32_t n = pResults->m_index_selector ? get_bc7_alpha_index_size(best_mode, pResults->m_index_selector) : get_bc7_color_index_size(best_mode, pResults->m_index_selector);

			if ((idx == anchor[0]) || (idx == anchor[1]) || (idx == anchor[2]))
				n--;

			set_block_bits(pBlock_bytes, pResults->m_index_selector ? alpha_selectors[idx] : color_selectors[idx], n, &cur_bit_ofs);
		}
	}

	if (get_bc7_mode_has_seperate_alpha_selectors(best_mode))
	{
		for (uniform uint32_t y = 0; y < 4; y++)
		{
			for (uniform uint32_t x = 0; x < 4; x++)
			{
				int idx = x + y * 4;

				uint32_t n = pResults->m_index_selector ? get_bc7_color_index_size(best_mode, pResults->m_index_selector) : get_bc7_alpha_index_size(best_mode, pResults->m_index_selector);

				if ((idx == anchor[0]) || (idx == anchor[1]) || (idx == anchor[2]))
					n--;

				set_block_bits(pBlock_bytes, pResults->m_index_selector ? color_selectors[idx] : alpha_selectors[idx], n, &cur_bit_ofs);
			}
		}
	}

	assert(cur_bit_ofs == 128);
}

static inline void encode_bc7_block_mode6(void *pBlock, varying bc7_optimization_results *uniform pResults)
{
	color_quad_i low, high;
	uint32_t pbits[2];
		
	uint32_t invert_selectors = 0;
	if (pResults->m_selectors[0] & 8)
	{
		invert_selectors = 15;
							
		low = pResults->m_high[0];
		high = pResults->m_low[0];

		pbits[0] = pResults->m_pbits[0][1];
		pbits[1] = pResults->m_pbits[0][0];
	}
	else
	{
		low = pResults->m_low[0];
		high = pResults->m_high[0];

		pbits[0] = pResults->m_pbits[0][0];
		pbits[1] = pResults->m_pbits[0][1];
	}

	uint64_t l = 0, h = 0;

	l = 1 << 6;

	l |= (low.m_c[0] << 7);
	l |= (high.m_c[0] << 14);

	l |= (low.m_c[1] << 21);
	l |= ((uint64_t)high.m_c[1] << 28);

	l |= ((uint64_t)low.m_c[2] << 35);
	l |= ((uint64_t)high.m_c[2] << 42);

	l |= ((uint64_t)low.m_c[3] << 49);
	l |= ((uint64_t)high.m_c[3] << 56);

	l |= ((uint64_t)pbits[0] << 63);
		
	h = pbits[1];
		
	h |= ((invert_selectors ^ pResults->m_selectors[0]) << 1);

	// TODO: Just invert all these bits in one single operation, not as individual
	h |= ((invert_selectors ^ pResults->m_selectors[1]) << 4);
	h |= ((invert_selectors ^ pResults->m_selectors[2]) << 8);
	h |= ((invert_selectors ^ pResults->m_selectors[3]) << 12);
	h |= ((invert_selectors ^ pResults->m_selectors[4]) << 16);
	
	h |= ((invert_selectors ^ pResults->m_selectors[5]) << 20);
	h |= ((invert_selectors ^ pResults->m_selectors[6]) << 24);
	h |= ((invert_selectors ^ pResults->m_selectors[7]) << 28);
	h |= ((uint64_t)(invert_selectors ^ pResults->m_selectors[8]) << 32);

	h |= ((uint64_t)(invert_selectors ^ pResults->m_selectors[9]) << 36);
	h |= ((uint64_t)(invert_selectors ^ pResults->m_selectors[10]) << 40);
	h |= ((uint64_t)(invert_selectors ^ pResults->m_selectors[11]) << 44);
	h |= ((uint64_t)(invert_selectors ^ pResults->m_selectors[12]) << 48);

	h |= ((uint64_t)(invert_selectors ^ pResults->m_selectors[13]) << 52);
	h |= ((uint64_t)(invert_selectors ^ pResults->m_selectors[14]) << 56);
	h |= ((uint64_t)(invert_selectors ^ pResults->m_selectors[15]) << 60);

#pragma ignore warning(perf)	
	((uint64_t *)(pBlock))[0] = l;

#pragma ignore warning(perf)
	((uint64_t *)(pBlock))[1] = h;
}

static void handle_alpha_block_mode4(const varying color_quad_i *uniform pPixels, const uniform bc7e_compress_block_params *uniform pComp_params, uniform color_cell_compressor_params *uniform pParams, uint32_t lo_a, uint32_t hi_a, 
	varying bc7_optimization_results *uniform pOpt_results4, varying float *uniform pMode4_err)
{
	pParams->m_has_alpha = false;
	pParams->m_comp_bits = 5;
	pParams->m_has_pbits = false;
	pParams->m_endpoints_share_pbit = false;				
	assert(pParams->m_perceptual == pComp_params->m_perceptual);

	for (uniform uint32_t index_selector = 0; index_selector < 2; index_selector++)
	{
		if ((pComp_params->m_mode4_index_mask & (1 << index_selector)) == 0)
			continue;

		if (index_selector)
		{
			pParams->m_pSelector_weights = g_bc7_weights3;
			pParams->m_pSelector_weightsx = (const vec4F * uniform)&g_bc7_weights3x[0];
			pParams->m_num_selector_weights = 8;
		}
		else
		{
			pParams->m_pSelector_weights = g_bc7_weights2;
			pParams->m_pSelector_weightsx = (const vec4F * uniform)&g_bc7_weights2x[0];
			pParams->m_num_selector_weights = 4;
		}
		DEBUG_SPAM(pParams, " handle_alpha_block (mode 4, rotation %, index selector %)\n", pParams->m_rotation, index_selector);

		uint32_t la = minimumi((lo_a + 2) >> 2, 63);
		uint32_t ha = minimumi((hi_a + 2) >> 2, 63);

		if (la == ha)
		{
			if (lo_a != hi_a)
			{
				if (ha != 63)
					ha++;
				else if (la != 0)
					la--;
			}
		}

		int32_t best_alpha_err = INT32_MAX;	// 16 pixels each having 256 error squared fits in 20 bits
		uint32_t best_la = 0, best_ha = 0;
		int best_alpha_selectors[16];
		int32_t best_alpha_values[16];
						
		for (uniform int32_t pass = 0; pass < 2; pass++)
		{
			int32_t vals[8];

			if (index_selector == 0)
			{
				vals[0] = (la << 2) | (la >> 4);
				vals[7] = (ha << 2) | (ha >> 4);

				for (uniform uint32_t i = 1; i < 7; i++)
					vals[i] = (vals[0] * (64 - g_bc7_weights3[i]) + vals[7] * g_bc7_weights3[i] + 32) >> 6;
			}
			else
			{
				vals[0] = (la << 2) | (la >> 4);
				vals[3] = (ha << 2) | (ha >> 4);

				const uniform int32_t w_s1 = 21, w_s2 = 43;
				vals[1] = (vals[0] * (64 - w_s1) + vals[3] * w_s1 + 32) >> 6;
				vals[2] = (vals[0] * (64 - w_s2) + vals[3] * w_s2 + 32) >> 6;
			}

			int32_t trial_alpha_err = 0;

			int trial_alpha_selectors[16];
			int32_t trial_alpha_values[16];
			for (uniform uint32_t i = 0; i < 16; i++)
			{
				const int32_t a = pPixels[i].m_c[3];

				int s = 0;
				int32_t bv = vals[0];
				int32_t be = iabs32(a - bv);

				int e = iabs32(a - vals[1]); if (e < be) { be = e; s = 1; bv = vals[1]; }
				e = iabs32(a - vals[2]); if (e < be) { be = e; s = 2; bv = vals[2]; }
				e = iabs32(a - vals[3]); if (e < be) { be = e; s = 3; bv = vals[3]; }

				if (index_selector == 0)
				{
					e = iabs32(a - vals[4]); if (e < be) { be = e; s = 4; bv = vals[4]; }
					e = iabs32(a - vals[5]); if (e < be) { be = e; s = 5; bv = vals[5]; }
					e = iabs32(a - vals[6]); if (e < be) { be = e; s = 6; bv = vals[6]; }
					e = iabs32(a - vals[7]); if (e < be) { be = e; s = 7; bv = vals[7]; }
				}

				trial_alpha_err += be * be;

				trial_alpha_selectors[i] = s;
				trial_alpha_values[i] = bv;
			}

			if (trial_alpha_err < best_alpha_err)
			{
				best_alpha_err = trial_alpha_err;
				best_la = la;
				best_ha = ha;
				for (uniform uint32_t i = 0; i < 16; i++)
				{
					best_alpha_selectors[i] = trial_alpha_selectors[i];
					best_alpha_values[i] = trial_alpha_values[i];
				}
			}

			if (pass == 0) 
			{
				float xa, xb;
				compute_least_squares_endpoints_a(16, trial_alpha_selectors, index_selector ? (const vec4F * uniform)&g_bc7_weights2x[0] : (const vec4F * uniform)&g_bc7_weights3x[0], &xa, &xb, pPixels);
				const float xl = min(xa, xb);
				const float xh = max(xa, xb);
				la = (int)round(clampf(xl * (63.0f / 255.0f), 0.0f, 63.0f));
				ha = (int)round(clampf(xh * (63.0f / 255.0f), 0.0f, 63.0f));
			}
						
		} // pass

		if (pComp_params->m_uber_level > 0)
		{
			const uniform int D = min((int)pComp_params->m_uber_level, 3);
			for (uniform int ld = -D; ld <= D; ld++)
			{
				for (uniform int hd = -D; hd <= D; hd++)
				{
					la = clamp((int)best_la + ld, 0, 63);
					ha = clamp((int)best_ha + hd, 0, 63);
					
					int32_t vals[8];

					if (index_selector == 0)
					{
						vals[0] = (la << 2) | (la >> 4);
						vals[7] = (ha << 2) | (ha >> 4);

						for (uniform uint32_t i = 1; i < 7; i++)
							vals[i] = (vals[0] * (64 - g_bc7_weights3[i]) + vals[7] * g_bc7_weights3[i] + 32) >> 6;
					}
					else
					{
						vals[0] = (la << 2) | (la >> 4);
						vals[3] = (ha << 2) | (ha >> 4);

						const uniform int32_t w_s1 = 21, w_s2 = 43;
						vals[1] = (vals[0] * (64 - w_s1) + vals[3] * w_s1 + 32) >> 6;
						vals[2] = (vals[0] * (64 - w_s2) + vals[3] * w_s2 + 32) >> 6;
					}

					int32_t trial_alpha_err = 0;

					int trial_alpha_selectors[16];
					int32_t trial_alpha_values[16];
					for (uniform uint32_t i = 0; i < 16; i++)
					{
						const int32_t a = pPixels[i].m_c[3];

						int s = 0;
						int32_t bv = vals[0];
						int32_t be = iabs32(a - bv);

						int e = iabs32(a - vals[1]); if (e < be) { be = e; s = 1; bv = vals[1]; }
						e = iabs32(a - vals[2]); if (e < be) { be = e; s = 2; bv = vals[2]; }
						e = iabs32(a - vals[3]); if (e < be) { be = e; s = 3; bv = vals[3]; }

						if (index_selector == 0)
						{
							e = iabs32(a - vals[4]); if (e < be) { be = e; s = 4; bv = vals[4]; }
							e = iabs32(a - vals[5]); if (e < be) { be = e; s = 5; bv = vals[5]; }
							e = iabs32(a - vals[6]); if (e < be) { be = e; s = 6; bv = vals[6]; }
							e = iabs32(a - vals[7]); if (e < be) { be = e; s = 7; bv = vals[7]; }
						}

						trial_alpha_err += be * be;

						trial_alpha_selectors[i] = s;
						trial_alpha_values[i] = bv;
					}

					if (trial_alpha_err < best_alpha_err)
					{
						best_alpha_err = trial_alpha_err;
						best_la = la;
						best_ha = ha;
						for (uniform uint32_t i = 0; i < 16; i++)
						{
							best_alpha_selectors[i] = trial_alpha_selectors[i];
							best_alpha_values[i] = trial_alpha_values[i];
						}
					}
				
				} // hd

			} // ld
		}

		color_cell_compressor_results results;
		
		int selectors[16];
		results.m_pSelectors = selectors;

		int selectors_temp[16];
		results.m_pSelectors_temp = selectors_temp;
				
		float trial_err = color_cell_compression(4, pParams, &results, pComp_params, 16, pPixels, best_alpha_values, true);
		assert(trial_err == results.m_best_overall_err);

		// If we don't optimize for color * alpha, and if we're not perceptual or we're not rotated, then color and alpha errors are independent. 
		// If we optimize for color * alpha, or if we're perceptual with rotation, then color_cell_compression must include alpha error.
		if (!pParams->m_optimize_for_color_times_alpha && (!pParams->m_perceptual || pParams->m_rotation == 0))
			trial_err += (float)best_alpha_err * pParams->m_weights[3];

		DEBUG_SPAM(pParams, "  error %\n", trial_err);

		if (trial_err < *pMode4_err)
		{
			DEBUG_SPAM(pParams, "  *** better than %\n", *pMode4_err);
			*pMode4_err = trial_err;

			pOpt_results4->m_mode = 4;
			pOpt_results4->m_index_selector = index_selector;
			pOpt_results4->m_rotation = 0;
			pOpt_results4->m_partition = 0;

			pOpt_results4->m_low[0] = results.m_low_endpoint;
			pOpt_results4->m_high[0] = results.m_high_endpoint;
			pOpt_results4->m_low[0].m_c[3] = best_la;
			pOpt_results4->m_high[0].m_c[3] = best_ha;

			for (uniform uint32_t i = 0; i < 16; i++)
				pOpt_results4->m_selectors[i] = selectors[i];

			for (uniform uint32_t i = 0; i < 16; i++)
				pOpt_results4->m_alpha_selectors[i] = best_alpha_selectors[i];
		}

	} // index_selector
}

static void handle_alpha_block_mode5(const varying color_quad_i *uniform pPixels, const uniform bc7e_compress_block_params *uniform pComp_params, uniform color_cell_compressor_params *uniform pParams, uint32_t lo_a, uint32_t hi_a, 
	varying bc7_optimization_results *uniform pOpt_results5, varying float *uniform pMode5_err)
{
	pParams->m_pSelector_weights = g_bc7_weights2;
	pParams->m_pSelector_weightsx = (const vec4F * uniform)&g_bc7_weights2x[0];
	pParams->m_num_selector_weights = 4;

	pParams->m_comp_bits = 7;
	pParams->m_has_alpha = false;
	pParams->m_has_pbits = false;
	pParams->m_endpoints_share_pbit = false;				
	
	assert(pParams->m_perceptual == pComp_params->m_perceptual);

	int32_t mode5_alpha_err;
	int32_t best_alpha_values[16];
	uint8_t best_alpha_lo;
	uint8_t best_alpha_hi;
	cif (lo_a == hi_a)
	{
		mode5_alpha_err = 0;
		best_alpha_lo = lo_a;
		best_alpha_hi = lo_a;
		for (uniform uint32_t i = 0; i < 16; i++)
		{
			pOpt_results5->m_alpha_selectors[i] = 0;
			best_alpha_values[i] = lo_a;
		}
	}
	else
	{
		mode5_alpha_err = INT32_MAX;
		for (uniform uint32_t pass = 0; pass < 2; pass++)
		{
			int32_t vals[4];
			vals[0] = lo_a;
			vals[3] = hi_a;

			const uniform int32_t w_s1 = 21, w_s2 = 43;
			vals[1] = (vals[0] * (64 - w_s1) + vals[3] * w_s1 + 32) >> 6;
			vals[2] = (vals[0] * (64 - w_s2) + vals[3] * w_s2 + 32) >> 6;

			int trial_alpha_selectors[16];
			int32_t trial_alpha_values[16];

			int32_t trial_alpha_err = 0;
			for (uniform uint32_t i = 0; i < 16; i++)
			{
				const int32_t a = pPixels[i].m_c[3];

				int s = 0;
				int32_t bv = vals[0];
				int32_t be = iabs32(a - bv);
				int e = iabs32(a - vals[1]); if (e < be) { be = e; s = 1; bv = vals[1]; }
				e = iabs32(a - vals[2]); if (e < be) { be = e; s = 2; bv = vals[2]; }
				e = iabs32(a - vals[3]); if (e < be) { be = e; s = 3; bv = vals[3]; }

				trial_alpha_selectors[i] = s;
				trial_alpha_values[i] = bv;

				trial_alpha_err += be * be;
			}

			if (trial_alpha_err < mode5_alpha_err)
			{
				mode5_alpha_err = trial_alpha_err;
				best_alpha_lo = lo_a;
				best_alpha_hi = hi_a;
				for (uniform uint32_t i = 0; i < 16; i++)
				{
					pOpt_results5->m_alpha_selectors[i] = trial_alpha_selectors[i];
					best_alpha_values[i] = trial_alpha_values[i];
				}
			}

			if (!pass)
			{
				float xa, xb;
				compute_least_squares_endpoints_a(16, trial_alpha_selectors, (const vec4F * uniform)&g_bc7_weights2x[0], &xa, &xb, pPixels);
				const float xl = min(xa, xb);
				const float xh = max(xa, xb);

				uint32_t new_lo_a = (int)round(clampf(xl, 0, 255));
				uint32_t new_hi_a = (int)round(clampf(xh, 0, 255));

				if ((new_lo_a == lo_a) && (new_hi_a == hi_a))
					break;

				lo_a = new_lo_a;
				hi_a = new_hi_a;
			}
		}

		if (pComp_params->m_uber_level > 0)
		{
			const uniform int D = min((int)pComp_params->m_uber_level, 3);
			for (uniform int ld = -D; ld <= D; ld++)
			{
				for (uniform int hd = -D; hd <= D; hd++)
				{
					lo_a = clamp((int)best_alpha_lo + ld, 0, 255);
					hi_a = clamp((int)best_alpha_hi + hd, 0, 255);
					
					int32_t vals[4];
					vals[0] = lo_a;
					vals[3] = hi_a;

					const uniform int32_t w_s1 = 21, w_s2 = 43;
					vals[1] = (vals[0] * (64 - w_s1) + vals[3] * w_s1 + 32) >> 6;
					vals[2] = (vals[0] * (64 - w_s2) + vals[3] * w_s2 + 32) >> 6;

					int trial_alpha_selectors[16];
					int32_t trial_alpha_values[16];

					int32_t trial_alpha_err = 0;
					for (uniform uint32_t i = 0; i < 16; i++)
					{
						const int32_t a = pPixels[i].m_c[3];

						int s = 0;
						int32_t bv = vals[0];
						int32_t be = iabs32(a - bv);
						int e = iabs32(a - vals[1]); if (e < be) { be = e; s = 1; bv = vals[1]; }
						e = iabs32(a - vals[2]); if (e < be) { be = e; s = 2; bv = vals[2]; }
						e = iabs32(a - vals[3]); if (e < be) { be = e; s = 3; bv = vals[3]; }

						trial_alpha_selectors[i] = s;
						trial_alpha_values[i] = bv;
								
						trial_alpha_err += be * be;
					}

					if (trial_alpha_err < mode5_alpha_err)
					{
						mode5_alpha_err = trial_alpha_err;
						best_alpha_lo = lo_a;
						best_alpha_hi = hi_a;
						for (uniform uint32_t i = 0; i < 16; i++)
						{
							pOpt_results5->m_alpha_selectors[i] = trial_alpha_selectors[i];
							best_alpha_values[i] = trial_alpha_values[i];
						}
					}
				
				} // hd

			} // ld
		}
	}

	color_cell_compressor_results results5;
	results5.m_pSelectors = pOpt_results5->m_selectors;

	int selectors_temp[16];
	results5.m_pSelectors_temp = selectors_temp;

	*pMode5_err = color_cell_compression(5, pParams, &results5, pComp_params, 16, pPixels, best_alpha_values, true);
	assert(*pMode5_err == results5.m_best_overall_err);
	// If we don't optimize for color * alpha, and if we're not perceptual or we're not rotated, then color and alpha errors are independent. 
	// If we optimize for color * alpha, or if we're perceptual with rotation, then color_cell_compression must include alpha error.
	if (!pParams->m_optimize_for_color_times_alpha && (!pParams->m_perceptual || pParams->m_rotation == 0))
		*pMode5_err += (float)mode5_alpha_err * pParams->m_weights[3];

	pOpt_results5->m_low[0] = results5.m_low_endpoint;
	pOpt_results5->m_high[0] = results5.m_high_endpoint;
	pOpt_results5->m_low[0].m_c[3] = best_alpha_lo;
	pOpt_results5->m_high[0].m_c[3] = best_alpha_hi;

	pOpt_results5->m_mode = 5;
	pOpt_results5->m_index_selector = 0;
	pOpt_results5->m_rotation = 0;
	pOpt_results5->m_partition = 0;
}

static float handle_alpha_block(void *varying pBlock, const varying color_quad_i *uniform pPixels, const uniform bc7e_compress_block_params *uniform pComp_params, uniform color_cell_compressor_params *uniform pParams, uint32_t lo_a, uint32_t hi_a)
{
	assert(pParams->m_perceptual == pComp_params->m_perceptual);
	DEBUG_SPAM(pParams, "handle_alpha_block\n");

	bc7_optimization_results opt_results;
	
	float best_err = FLT_MAX;
		
	// Mode 4
	if (pComp_params->m_alpha_settings.m_use_mode4)
	{
		uniform color_cell_compressor_params params4 = *pParams;

		const uniform int num_rotations = (!pComp_params->m_alpha_settings.m_use_mode4_rotation) ? 1 : 4;
		for (uniform uint32_t rotation = 0; rotation < num_rotations; rotation++)
		{
			if ((pComp_params->m_mode4_rotation_mask & (1 << rotation)) == 0)
				continue;

			params4.m_rotation = rotation;
			memcpy(params4.m_weights, pParams->m_weights, sizeof(params4.m_weights));
			if (rotation)
				swapf(&params4.m_weights[rotation - 1], &params4.m_weights[3]);
							
			color_quad_i rot_pixels[16];
			const varying color_quad_i *uniform pTrial_pixels = pPixels;
			uint32_t trial_lo_a = lo_a, trial_hi_a = hi_a;
			if (rotation)
			{
				trial_lo_a = 255;
				trial_hi_a = 0;

				for (uniform uint32_t i = 0; i < 16; i++)
				{
					color_quad_i c = pPixels[i];
					swapi(&c.m_c[3], &c.m_c[rotation - 1]);
					rot_pixels[i] = c;

					trial_lo_a = minimumu(trial_lo_a, c.m_c[3]);
					trial_hi_a = maximumu(trial_hi_a, c.m_c[3]);
				}

				pTrial_pixels = rot_pixels;
			}

			bc7_optimization_results trial_opt_results4;

			float trial_mode4_err = best_err;

			handle_alpha_block_mode4(pTrial_pixels, pComp_params, &params4, trial_lo_a, trial_hi_a, &trial_opt_results4, &trial_mode4_err);
			DEBUG_SPAM(pParams, "  mode 4 error %\n", trial_mode4_err);

			if (trial_mode4_err < best_err)
			{
				DEBUG_SPAM(pParams, "  *** beats old best %\n", best_err);
				best_err = trial_mode4_err;

				opt_results.m_mode = 4;
				opt_results.m_index_selector = trial_opt_results4.m_index_selector;
				opt_results.m_rotation = rotation;
				opt_results.m_partition = 0;

				opt_results.m_low[0] = trial_opt_results4.m_low[0];
				opt_results.m_high[0] = trial_opt_results4.m_high[0];

				for (uniform uint32_t i = 0; i < 16; i++)
					opt_results.m_selectors[i] = trial_opt_results4.m_selectors[i];
				
				for (uniform uint32_t i = 0; i < 16; i++)
					opt_results.m_alpha_selectors[i] = trial_opt_results4.m_alpha_selectors[i];
			}
		} // rotation
	}
	
	// Mode 6
	if (pComp_params->m_alpha_settings.m_use_mode6)
	{
		DEBUG_SPAM(pParams, " handle_alpha_block (mode 6)\n");
		uniform color_cell_compressor_params params6 = *pParams;

		params6.m_weights[0] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[0];
		params6.m_weights[1] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[1];
		params6.m_weights[2] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[2];
		params6.m_weights[3] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[3];

		color_cell_compressor_results results6;
		
		params6.m_pSelector_weights = g_bc7_weights4;
		params6.m_pSelector_weightsx = (const vec4F *)&g_bc7_weights4x[0];
		params6.m_num_selector_weights = 16;

		params6.m_comp_bits = 7;
		params6.m_has_pbits = true;
		params6.m_endpoints_share_pbit = false;
		params6.m_has_alpha = true;
				
		int selectors[16];
		results6.m_pSelectors = selectors;

		int selectors_temp[16];
		results6.m_pSelectors_temp = selectors_temp;
				
		const float mode6_err = color_cell_compression(6, &params6, &results6, pComp_params, 16, pPixels, NULL, true);
		assert(mode6_err == results6.m_best_overall_err);
		DEBUG_SPAM(pParams, "  modde 6 error %\n", mode6_err);

		if (mode6_err < best_err)
		{
			DEBUG_SPAM(pParams, "  *** beats old best %\n", best_err);
			best_err = mode6_err;

			opt_results.m_mode = 6;
			opt_results.m_index_selector = 0;
			opt_results.m_rotation = 0;
			opt_results.m_partition = 0;

			opt_results.m_low[0] = results6.m_low_endpoint;
			opt_results.m_high[0] = results6.m_high_endpoint;

			opt_results.m_pbits[0][0] = results6.m_pbits[0];
			opt_results.m_pbits[0][1] = results6.m_pbits[1];

			for (uniform int i = 0; i < 16; i++)
				opt_results.m_selectors[i] = selectors[i];
		}
	}

	// Mode 5
	if (pComp_params->m_alpha_settings.m_use_mode5)
	{
		uniform color_cell_compressor_params params5 = *pParams;

		const uniform int num_rotations = (!pComp_params->m_alpha_settings.m_use_mode5_rotation) ? 1 : 4;
		for (uniform uint32_t rotation = 0; rotation < num_rotations; rotation++)
		{
			if ((pComp_params->m_mode5_rotation_mask & (1 << rotation)) == 0)
				continue;

			params5.m_rotation = rotation;
			memcpy(params5.m_weights, pParams->m_weights, sizeof(params5.m_weights));
			if (rotation)
				swapf(&params5.m_weights[rotation - 1], &params5.m_weights[3]);

			color_quad_i rot_pixels[16];
			const varying color_quad_i *uniform pTrial_pixels = pPixels;
			uint32_t trial_lo_a = lo_a, trial_hi_a = hi_a;
			if (rotation)
			{
				trial_lo_a = 255;
				trial_hi_a = 0;

				for (uniform uint32_t i = 0; i < 16; i++)
				{
					color_quad_i c = pPixels[i];
					swapi(&c.m_c[3], &c.m_c[rotation - 1]);
					rot_pixels[i] = c;

					trial_lo_a = minimumu(trial_lo_a, c.m_c[3]);
					trial_hi_a = maximumu(trial_hi_a, c.m_c[3]);
				}

				pTrial_pixels = rot_pixels;
			}

			bc7_optimization_results trial_opt_results5;

			float trial_mode5_err = best_err;

			handle_alpha_block_mode5(pTrial_pixels, pComp_params, &params5, trial_lo_a, trial_hi_a, &trial_opt_results5, &trial_mode5_err);
			DEBUG_SPAM(pParams, "  mode 5 error %\n", trial_mode5_err);

			if (trial_mode5_err < best_err)
			{
				DEBUG_SPAM(pParams, "  *** beats old best %\n", best_err);
				best_err = trial_mode5_err;

				opt_results = trial_opt_results5;
				opt_results.m_rotation = rotation;
			}
		} // rotation
	}

	// Mode 7
	if (pComp_params->m_alpha_settings.m_use_mode7)
	{
		solution solutions[BC7E_MAX_PARTITIONS7];
		uniform uint32_t num_solutions = estimate_partition_list(7, pPixels, pComp_params, solutions, pComp_params->m_alpha_settings.m_max_mode7_partitions_to_try);

		uniform color_cell_compressor_params params7 = *pParams;
		
		params7.m_weights[0] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[0];
		params7.m_weights[1] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[1];
		params7.m_weights[2] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[2];
		params7.m_weights[3] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[3];
		
		params7.m_pSelector_weights = g_bc7_weights2;
		params7.m_pSelector_weightsx = (const vec4F *)&g_bc7_weights2x[0];
		params7.m_num_selector_weights = 4;

		params7.m_comp_bits = 5;
		params7.m_has_pbits = true;
		params7.m_endpoints_share_pbit = false;
				
		params7.m_has_alpha = true;

		int selectors_temp[16];

		const uniform bool disable_faster_part_selection = false;
		const uniform bool refine_while_choosing = (disable_faster_part_selection || num_solutions <= 2);

		for (uniform uint32_t solution_index = 0; solution_index < num_solutions; solution_index++)
		{
			const uint32_t trial_partition = solutions[solution_index].m_index;
			assert(trial_partition < 64);
			DEBUG_SPAM(pParams, " handle_alpha_block (mode 7, partition %, err est %)\n", trial_partition, solutions[solution_index].m_err);

			const int *pPartition = &g_bc7_partition2[trial_partition * 16];

			color_quad_i subset_colors[2][16];

			uint32_t subset_total_colors7[2];
			subset_total_colors7[0] = 0;
			subset_total_colors7[1] = 0;
			 
			int subset_pixel_index7[2][16];
			int subset_selectors7[2][16];
			color_cell_compressor_results subset_results7[2];

			for (uniform uint32_t idx = 0; idx < 16; idx++)
			{
#pragma ignore warning(perf)
				const uint32_t p = pPartition[idx];
				assert(p < 2);

#pragma ignore warning(perf)
				subset_colors[p][subset_total_colors7[p]] = pPixels[idx];
#pragma ignore warning(perf)
				subset_pixel_index7[p][subset_total_colors7[p]] = idx;
#pragma ignore warning(perf)
				subset_total_colors7[p]++;
			}

			float trial_err = 0.0f;
			for (uniform uint32_t subset = 0; subset < 2; subset++)
			{
				varying color_cell_compressor_results *uniform pResults = &subset_results7[subset];

				pResults->m_pSelectors = &subset_selectors7[subset][0];
				pResults->m_pSelectors_temp = selectors_temp;

				const float err = color_cell_compression(7, &params7, pResults, pComp_params, subset_total_colors7[subset], &subset_colors[subset][0], NULL, refine_while_choosing);
				assert(err == pResults->m_best_overall_err);

				trial_err += err;
				if (trial_err > best_err)
					break;
			} // subset
			DEBUG_SPAM(pParams, "  mode 7 error >= %\n", trial_err);

			if (trial_err < best_err)
			{
				DEBUG_SPAM(pParams, "  *** beats old best %\n", best_err);
				best_err = trial_err;
										
				opt_results.m_mode = 7;
				opt_results.m_index_selector = 0;
				opt_results.m_rotation = 0;
				opt_results.m_partition = trial_partition;

				for (uniform uint32_t subset = 0; subset < 2; subset++)
				{
					for (uniform uint32_t i = 0; i < subset_total_colors7[subset]; i++)
					{
						const uint32_t pixel_index = subset_pixel_index7[subset][i];

#pragma ignore warning(perf)
						opt_results.m_selectors[pixel_index] = subset_selectors7[subset][i];
					}

					opt_results.m_low[subset] = subset_results7[subset].m_low_endpoint;
					opt_results.m_high[subset] = subset_results7[subset].m_high_endpoint;

					opt_results.m_pbits[subset][0] = subset_results7[subset].m_pbits[0];
					opt_results.m_pbits[subset][1] = subset_results7[subset].m_pbits[1];
				}
			}

		} // solution_index

		if ((!refine_while_choosing) && (opt_results.m_mode == 7))
		{
			const uint32_t trial_partition = opt_results.m_partition;
			assert(trial_partition < 64);

			const int *pPartition = &g_bc7_partition2[trial_partition * 16];

			color_quad_i subset_colors[2][16];

			uint32_t subset_total_colors7[2];
			subset_total_colors7[0] = 0;
			subset_total_colors7[1] = 0;
			 
			int subset_pixel_index7[2][16];
			int subset_selectors7[2][16];
			color_cell_compressor_results subset_results7[2];

			for (uniform uint32_t idx = 0; idx < 16; idx++)
			{
#pragma ignore warning(perf)
				const uint32_t p = pPartition[idx];
				assert(p < 2);

#pragma ignore warning(perf)
				subset_colors[p][subset_total_colors7[p]] = pPixels[idx];
#pragma ignore warning(perf)
				subset_pixel_index7[p][subset_total_colors7[p]] = idx;
#pragma ignore warning(perf)
				subset_total_colors7[p]++;
			}

			float trial_err = 0;
			for (uniform uint32_t subset = 0; subset < 2; subset++)
			{
				varying color_cell_compressor_results *uniform pResults = &subset_results7[subset];

				pResults->m_pSelectors = &subset_selectors7[subset][0];
				pResults->m_pSelectors_temp = selectors_temp;

				const float err = color_cell_compression(7, &params7, pResults, pComp_params, subset_total_colors7[subset], &subset_colors[subset][0], NULL, true);
				assert(err == pResults->m_best_overall_err);

				trial_err += err;
				if (trial_err > best_err)
					break;
			} // subset

			if (trial_err < best_err)
			{
				best_err = trial_err;
										
				for (uniform uint32_t subset = 0; subset < 2; subset++)
				{
					for (uniform uint32_t i = 0; i < subset_total_colors7[subset]; i++)
					{
						const uint32_t pixel_index = subset_pixel_index7[subset][i];

#pragma ignore warning(perf)
						opt_results.m_selectors[pixel_index] = subset_selectors7[subset][i];
					}

					opt_results.m_low[subset] = subset_results7[subset].m_low_endpoint;
					opt_results.m_high[subset] = subset_results7[subset].m_high_endpoint;

					opt_results.m_pbits[subset][0] = subset_results7[subset].m_pbits[0];
					opt_results.m_pbits[subset][1] = subset_results7[subset].m_pbits[1];
				}
			}
		}
	}

	DEBUG_SPAM(pParams, "handle_alpha_block finished\n");
	DEBUG_SPAM_CODE(pParams, spam_bc7_block(&opt_results));
	encode_bc7_block(pBlock, &opt_results);
	return best_err;
}

static float handle_opaque_block(void *varying pBlock, const varying color_quad_i *uniform pPixels, const uniform bc7e_compress_block_params *uniform pComp_params, uniform color_cell_compressor_params *uniform pParams)
{
	DEBUG_SPAM(pParams, "handle_opaque_block\n");

	int selectors_temp[16];
		
	bc7_optimization_results opt_results;
		
	float best_err = FLT_MAX;

	// Mode 6
	if (pComp_params->m_opaque_settings.m_use_mode[6])
	{
		DEBUG_SPAM(pParams, " handle_opaque_block (mode 6)\n");
		pParams->m_pSelector_weights = g_bc7_weights4;
		pParams->m_pSelector_weightsx = (const vec4F * uniform)&g_bc7_weights4x[0];
		pParams->m_num_selector_weights = 16;

		pParams->m_comp_bits = 7;
		pParams->m_has_pbits = true;
		pParams->m_endpoints_share_pbit = false;

		assert(pParams->m_perceptual == pComp_params->m_perceptual);
				
		color_cell_compressor_results results6;						
		results6.m_pSelectors = opt_results.m_selectors;
		results6.m_pSelectors_temp = selectors_temp;

		best_err = color_cell_compression(6, pParams, &results6, pComp_params, 16, pPixels, NULL, true);
		DEBUG_SPAM(pParams, "  *** best err %\n", best_err);
						
		opt_results.m_mode = 6;
		opt_results.m_index_selector = 0;
		opt_results.m_rotation = 0;
		opt_results.m_partition = 0;

		opt_results.m_low[0] = results6.m_low_endpoint;
		opt_results.m_high[0] = results6.m_high_endpoint;

		opt_results.m_pbits[0][0] = results6.m_pbits[0];
		opt_results.m_pbits[0][1] = results6.m_pbits[1];
	}

	solution solutions2[BC7E_MAX_PARTITIONS3];
	uniform uint32_t num_solutions2 = 0;
	if (pComp_params->m_opaque_settings.m_use_mode[1] || pComp_params->m_opaque_settings.m_use_mode[3])
	{
		if (pComp_params->m_opaque_settings.m_max_mode13_partitions_to_try == 1)
		{
			solutions2[0].m_index = estimate_partition(1, pPixels, pComp_params);
			num_solutions2 = 1;
		}
		else
		{
			num_solutions2 = estimate_partition_list(1, pPixels, pComp_params, solutions2, pComp_params->m_opaque_settings.m_max_mode13_partitions_to_try);
		}
	}
		
	const uniform bool disable_faster_part_selection = false;
	const uniform bool refine_while_choosing2 = (disable_faster_part_selection || num_solutions2 <= 2);
								
	// Mode 1
	if (pComp_params->m_opaque_settings.m_use_mode[1])
	{
		pParams->m_pSelector_weights = g_bc7_weights3;
		pParams->m_pSelector_weightsx = (const vec4F *)&g_bc7_weights3x[0];
		pParams->m_num_selector_weights = 8;

		pParams->m_comp_bits = 6;
		pParams->m_has_pbits = true;
		pParams->m_endpoints_share_pbit = true;

		assert(pParams->m_perceptual == pComp_params->m_perceptual);

		for (uniform uint32_t solution_index = 0; solution_index < num_solutions2; solution_index++)
		{
			const uint32_t trial_partition = solutions2[solution_index].m_index;
			assert(trial_partition < 64);
			DEBUG_SPAM(pParams, " handle_opaque_block (mode 1, partition %)\n", trial_partition);

			const int *pPartition = &g_bc7_partition2[trial_partition * 16];
						
			color_quad_i subset_colors[2][16];

			uint32_t subset_total_colors1[2];
			subset_total_colors1[0] = 0;
			subset_total_colors1[1] = 0;
				
			int subset_pixel_index1[2][16];
			int subset_selectors1[2][16];
			color_cell_compressor_results subset_results1[2];

			for (uniform uint32_t idx = 0; idx < 16; idx++)
			{
#pragma ignore warning(perf)
				const uint32_t p = pPartition[idx];
				assert(p < 2);

#pragma ignore warning(perf)
				subset_colors[p][subset_total_colors1[p]] = pPixels[idx];
#pragma ignore warning(perf)
				subset_pixel_index1[p][subset_total_colors1[p]] = idx;
#pragma ignore warning(perf)
				subset_total_colors1[p]++;
			}
								
			float trial_err = 0.0f;
			for (uniform uint32_t subset = 0; subset < 2; subset++)
			{
				varying color_cell_compressor_results *uniform pResults = &subset_results1[subset];

				pResults->m_pSelectors = &subset_selectors1[subset][0];
				pResults->m_pSelectors_temp = selectors_temp;

				const float err = color_cell_compression(1, pParams, pResults, pComp_params, subset_total_colors1[subset], &subset_colors[subset][0], NULL, refine_while_choosing2);
				assert(err == pResults->m_best_overall_err);

				trial_err += err;
				if (trial_err > best_err)
					break;
					
			} // subset
			DEBUG_SPAM(pParams, "  mode 1 error %\n", trial_err);

			if (trial_err < best_err)
			{
				DEBUG_SPAM(pParams, "  *** beats old best %\n", best_err);
				best_err = trial_err;

				opt_results.m_mode = 1;
				opt_results.m_index_selector = 0;
				opt_results.m_rotation = 0;
				opt_results.m_partition = trial_partition;

				for (uniform uint32_t subset = 0; subset < 2; subset++)
				{
					for (uniform uint32_t i = 0; i < subset_total_colors1[subset]; i++)
					{
						const uint32_t pixel_index = subset_pixel_index1[subset][i];

#pragma ignore warning(perf)
						opt_results.m_selectors[pixel_index] = subset_selectors1[subset][i];
					}

					opt_results.m_low[subset] = subset_results1[subset].m_low_endpoint;
					opt_results.m_high[subset] = subset_results1[subset].m_high_endpoint;

					opt_results.m_pbits[subset][0] = subset_results1[subset].m_pbits[0];
				}
			}
		}

		if ((!refine_while_choosing2) && (opt_results.m_mode == 1))
		{
			const uint32_t trial_partition = opt_results.m_partition;
			assert(trial_partition < 64);
			DEBUG_SPAM(pParams, "  handle_opaque_block (mode 1, refine partition %)\n", trial_partition);

			const int *pPartition = &g_bc7_partition2[trial_partition * 16];
						
			color_quad_i subset_colors[2][16];

			uint32_t subset_total_colors1[2];
			subset_total_colors1[0] = 0;
			subset_total_colors1[1] = 0;
				
			int subset_pixel_index1[2][16];
			int subset_selectors1[2][16];
			color_cell_compressor_results subset_results1[2];

			for (uniform uint32_t idx = 0; idx < 16; idx++)
			{
#pragma ignore warning(perf)
				const uint32_t p = pPartition[idx];
				assert(p < 2);

#pragma ignore warning(perf)
				subset_colors[p][subset_total_colors1[p]] = pPixels[idx];
#pragma ignore warning(perf)
				subset_pixel_index1[p][subset_total_colors1[p]] = idx;
#pragma ignore warning(perf)
				subset_total_colors1[p]++;
			}
								
			float trial_err = 0.0f;
			for (uniform uint32_t subset = 0; subset < 2; subset++)
			{
				varying color_cell_compressor_results *uniform pResults = &subset_results1[subset];

				pResults->m_pSelectors = &subset_selectors1[subset][0];
				pResults->m_pSelectors_temp = selectors_temp;

				const float err = color_cell_compression(1, pParams, pResults, pComp_params, subset_total_colors1[subset], &subset_colors[subset][0], NULL, true);
				assert(err == pResults->m_best_overall_err);

				trial_err += err;
				if (trial_err > best_err)
					break;
					
			} // subset
			DEBUG_SPAM(pParams, "  mode 1 refinement error %\n", trial_err);

			if (trial_err < best_err)
			{
				DEBUG_SPAM(pParams, "  *** beats old best %\n", best_err);
				best_err = trial_err;

				for (uniform uint32_t subset = 0; subset < 2; subset++)
				{
					for (uniform uint32_t i = 0; i < subset_total_colors1[subset]; i++)
					{
						const uint32_t pixel_index = subset_pixel_index1[subset][i];
#pragma ignore warning(perf)
						opt_results.m_selectors[pixel_index] = subset_selectors1[subset][i];
					}

					opt_results.m_low[subset] = subset_results1[subset].m_low_endpoint;
					opt_results.m_high[subset] = subset_results1[subset].m_high_endpoint;

					opt_results.m_pbits[subset][0] = subset_results1[subset].m_pbits[0];
				}
			}
		}
	}
		
	// Mode 0
	if (pComp_params->m_opaque_settings.m_use_mode[0])
	{
		solution solutions3[BC7E_MAX_PARTITIONS0];
		uniform uint32_t num_solutions3 = 0;
		if (pComp_params->m_opaque_settings.m_max_mode0_partitions_to_try == 1)
		{
			solutions3[0].m_index = estimate_partition(0, pPixels, pComp_params);
			num_solutions3 = 1;
		}
		else
		{
			num_solutions3 = estimate_partition_list(0, pPixels, pComp_params, solutions3, pComp_params->m_opaque_settings.m_max_mode0_partitions_to_try);
		}

		pParams->m_pSelector_weights = g_bc7_weights3;
		pParams->m_pSelector_weightsx = (const vec4F *)&g_bc7_weights3x[0];
		pParams->m_num_selector_weights = 8;

		pParams->m_comp_bits = 4;
		pParams->m_has_pbits = true;
		pParams->m_endpoints_share_pbit = false;

		assert(pParams->m_perceptual == pComp_params->m_perceptual);
				
		for (uniform uint32_t solution_index = 0; solution_index < num_solutions3; solution_index++)
		{
			const uint32_t best_partition0 = solutions3[solution_index].m_index;
			DEBUG_SPAM(pParams, " handle_opaque_block (mode 1, partition %)\n", best_partition0);

			const int *pPartition = &g_bc7_partition3[best_partition0 * 16];

			color_quad_i subset_colors[3][16];
						
			uint32_t subset_total_colors0[3];
			subset_total_colors0[0] = 0;
			subset_total_colors0[1] = 0;
			subset_total_colors0[2] = 0;

			int subset_pixel_index0[3][16];
						
			for (uniform uint32_t idx = 0; idx < 16; idx++)
			{
#pragma ignore warning(perf)
				const uint32_t p = pPartition[idx];

#pragma ignore warning(perf)
				subset_colors[p][subset_total_colors0[p]] = pPixels[idx];
#pragma ignore warning(perf)				
				subset_pixel_index0[p][subset_total_colors0[p]] = idx;
#pragma ignore warning(perf)
				subset_total_colors0[p]++;
			}
									
			color_cell_compressor_results subset_results0[3];
			int subset_selectors0[3][16];			

			float mode0_err = 0.0f;
			for (uniform uint32_t subset = 0; subset < 3; subset++)
			{
				varying color_cell_compressor_results *uniform pResults = &subset_results0[subset];

				pResults->m_pSelectors = &subset_selectors0[subset][0];
				pResults->m_pSelectors_temp = selectors_temp;

				const float err = color_cell_compression(0, pParams, pResults, pComp_params, subset_total_colors0[subset], &subset_colors[subset][0], NULL, true);
				assert(err == pResults->m_best_overall_err);

				mode0_err += err;
				if (mode0_err > best_err)
					break;
			} // subset
			DEBUG_SPAM(pParams, "  mode 0 error %\n", mode0_err);

			if (mode0_err < best_err)
			{
				DEBUG_SPAM(pParams, "  *** beats old best %\n", best_err);
				best_err = mode0_err;

				opt_results.m_mode = 0;
				opt_results.m_index_selector = 0;
				opt_results.m_rotation = 0;
				opt_results.m_partition = best_partition0;

				for (uniform uint32_t subset = 0; subset < 3; subset++)
				{
					for (uniform uint32_t i = 0; i < subset_total_colors0[subset]; i++)
					{
						const uint32_t pixel_index = subset_pixel_index0[subset][i];

#pragma ignore warning(perf)
						opt_results.m_selectors[pixel_index] = subset_selectors0[subset][i];
					}

					opt_results.m_low[subset] = subset_results0[subset].m_low_endpoint;
					opt_results.m_high[subset] = subset_results0[subset].m_high_endpoint;

					opt_results.m_pbits[subset][0] = subset_results0[subset].m_pbits[0];
					opt_results.m_pbits[subset][1] = subset_results0[subset].m_pbits[1];
				}
			}
		}
	}
		
	// Mode 3
	if (pComp_params->m_opaque_settings.m_use_mode[3])
	{
		pParams->m_pSelector_weights = g_bc7_weights2;
		pParams->m_pSelector_weightsx = (const vec4F *)&g_bc7_weights2x[0];
		pParams->m_num_selector_weights = 4;

		pParams->m_comp_bits = 7;
		pParams->m_has_pbits = true;
		pParams->m_endpoints_share_pbit = false;

		assert(pParams->m_perceptual == pComp_params->m_perceptual);

		for (uniform uint32_t solution_index = 0; solution_index < num_solutions2; solution_index++)
		{
			const uint32_t trial_partition = solutions2[solution_index].m_index;
			assert(trial_partition < 64);
			DEBUG_SPAM(pParams, " handle_opaque_block (mode 3, partition %)\n", trial_partition);

			const int *pPartition = &g_bc7_partition2[trial_partition * 16];

			color_quad_i subset_colors[2][16];

			uint32_t subset_total_colors3[2];
			subset_total_colors3[0] = 0;
			subset_total_colors3[1] = 0;
			 
			int subset_pixel_index3[2][16];
			int subset_selectors3[2][16];
			color_cell_compressor_results subset_results3[2];

			for (uniform uint32_t idx = 0; idx < 16; idx++)
			{
#pragma ignore warning(perf)
				const uint32_t p = pPartition[idx];
				assert(p < 2);

#pragma ignore warning(perf)
				subset_colors[p][subset_total_colors3[p]] = pPixels[idx];

#pragma ignore warning(perf)
				subset_pixel_index3[p][subset_total_colors3[p]] = idx;

#pragma ignore warning(perf)
				subset_total_colors3[p]++;
			}

			float trial_err = 0.0f;
			for (uniform uint32_t subset = 0; subset < 2; subset++)
			{
				varying color_cell_compressor_results *uniform pResults = &subset_results3[subset];

				pResults->m_pSelectors = &subset_selectors3[subset][0];
				pResults->m_pSelectors_temp = selectors_temp;

				const float err = color_cell_compression(3, pParams, pResults, pComp_params, subset_total_colors3[subset], &subset_colors[subset][0], NULL, refine_while_choosing2);
				assert(err == pResults->m_best_overall_err);

				trial_err += err;
				if (trial_err > best_err)
					break;
			} // subset
			DEBUG_SPAM(pParams, "  mode 3 error %\n", trial_err);

			if (trial_err < best_err)
			{
				DEBUG_SPAM(pParams, "  *** beats old best %\n", best_err);
				best_err = trial_err;
										
				opt_results.m_mode = 3;
				opt_results.m_index_selector = 0;
				opt_results.m_rotation = 0;
				opt_results.m_partition = trial_partition;

				for (uniform uint32_t subset = 0; subset < 2; subset++)
				{
					for (uniform uint32_t i = 0; i < subset_total_colors3[subset]; i++)
					{
						const uint32_t pixel_index = subset_pixel_index3[subset][i];

#pragma ignore warning(perf)
						opt_results.m_selectors[pixel_index] = subset_selectors3[subset][i];
					}

					opt_results.m_low[subset] = subset_results3[subset].m_low_endpoint;
					opt_results.m_high[subset] = subset_results3[subset].m_high_endpoint;

					opt_results.m_pbits[subset][0] = subset_results3[subset].m_pbits[0];
					opt_results.m_pbits[subset][1] = subset_results3[subset].m_pbits[1];
				}
			}

		} // solution_index

		if ((!refine_while_choosing2) && (opt_results.m_mode == 3))
		{
			const uint32_t trial_partition = opt_results.m_partition;
			assert(trial_partition < 64);
			DEBUG_SPAM(pParams, " handle_opaque_block (mode 3, partition % refinement)\n", trial_partition);

			const int *pPartition = &g_bc7_partition2[trial_partition * 16];

			color_quad_i subset_colors[2][16];

			uint32_t subset_total_colors3[2];
			subset_total_colors3[0] = 0;
			subset_total_colors3[1] = 0;
			 
			int subset_pixel_index3[2][16];
			int subset_selectors3[2][16];
			color_cell_compressor_results subset_results3[2];

			for (uniform uint32_t idx = 0; idx < 16; idx++)
			{
#pragma ignore warning(perf)
				const uint32_t p = pPartition[idx];
				assert(p < 2);

#pragma ignore warning(perf)
				subset_colors[p][subset_total_colors3[p]] = pPixels[idx];

#pragma ignore warning(perf)
				subset_pixel_index3[p][subset_total_colors3[p]] = idx;

#pragma ignore warning(perf)
				subset_total_colors3[p]++;
			}

			float trial_err = 0.0f;
			for (uniform uint32_t subset = 0; subset < 2; subset++)
			{
				varying color_cell_compressor_results *uniform pResults = &subset_results3[subset];

				pResults->m_pSelectors = &subset_selectors3[subset][0];
				pResults->m_pSelectors_temp = selectors_temp;

				const float err = color_cell_compression(3, pParams, pResults, pComp_params, subset_total_colors3[subset], &subset_colors[subset][0], NULL, true);
				assert(err == pResults->m_best_overall_err);

				trial_err += err;
				if (trial_err > best_err)
					break;
			} // subset
			DEBUG_SPAM(pParams, "  mode 3 refinement error %\n", trial_err);

			if (trial_err < best_err)
			{
				DEBUG_SPAM(pParams, "  *** beats old best %\n", best_err);
				best_err = trial_err;
										
				for (uniform uint32_t subset = 0; subset < 2; subset++)
				{
					for (uniform uint32_t i = 0; i < subset_total_colors3[subset]; i++)
					{
						const uint32_t pixel_index = subset_pixel_index3[subset][i];

#pragma ignore warning(perf)
						opt_results.m_selectors[pixel_index] = subset_selectors3[subset][i];
					}

					opt_results.m_low[subset] = subset_results3[subset].m_low_endpoint;
					opt_results.m_high[subset] = subset_results3[subset].m_high_endpoint;

					opt_results.m_pbits[subset][0] = subset_results3[subset].m_pbits[0];
					opt_results.m_pbits[subset][1] = subset_results3[subset].m_pbits[1];
				}
			}
		}
	}

	// Mode 5
	if (pComp_params->m_opaque_settings.m_use_mode[5])
	{
		uniform color_cell_compressor_params params5 = *pParams;

		for (uniform uint32_t rotation = 0; rotation < 4; rotation++)
		{
			if ((pComp_params->m_mode5_rotation_mask & (1 << rotation)) == 0)
				continue;

			params5.m_rotation = rotation;
			memcpy(params5.m_weights, pParams->m_weights, sizeof(params5.m_weights));
			if (rotation)
				swapf(&params5.m_weights[rotation - 1], &params5.m_weights[3]);

			color_quad_i rot_pixels[16];
			const varying color_quad_i *uniform pTrial_pixels = pPixels;
			uint32_t trial_lo_a = 255, trial_hi_a = 255;
			if (rotation)
			{
				trial_lo_a = 255;
				trial_hi_a = 0;

				for (uniform uint32_t i = 0; i < 16; i++)
				{
					color_quad_i c = pPixels[i];
					swapi(&c.m_c[3], &c.m_c[rotation - 1]);
					rot_pixels[i] = c;

					trial_lo_a = minimumu(trial_lo_a, c.m_c[3]);
					trial_hi_a = maximumu(trial_hi_a, c.m_c[3]);
				}

				pTrial_pixels = rot_pixels;
			}

			bc7_optimization_results trial_opt_results5;

			float trial_mode5_err = 0.0f;

			handle_alpha_block_mode5(pTrial_pixels, pComp_params, &params5, trial_lo_a, trial_hi_a, &trial_opt_results5, &trial_mode5_err);
			DEBUG_SPAM(pParams, "  mode 5 error %\n", trial_mode5_err);

			if (trial_mode5_err < best_err)
			{
				DEBUG_SPAM(pParams, "  *** beats old best %\n", best_err);
				best_err = trial_mode5_err;

				opt_results = trial_opt_results5;
				opt_results.m_rotation = rotation;
			}
		} // rotation
	}

	// Mode 2
	if (pComp_params->m_opaque_settings.m_use_mode[2])
	{
		solution solutions3[BC7E_MAX_PARTITIONS2];
		uniform uint32_t num_solutions3 = 0;
		if (pComp_params->m_opaque_settings.m_max_mode2_partitions_to_try == 1)
		{
			solutions3[0].m_index = estimate_partition(2, pPixels, pComp_params);
			num_solutions3 = 1;
		}
		else
		{
			num_solutions3 = estimate_partition_list(2, pPixels, pComp_params, solutions3, pComp_params->m_opaque_settings.m_max_mode2_partitions_to_try);
		}

		pParams->m_pSelector_weights = g_bc7_weights2;
		pParams->m_pSelector_weightsx = (const vec4F *)&g_bc7_weights2x[0];
		pParams->m_num_selector_weights = 4;

		pParams->m_comp_bits = 5;
		pParams->m_has_pbits = false;
		pParams->m_endpoints_share_pbit = false;

		assert(pParams->m_perceptual == pComp_params->m_perceptual);

		for (uniform uint32_t solution_index = 0; solution_index < num_solutions3; solution_index++)
		{
			const int32_t best_partition2 = solutions3[solution_index].m_index;
			DEBUG_SPAM(pParams, " handle_opaque_block (mode 2, partition %)\n", best_partition2);
						
			uint32_t subset_total_colors2[3];
			subset_total_colors2[0] = 0;
			subset_total_colors2[1] = 0;
			subset_total_colors2[2] = 0;

			int subset_pixel_index2[3][16];
							
			const int *pPartition = &g_bc7_partition3[best_partition2 * 16];

			color_quad_i subset_colors[3][16];

			for (uniform uint32_t idx = 0; idx < 16; idx++)
			{
#pragma ignore warning(perf)
				const uint32_t p = pPartition[idx];

#pragma ignore warning(perf)
				subset_colors[p][subset_total_colors2[p]] = pPixels[idx];

#pragma ignore warning(perf)
				subset_pixel_index2[p][subset_total_colors2[p]] = idx;

#pragma ignore warning(perf)
				subset_total_colors2[p]++;
			}
			
			int subset_selectors2[3][16];
			color_cell_compressor_results subset_results2[3];
						
			float mode2_err = 0.0f;
			for (uniform uint32_t subset = 0; subset < 3; subset++)
			{
				varying color_cell_compressor_results *uniform pResults = &subset_results2[subset];

				pResults->m_pSelectors = &subset_selectors2[subset][0];
				pResults->m_pSelectors_temp = selectors_temp;

				const float err = color_cell_compression(2, pParams, pResults, pComp_params, subset_total_colors2[subset], &subset_colors[subset][0], NULL, true);
				assert(err == pResults->m_best_overall_err);

				mode2_err += err;
				if (mode2_err > best_err)
					break;
			} // subset
			DEBUG_SPAM(pParams, "  mode 2 error %\n", mode2_err);

			if (mode2_err < best_err)
			{
				DEBUG_SPAM(pParams, "  *** beats old best %\n", best_err);
				best_err = mode2_err;

				opt_results.m_mode = 2;
				opt_results.m_index_selector = 0;
				opt_results.m_rotation = 0;
				opt_results.m_partition = best_partition2;

				for (uniform uint32_t subset = 0; subset < 3; subset++)
				{
					for (uniform uint32_t i = 0; i < subset_total_colors2[subset]; i++)
					{
						const uint32_t pixel_index = subset_pixel_index2[subset][i];

#pragma ignore warning(perf)
						opt_results.m_selectors[pixel_index] = subset_selectors2[subset][i];
					}

					opt_results.m_low[subset] = subset_results2[subset].m_low_endpoint;
					opt_results.m_high[subset] = subset_results2[subset].m_high_endpoint;
				}
			}
		}
	}

	// Mode 4
	if (pComp_params->m_opaque_settings.m_use_mode[4])
	{
		uniform color_cell_compressor_params params4 = *pParams;

		for (uniform uint32_t rotation = 0; rotation < 4; rotation++)
		{
			if ((pComp_params->m_mode4_rotation_mask & (1 << rotation)) == 0)
				continue;

			params4.m_rotation = rotation;
			memcpy(params4.m_weights, pParams->m_weights, sizeof(params4.m_weights));
			if (rotation)
				swapf(&params4.m_weights[rotation - 1], &params4.m_weights[3]);
							
			color_quad_i rot_pixels[16];
			const varying color_quad_i *uniform pTrial_pixels = pPixels;
			uint32_t trial_lo_a = 255, trial_hi_a = 255;
			if (rotation)
			{
				trial_lo_a = 255;
				trial_hi_a = 0;

				for (uniform uint32_t i = 0; i < 16; i++)
				{
					color_quad_i c = pPixels[i];
					swapi(&c.m_c[3], &c.m_c[rotation - 1]);
					rot_pixels[i] = c;

					trial_lo_a = minimumu(trial_lo_a, c.m_c[3]);
					trial_hi_a = maximumu(trial_hi_a, c.m_c[3]);
				}

				pTrial_pixels = rot_pixels;
			}

			bc7_optimization_results trial_opt_results4;

			float trial_mode4_err = best_err;

			handle_alpha_block_mode4(pTrial_pixels, pComp_params, &params4, trial_lo_a, trial_hi_a, &trial_opt_results4, &trial_mode4_err);
			DEBUG_SPAM(pParams, "  mode 4 error %\n", trial_mode4_err);

			if (trial_mode4_err < best_err)
			{
				DEBUG_SPAM(pParams, "  *** beats old best %\n", best_err);
				best_err = trial_mode4_err;

				opt_results.m_mode = 4;
				opt_results.m_index_selector = trial_opt_results4.m_index_selector;
				opt_results.m_rotation = rotation;
				opt_results.m_partition = 0;

				opt_results.m_low[0] = trial_opt_results4.m_low[0];
				opt_results.m_high[0] = trial_opt_results4.m_high[0];

				for (uniform uint32_t i = 0; i < 16; i++)
					opt_results.m_selectors[i] = trial_opt_results4.m_selectors[i];
				
				for (uniform uint32_t i = 0; i < 16; i++)
					opt_results.m_alpha_selectors[i] = trial_opt_results4.m_alpha_selectors[i];
			}
		} // rotation
	}
	
	DEBUG_SPAM(pParams, "handle_opaque_block finished\n");
	DEBUG_SPAM_CODE(pParams, spam_bc7_block(&opt_results));
	encode_bc7_block(pBlock, &opt_results);
	return best_err;
}

// all solid color blocks can be 100% perfectly encoded with just mode 5
static float handle_block_solid(void *varying pBlock, uint32_t cr, uint32_t cg, uint32_t cb, uint32_t ca)
{
	#pragma ignore warning(perf)
	uint32_t er = g_bc7_mode_5_optimal_endpoints[cr];
	#pragma ignore warning(perf)
	uint32_t eg = g_bc7_mode_5_optimal_endpoints[cg];
	#pragma ignore warning(perf)
	uint32_t eb = g_bc7_mode_5_optimal_endpoints[cb];
	color_quad_i lp, hp;
	color_quad_i_set(&lp, er & 0xFF, eg & 0xFF, eb & 0xFF, ca);
	color_quad_i_set(&hp, er >> 8, eg >> 8, eb >> 8, ca);

	bc7_optimization_results opt;
	opt.m_mode = 5;
	opt.m_low[0] = lp;
	opt.m_high[0] = hp;
	opt.m_pbits[0][0] = 0;
	opt.m_pbits[0][1] = 0;
	opt.m_index_selector = 0;
	opt.m_rotation = 0;
	opt.m_partition = 0;
	for (uniform int i = 0; i < 16; ++i)
		opt.m_selectors[i] = BC7E_MODE_5_OPTIMAL_INDEX;
	for (uniform int i = 0; i < 16; ++i)
		opt.m_alpha_selectors[i] = 0;
	encode_bc7_block(pBlock, &opt);
	return 0.0f;
}

static float handle_opaque_block_mode6(void *varying pBlock, const varying color_quad_i *uniform pPixels, const uniform bc7e_compress_block_params *uniform pComp_params, uniform color_cell_compressor_params *uniform pParams)
{
	int selectors_temp[16];
		
	bc7_optimization_results opt_results;
		
	// Mode 6
	pParams->m_pSelector_weights = g_bc7_weights4;
	pParams->m_pSelector_weightsx = (const vec4F * uniform)&g_bc7_weights4x[0];
	pParams->m_num_selector_weights = 16;

	pParams->m_comp_bits = 7;
	pParams->m_has_pbits = true;
	pParams->m_endpoints_share_pbit = false;

	assert(pParams->m_perceptual == pComp_params->m_perceptual);

	color_cell_compressor_results results6;
	results6.m_pSelectors = opt_results.m_selectors;
	results6.m_pSelectors_temp = selectors_temp;

	const float best_err = color_cell_compression(6, pParams, &results6, pComp_params, 16, pPixels, NULL, true);

	opt_results.m_mode = 6;
	opt_results.m_index_selector = 0;
	opt_results.m_rotation = 0;
	opt_results.m_partition = 0;

	opt_results.m_low[0] = results6.m_low_endpoint;
	opt_results.m_high[0] = results6.m_high_endpoint;

	opt_results.m_pbits[0][0] = results6.m_pbits[0];
	opt_results.m_pbits[0][1] = results6.m_pbits[1];
		
	encode_bc7_block_mode6(pBlock, &opt_results);
	return best_err;
}

export void bc7e_compress_blocks(uniform uint32_t num_blocks, uniform uint64_t * uniform pBlocks, const uniform uint32_t * uniform pPixelsRGBA, const uniform bc7e_compress_block_params * uniform pComp_params)
{
	if (!g_codec_initialized)
	{
		// Caller has forgotten to initialize the codec, or another thread is still working on that. We can't continue.
		// What do we do here?
		assert(0);

		memset(pBlocks, 0, num_blocks * 16);
		return;
	}

	uniform color_cell_compressor_params params;
	color_cell_compressor_params_clear(&params);

	params.m_debug_spam = pComp_params->m_debugSpam;

	memcpy(params.m_weights, pComp_params->m_weights, sizeof(params.m_weights));
	params.m_perceptual = pComp_params->m_perceptual;
	params.m_optimize_for_color_times_alpha = (pComp_params->m_optimize_for != BC7E_OPTIMIZE_FOR_INDEPENDENT_CHANNELS);
	params.m_frame_buffer_other_value = (pComp_params->m_optimize_for == BC7E_OPTIMIZE_FOR_ALPHA_BLENDING) ? 255.0f : 0.0f;
	if (pComp_params->m_optimize_for == BC7E_OPTIMIZE_FOR_ALPHA_TEST)
	{
		const uniform float H = ceil(pComp_params->m_alpha_test_threshold_max);
		const uniform float L = min(floor(pComp_params->m_alpha_test_threshold_min), H - 1.0f);
		params.m_alpha_scale = 255.0f / (H - L);
		params.m_alpha_bias = L;
	}
	else
	{
		params.m_alpha_scale = 1.0f;
		params.m_alpha_bias = 0.0f;
	}
	if (params.m_optimize_for_color_times_alpha)
	{
		params.m_weights[3] = params.m_weights[0] + params.m_weights[1] + params.m_weights[2];
		if (params.m_perceptual && params.m_optimize_for_color_times_alpha)
		{
			// Need to initialize perceptual constants that depend on m_weights
			const uniform float sqrt_wt_y = sqrt((float)pComp_params->m_weights[0]);
			const uniform float sqrt_wt_cr = sqrt((float)pComp_params->m_weights[1]);
			const uniform float sqrt_wt_cb = sqrt((float)pComp_params->m_weights[2]);
			params.m_sqrt_weights.m_c[0] = sqrt_wt_y;
			params.m_sqrt_weights.m_c[1] = sqrt_wt_cr * k_norm_Cr;
			params.m_sqrt_weights.m_c[2] = sqrt_wt_cb * k_norm_Cb;
			if (pComp_params->m_optimize_for == BC7E_OPTIMIZE_FOR_ALPHA_BLENDING)
			{
				// Minimizing alpha blending error needs the parallelepiped basis vectors
				params.m_dycrcb_r.m_c[0] = (k_Y_R  * 255) * sqrt_wt_y;
				params.m_dycrcb_r.m_c[1] = (k_Cr_R * 255) * sqrt_wt_cr;
				params.m_dycrcb_r.m_c[2] = (k_Cb_R * 255) * sqrt_wt_cb;
				params.m_dycrcb_g.m_c[0] = (k_Y_G  * 255) * sqrt_wt_y;
				params.m_dycrcb_g.m_c[1] = (k_Cr_G * 255) * sqrt_wt_cr;
				params.m_dycrcb_g.m_c[2] = (k_Cb_G * 255) * sqrt_wt_cb;
				params.m_dycrcb_b.m_c[0] = (k_Y_B  * 255) * sqrt_wt_y;
				params.m_dycrcb_b.m_c[1] = (k_Cr_B * 255) * sqrt_wt_cr;
				params.m_dycrcb_b.m_c[2] = (k_Cb_B * 255) * sqrt_wt_cb;
				params.m_dycrcb_mid = 127.5f * sqrt_wt_y;
			}
			else
			{
				// Minimizing color * alpha error needs to ignore the parallelepiped, which leaves the basis vectors zero
				assert(pComp_params->m_optimize_for == BC7E_OPTIMIZE_FOR_COLOR_TIMES_ALPHA);
				assert(params.m_dycrcb_r.m_c[0] == 0.0f);
				assert(params.m_dycrcb_g.m_c[1] == 0.0f);
				assert(params.m_dycrcb_b.m_c[2] == 0.0f);
				assert(params.m_dycrcb_mid == 0.0f);
			}
		}
	}

	assert(pComp_params->m_mode4_rotation_mask != 0);
	assert(pComp_params->m_mode4_index_mask != 0);
	assert(pComp_params->m_mode5_rotation_mask != 0);
	assert(pComp_params->m_uber1_mask != 0);
	
	foreach (block_index = 0 ... num_blocks)
	{
		const uniform color_quad_u8 *varying pSrcPixels = &((const uniform color_quad_u8 *)(pPixelsRGBA))[block_index * 16];

		varying color_quad_i temp_pixels[16];

		int lo_r = 255, hi_r = 0;
		int lo_g = 255, hi_g = 0;
		int lo_b = 255, hi_b = 0;
		float lo_a = 255, hi_a = 0;
		
		for (uniform uint32_t i = 0; i < 16; i++)
		{
#pragma ignore warning(perf)
			color_quad_u8 c = pSrcPixels[i];

			int r = c.m_c[0];
			int g = c.m_c[1];
			int b = c.m_c[2];
			int a = c.m_c[3];
												
			temp_pixels[i].m_c[0] = r;
			temp_pixels[i].m_c[1] = g;
			temp_pixels[i].m_c[2] = b;
			temp_pixels[i].m_c[3] = a;

			lo_r = min(lo_r, r); hi_r = max(hi_r, r);
			lo_g = min(lo_g, g); hi_g = max(hi_g, g);
			lo_b = min(lo_b, b); hi_b = max(hi_b, b);

			float fa = a;

			lo_a = min(lo_a, fa);
			hi_a = max(hi_a, fa);
		}

		bool all_same = lo_r==hi_r && lo_g==hi_g && lo_b==hi_b && lo_a==hi_a;

		uniform uint64_t *varying pBlock = &pBlocks[block_index * 2];

		float best_err;
		cif (all_same)
		{
#pragma ignore warning(perf)
			best_err = handle_block_solid(pBlock, lo_r, lo_g, lo_b, lo_a);
		}
		else
		{
			const bool has_alpha = (lo_a < 255);
			// TODO: alpha block mode 6 only
			cif (has_alpha)
				best_err = handle_alpha_block(pBlock, temp_pixels, pComp_params, &params, (int)lo_a, (int)hi_a);
			else
			{
				if (pComp_params->m_mode6_only)
					best_err = handle_opaque_block_mode6(pBlock, temp_pixels, pComp_params, &params);
				else
					best_err = handle_opaque_block(pBlock, temp_pixels, pComp_params, &params);
			}
		}
		if (pComp_params->m_block_error_metric_results != NULL)
			pComp_params->m_block_error_metric_results[block_index] = best_err;
	}
}

export void bc7e_compress_block_params_init(bc7e_compress_block_params * uniform p, uniform bool perceptual)
{
	p->m_max_partitions_mode[0] = BC7E_MAX_PARTITIONS0;
	p->m_max_partitions_mode[1] = BC7E_MAX_PARTITIONS1;
	p->m_max_partitions_mode[2] = BC7E_MAX_PARTITIONS2;
	p->m_max_partitions_mode[3] = BC7E_MAX_PARTITIONS3;
	p->m_max_partitions_mode[4] = 0;
	p->m_max_partitions_mode[5] = 0;
	p->m_max_partitions_mode[6] = 0;
	p->m_max_partitions_mode[7] = BC7E_MAX_PARTITIONS7;

	p->m_perceptual = perceptual;
	if (perceptual)
	{
		p->m_weights[0] = 128.0f;
		p->m_weights[1] = 64.0f;
		p->m_weights[2] = 16.0f;
		p->m_weights[3] = 256.0f;
	}
	else
	{
		p->m_weights[0] = 1.0f;
		p->m_weights[1] = 1.0f;
		p->m_weights[2] = 1.0f;
		p->m_weights[3] = 1.0f;
	}

	p->m_optimize_for = BC7E_OPTIMIZE_FOR_INDEPENDENT_CHANNELS;
	p->m_pbit_search = false;
	p->m_mode6_only = false;
	p->m_debugSpam = false;
	p->m_refinement_passes = 1;
	p->m_mode4_rotation_mask = 0xF;
	p->m_mode4_index_mask = 3;
	p->m_mode5_rotation_mask = 0xF;
	p->m_uber1_mask = 7;
	for (uniform uint32_t i = 0; i < 7; i++)
		p->m_opaque_settings.m_use_mode[i] = true;
	p->m_opaque_settings.m_max_mode13_partitions_to_try = 1;
	p->m_opaque_settings.m_max_mode0_partitions_to_try = 1;
	p->m_opaque_settings.m_max_mode2_partitions_to_try = 1;
	p->m_alpha_settings.m_use_mode4 = true;
	p->m_alpha_settings.m_use_mode5 = true;
	p->m_alpha_settings.m_use_mode6 = true;
	p->m_alpha_settings.m_use_mode7 = true;
	p->m_alpha_settings.m_use_mode4_rotation = true;
	p->m_alpha_settings.m_use_mode5_rotation = true;
	p->m_alpha_settings.m_max_mode7_partitions_to_try = 1;
	p->m_alpha_settings.m_mode67_error_weight_mul[0] = 1.0f;
	p->m_alpha_settings.m_mode67_error_weight_mul[1] = 1.0f;
	p->m_alpha_settings.m_mode67_error_weight_mul[2] = 1.0f;
	p->m_alpha_settings.m_mode67_error_weight_mul[3] = 1.0f;
	p->m_uber_level = 0;

	p->m_block_error_metric_results = NULL;
}

export void bc7e_compress_block_params_init_slowest(bc7e_compress_block_params * uniform p, uniform bool perceptual)
{
	bc7e_compress_block_params_init(p, perceptual);

	p->m_opaque_settings.m_max_mode13_partitions_to_try = 4;
	p->m_opaque_settings.m_max_mode0_partitions_to_try = 4;
	p->m_opaque_settings.m_max_mode2_partitions_to_try = 4;
	p->m_alpha_settings.m_max_mode7_partitions_to_try = 4;

	p->m_pbit_search = true;
	p->m_uber_level = 4;
}

export void bc7e_compress_block_params_init_veryslow(bc7e_compress_block_params * uniform p, uniform bool perceptual)
{
	bc7e_compress_block_params_init(p, perceptual);

	p->m_opaque_settings.m_max_mode13_partitions_to_try = 2;
	p->m_opaque_settings.m_max_mode0_partitions_to_try = 2;
	p->m_opaque_settings.m_max_mode2_partitions_to_try = 2;
	p->m_alpha_settings.m_max_mode7_partitions_to_try = 2;

	p->m_pbit_search = true;
	p->m_uber_level = 2;
}

export void bc7e_compress_block_params_init_slow(bc7e_compress_block_params * uniform p, uniform bool perceptual)
{
	bc7e_compress_block_params_init(p, perceptual);

	p->m_alpha_settings.m_max_mode7_partitions_to_try = 2;

	p->m_pbit_search = true;
	p->m_uber_level = 0;
}

export void bc7e_compress_block_params_init_basic(bc7e_compress_block_params * uniform p, uniform bool perceptual)
{
	bc7e_compress_block_params_init(p, perceptual);

	if (perceptual)
	{
		p->m_opaque_settings.m_use_mode[0] = false;
		p->m_opaque_settings.m_use_mode[2] = false;
		p->m_opaque_settings.m_use_mode[3] = false;
		p->m_opaque_settings.m_use_mode[4] = false;
		p->m_opaque_settings.m_use_mode[5] = false;
		
		p->m_pbit_search = false;
		p->m_uber_level = 1;
	}
	else
	{
		p->m_max_partitions_mode[1] = 32;
		p->m_max_partitions_mode[2] = 32;
		p->m_max_partitions_mode[3] = 32;
		p->m_max_partitions_mode[7] = 32;

		p->m_opaque_settings.m_use_mode[2] = false;
		
		p->m_pbit_search = false;
		p->m_uber_level = 1;
	}
}

export void bc7e_compress_block_params_init_fast(bc7e_compress_block_params * uniform p, uniform bool perceptual)
{
	bc7e_compress_block_params_init(p, perceptual);

	if (perceptual)
	{
		p->m_opaque_settings.m_use_mode[0] = false;
		p->m_opaque_settings.m_use_mode[2] = false;
		p->m_opaque_settings.m_use_mode[3] = false;
		p->m_opaque_settings.m_use_mode[4] = false;
		p->m_opaque_settings.m_use_mode[5] = false;

		p->m_alpha_settings.m_use_mode5 = false;
	
		p->m_opaque_settings.m_max_mode13_partitions_to_try = 1;
		
		p->m_pbit_search = false;
		p->m_uber_level = 0;
	}		
	else
	{
		p->m_opaque_settings.m_use_mode[0] = false;
		p->m_opaque_settings.m_use_mode[2] = false;
		p->m_opaque_settings.m_use_mode[4] = false;
		p->m_opaque_settings.m_use_mode[5] = false;

		p->m_alpha_settings.m_use_mode5 = false;
	
		p->m_opaque_settings.m_max_mode13_partitions_to_try = 2;
		
		p->m_pbit_search = false;
		p->m_uber_level = 0;
	}
}

export void bc7e_compress_block_params_init_veryfast(bc7e_compress_block_params * uniform p, uniform bool perceptual)
{
	bc7e_compress_block_params_init(p, perceptual);

	if (perceptual)
	{
		p->m_opaque_settings.m_use_mode[0] = false;
		p->m_opaque_settings.m_use_mode[2] = false;
		p->m_opaque_settings.m_use_mode[3] = false;
		p->m_opaque_settings.m_use_mode[4] = false;
		p->m_opaque_settings.m_use_mode[5] = false;

		p->m_alpha_settings.m_use_mode5 = false;
		
		p->m_pbit_search = false;
		p->m_uber_level = 0;
	}
	else
	{
		p->m_opaque_settings.m_use_mode[2] = false;
		p->m_opaque_settings.m_use_mode[4] = false;
		p->m_opaque_settings.m_use_mode[5] = false;

		p->m_alpha_settings.m_use_mode5 = false;
		
		p->m_pbit_search = false;
		p->m_uber_level = 0;
	}
}

export void bc7e_compress_block_params_init_ultrafast(bc7e_compress_block_params * uniform p, uniform bool perceptual)
{
	bc7e_compress_block_params_init(p, perceptual);

	p->m_mode6_only = true;

	p->m_alpha_settings.m_use_mode4 = true;
	p->m_alpha_settings.m_use_mode5 = true;
	p->m_alpha_settings.m_use_mode7 = false;

	p->m_mode4_rotation_mask = 1+4;
	p->m_mode4_index_mask = 3;
	p->m_mode5_rotation_mask = 1;
				
	p->m_pbit_search = false;
	p->m_uber_level = 0;
}
