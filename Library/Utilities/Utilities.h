#ifndef FLUIDSIM3D_UTILITIES_H
#define FLUIDSIM3D_UTILITIES_H

#include <assert.h>

#include <limits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>

#include "tbb/enumerable_thread_specific.h"

#undef max
#undef min

namespace FluidSim3D
{
constexpr double PI = 3.1415926535897932384626433832795;

//
// TBB utilities
//

constexpr int tbbLightGrainSize = 1000;
constexpr int tbbHeavyGrainSize = 100;

template <typename VectorType>
void mergeLocalThreadVectors(VectorType& combinedVector,
                             tbb::enumerable_thread_specific<VectorType>& parallelVector)
{
    size_t vectorSize = 0;

    parallelVector.combine_each([&](const VectorType& localVector)
    {
        vectorSize += localVector.size();
    });

    combinedVector.reserve(combinedVector.size() + vectorSize);

    parallelVector.combine_each([&](const VectorType& localVector)
    {
        combinedVector.insert(combinedVector.end(), localVector.begin(), localVector.end());
    });
}

//
// BFS markers
//

enum class VisitedCellLabels
{
    UNVISITED_CELL,
    VISITED_CELL,
    FINISHED_CELL
};

// Borrowed from https://github.com/sideeffects/WindingNumber/blob/master/SYS_Types.h
#if defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE	__attribute__ ((always_inline)) inline
#elif defined(_MSC_VER)
#define FORCE_INLINE	__forceinline
#else
#define FORCE_INLINE	inline
#endif

using Vec2d = Eigen::Matrix<double, 2, 1>;
using Vec2i = Eigen::Matrix<int, 2, 1>;

using Vec3d = Eigen::Matrix<double, 3, 1>;
using Vec3i = Eigen::Matrix<int, 3, 1>;

template<typename T>
using Vec3t = Eigen::Matrix<T, 3, 1>;

using VecVec3d = std::vector<Vec3d, Eigen::aligned_allocator<Vec3d>>;
using VecVec3i = std::vector<Vec3i, Eigen::aligned_allocator<Vec3i>>;

template<typename T, int N>
using VecXt = Eigen::Matrix<T, N, 1>;

using AlignedBox3d = Eigen::AlignedBox<double, 3>;
using AlignedBox3i = Eigen::AlignedBox<int, 3>;

using VectorXd = Eigen::VectorXd;

template<typename T>
using VectorXt = Eigen::Matrix<T, Eigen::Dynamic, 1>;

using SparseMatrix = Eigen::SparseMatrix<double>;

template<typename T, int N>
VecXt<T, N> clamp(const VecXt<T, N>& vIn, const VecXt<T, N>& vMin, const VecXt<T, N>& vMax)
{
	VecXt<T, N> vOut;

	for (int i = 0; i < vIn.rows(); ++i)
		vOut[i] = std::clamp(vIn[i], vMin[i], vMax[i]);

	return vOut;
}

template<typename T, int N>
VecXt<T, N> ceil(const VecXt<T, N>& vIn)
{
	return vIn.array().ceil().matrix();
}

template<typename T, int N>
VecXt<T, N> floor(const VecXt<T, N>& vIn)
{
	return vIn.array().floor().matrix();
}

template<typename T, int N>
bool operator==(const VecXt<T, N>& v0, const VecXt<T, N>& v1)
{
	return (v0.array() == v1.array()).all();
}

template<typename T, int N>
bool operator!=(const VecXt<T, N>& v0, const VecXt<T, N>& v1)
{
	return (v0.array() != v1.array()).all();
}

template<typename RealType>
bool isNearlyEqual(const RealType a, const RealType b, const RealType tolerance = 1e-5, const bool useRelative = true)
{
	if (a == b)
		return true;

	RealType absDiff = std::fabs(a - b);

	RealType avgMag(1);

	if (useRelative)
	{
		avgMag = (std::fabs(a) + std::fabs(b)) / RealType(2);
	}

	return absDiff < tolerance * avgMag;
}

// Transforms even the sequence 0,1,2,3,... into reasonably good random numbers
// Challenge: improve on this in speed and "randomness"!
// This seems to pass several statistical tests, and is a bijective map (of 32-bit unsigneds)
inline unsigned randhash(unsigned seed)
{
    unsigned i = (seed ^ 0xA3C59AC3u) * 2654435769u;
    i ^= (i >> 16);
    i *= 2654435769u;
    i ^= (i >> 16);
    i *= 2654435769u;
    return i;
}

// the inverse of randhash
inline unsigned unhash(unsigned h)
{
    h *= 340573321u;
    h ^= (h >> 16);
    h *= 340573321u;
    h ^= (h >> 16);
    h *= 340573321u;
    h ^= 0xA3C59AC3u;
    return h;
}

// returns repeatable stateless pseudo-random number in [0,1]
inline double randhashd(unsigned seed) { return randhash(seed) / double(std::numeric_limits<unsigned>::max()); }

inline float randhashf(unsigned seed) { return randhash(seed) / float(std::numeric_limits<unsigned>::max()); }

// returns repeatable stateless pseudo-random number in [a,b]
inline double randhashd(unsigned seed, double a, double b) { return (b - a) * randhash(seed) / double(std::numeric_limits<unsigned>::max()) + a; }

inline float randhashf(unsigned seed, float a, float b) { return (b - a) * randhash(seed) / float(std::numeric_limits<unsigned>::max()) + a; }

template <typename S, typename T>
S lerp(const S& value0, const S& value1, const T& f)
{
    return (1. - f) * value0 + f * value1;
}

template<typename S, typename T>
S lerpGradient(const S& value0, const S& value1, const T&)
{
    return value1 - value0;
}

template <typename S, typename T>
S bilerp(const S& value00, const S& value10, const S& value01, const S& value11, const T& fx, const T& fy)
{
    return lerp(lerp(value00, value10, fx), lerp(value01, value11, fx), fy);
}

template <typename S, typename T>
S trilerp(const S& value000, const S& value100, const S& value010, const S& value110, const S& value001,
          const S& value101, const S& value011, const S& value111, const T& fx, const T& fy, const T& fz)
{
    return lerp(bilerp(value000, value100, value010, value110, fx, fy),
                bilerp(value001, value101, value011, value111, fx, fy), fz);
}

template<typename S, typename T>
Vec3t<S> trilerpGradient(const S& value000, const S& value100, const S& value010, const S& value110, const S& value001,
    const S& value101, const S& value011, const S& value111, const T& fx, const T& fy, const T& fz)
{
    Vec3t<S> gradient;
    gradient[0] = lerp(lerp(lerpGradient(value000, value100, fx), lerpGradient(value010, value110, fx), fy), lerp(lerpGradient(value001, value101, fx), lerpGradient(value011, value111, fx), fy), fz);
    gradient[1] = lerp(lerp(lerpGradient(value000, value010, fy), lerpGradient(value001, value011, fy), fz), lerp(lerpGradient(value100, value110, fy), lerpGradient(value101, value111, fy), fz), fx);
    gradient[2] = lerp(lerp(lerpGradient(value000, value001, fz), lerpGradient(value100, value101, fz), fx), lerp(lerpGradient(value010, value011, fz), lerpGradient(value110, value111, fz), fx), fy);
    return gradient;
}

// Catmull-Rom cubic interpolation (see https://en.wikipedia.org/wiki/Cubic_Hermite_spline).
template <typename S, typename T>
S cubicInterp(const S& value_1, const S& value0, const S& value1, const S& value2, const T& fx)
{
    T sqrfx = sqr(fx), cubefx = cube(fx);
    return T(0.5) * ((-cubefx + T(2.0) * sqrfx - fx) * value_1 + (T(3.0) * cubefx - T(5.0) * sqrfx + T(2.0)) * value0 +
                     (-T(3.0) * cubefx + T(4.0) * sqrfx + fx) * value1 + (cubefx - sqrfx) * value2);
};

}

#endif