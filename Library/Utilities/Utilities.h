#ifndef LIBRARY_UTILITIES_H
#define LIBRARY_UTILITIES_H

#include <assert.h>
#include <vector>

#include "tbb/tbb.h"

#include "Predicates.h"
#include "Vec.h"

///////////////////////////////////
//
// Utilities.h
// Ryan Goldade 2017
//
// Everything should include this since
// we're defining floating point values
// as Real and not double or float.
//
//
////////////////////////////////////

namespace FluidSim3D::Utilities
{
	constexpr double PI = 3.1415926535897932384626433832795;

	inline Vec3i cellToCell(const Vec3i& cell, int axis, int direction)
	{
		Vec3i adjacentCell(cell);

		if (direction == 0)
			--adjacentCell[axis];
		else
		{
			assert(direction == 1);
			++adjacentCell[axis];
		}

		return adjacentCell;
	}

	inline Vec3i cellToFace(const Vec3i& cell, int axis, int direction)
	{
		Vec3i face(cell);

		if (direction == 1)
			++face[axis];
		else assert(direction == 0);

		return face;
	}

	inline Vec3i cellToEdge(const Vec3i& cell, int edgeAxis, int edgeIndex)
	{
		assert(edgeAxis >= 0 && edgeAxis < 3);
		assert(edgeIndex >= 0 && edgeIndex < 4);

		Vec3i edge(cell);

		for (int axisOffset : {0, 1})
		{
			if (edgeIndex & (1 << axisOffset))
			{
				int localAxis = (edgeAxis + 1 + axisOffset) % 3;
				++edge[localAxis];
			}
		}

		return edge;
	}

	inline Vec3i cellToNode(const Vec3i& cell, int nodeIndex)
	{
		assert(nodeIndex >= 0 && nodeIndex < 8);

		Vec3i node(cell);
		for (int axis : {0, 1, 2})
		{
			if (nodeIndex & (1 << axis))
				++node[axis];
		}

		return node;
	}

	inline Vec3i faceToCell(const Vec3i& face, int axis, int direction)
	{
		Vec3i cell(face);
		if (direction == 0)
			--cell[axis];
		else
			assert(direction == 1);

		return cell;
	}

	inline Vec3i faceToEdge(const Vec3i& face, int faceAxis, int edgeAxis, int direction)
	{
		assert(faceAxis >= 0 && faceAxis < 3 && edgeAxis >= 0 && edgeAxis < 3);
		assert(faceAxis != edgeAxis);

		Vec3i edge(face);
		if (direction == 1)
		{
			int offsetAxis = 3 - faceAxis - edgeAxis;
			++edge[offsetAxis];
		}
		else
			assert(direction == 0);

		return edge;
	}

	inline Vec3i faceToNode(const Vec3i& face, int faceAxis, int nodeIndex)
	{
		assert(faceAxis >= 0 && faceAxis < 3);
		assert(nodeIndex >= 0 && nodeIndex < 4);

		Vec3i node(face);
		for (int axisOffset : {0, 1})
		{
			if (nodeIndex & (1 << axisOffset))
			{
				int localAxis = (faceAxis + 1 + axisOffset) % 3;
				++node[localAxis];
			}
		}

		return node;
	}

	inline Vec3i faceToNodeCCW(const Vec3i& face, int faceAxis, int nodeIndex)
	{
		const Vec3i faceToNodeOffsets[3][4] = { { Vec3i(0,0,0), Vec3i(0,1,0), Vec3i(0,1,1), Vec3i(0,0,1) },
												{ Vec3i(0,0,0), Vec3i(0,0,1), Vec3i(1,0,1), Vec3i(1,0,0) },
												{ Vec3i(0,0,0), Vec3i(1,0,0), Vec3i(1,1,0), Vec3i(0,1,0) } };

		assert(faceAxis >= 0 && faceAxis < 3);
		assert(nodeIndex >= 0 && nodeIndex < 4);

		Vec3i node(face);
		node += faceToNodeOffsets[faceAxis][nodeIndex];

		return node;
	}

	inline Vec3i edgeToFace(const Vec3i& edge, int edgeAxis, int faceAxis, int direction)
	{
		assert(faceAxis >= 0 && faceAxis < 3 && edgeAxis >= 0 && edgeAxis < 3);
		assert(faceAxis != edgeAxis);

		Vec3i face(edge);
		if (direction == 0)
		{
			int offsetAxis = 3 - faceAxis - edgeAxis;
			--face[offsetAxis];
		}
		else
			assert(direction == 1);

		return face;
	}

	inline Vec3i edgeToCell(const Vec3i& edge, int edgeAxis, int cellIndex)
	{
		assert(edgeAxis >= 0 && edgeAxis < 3);
		assert(cellIndex >= 0 && cellIndex < 4);

		Vec3i cell(edge);
		for (int axisOffset : {0, 1})
		{
			if (!(cellIndex & (1 << axisOffset)))
			{
				int localAxis = (edgeAxis + 1 + axisOffset) % 3;
				--cell[localAxis];
			}
		}

		return cell;
	}

	inline Vec3i edgeToCellCCW(const Vec3i& edge, int edgeAxis, int cellIndex)
	{
		const Vec3i edgeToCellOffsets[3][4] = { { Vec3i( 0,-1,-1), Vec3i( 0, 0,-1), Vec3i( 0, 0, 0), Vec3i( 0,-1, 0) },
												{ Vec3i(-1, 0,-1), Vec3i(-1, 0, 0), Vec3i( 0, 0, 0), Vec3i( 0, 0,-1) },
												{ Vec3i(-1,-1, 0), Vec3i( 0,-1, 0), Vec3i( 0, 0, 0), Vec3i(-1, 0, 0) } };

		assert(edgeAxis >= 0 && edgeAxis < 3);
		assert(cellIndex >= 0 && cellIndex < 4);

		Vec3i cell(edge);
		cell += edgeToCellOffsets[edgeAxis][cellIndex];

		return cell;
	}

	inline Vec3i edgeToNode(const Vec3i& edge, int axis, int direction)
	{
		Vec3i node(edge);
		if (direction == 1)
			++node[axis];
		else assert(direction == 0);

		return node;
	}

	inline Vec3i nodeToFace(const Vec3i& node, int faceAxis, int faceIndex)
	{
		assert(faceAxis >= 0 && faceAxis < 3);
		assert(faceIndex >= 0 && faceIndex < 4);

		Vec3i face(node);
		for (int axisOffset : {0, 1})
		{
			if (!(faceIndex & (1 << axisOffset)))
			{
				int localAxis = (faceAxis + 1 + axisOffset) % 3;
				--face[localAxis];
			}
		}

		return face;
	}

	inline Vec3i nodeToCell(const Vec3i& node, int cellIndex)
	{
		assert(cellIndex >= 0 && cellIndex < 8);

		Vec3i cell(node);
		for (int axis : {0, 1, 2})
		{
			if (!(cellIndex & (1 << axis)))
				--cell[axis];
		}

		return cell;
	}

	const Vec3f colours[] = { Vec3f(1,0,0), Vec3f(0,1,0), Vec3f(0,0,1), Vec3f(1,1,0), Vec3f(1,0,1), Vec3f(0,1,1) };

	template<typename Real>
	Real lengthFraction(Real phi0, Real phi1)
	{
		Real theta = 0.;

		if (phi0 < 0)
		{
			if (phi1 < 0)
				theta = 1;
			else if (phi1 >= 0)
				theta = phi0 / (phi0 - phi1);
		}
		else if (phi1 < 0)
			theta = phi1 / (phi1 - phi0);

		return theta;
	}

	// Execute function "f" over range [start, end)
	template<typename T, typename Function>
	void forEachVoxelRange(const Vec<3, T>& start, const Vec<3, T>& end, const Function& f)
	{
		Vec<3, T> cell;
		for (cell[0] = start[0]; cell[0] < end[0]; ++cell[0])
			for (cell[1] = start[1]; cell[1] < end[1]; ++cell[1])
				for (cell[2] = start[2]; cell[2] < end[2]; ++cell[2])
					f(cell);
	}

	template<typename T, typename Function>
	void forEachVoxelRangeReverse(const Vec<3, T>& start, const Vec<3, T>& end, const Function& f)
	{
		Vec<3, T> cell;
		for (cell[0] = end[0] - 1; cell[0] >= start[0]; --cell[0])
			for (cell[1] = end[1] - 1; cell[1] >= start[1]; --cell[1])
				for (cell[2] = end[2] - 1; cell[2] >= start[2]; --cell[2])
					f(cell);
	}

	//
	// TBB utilities
	//

	constexpr int tbbLightGrainSize = 1000;
	constexpr int tbbHeavyGrainSize = 100;

	template<typename StorageType>
	void mergeLocalThreadVectors(std::vector<StorageType>& combinedVector,
									tbb::enumerable_thread_specific<std::vector<StorageType>>& parallelVector)
	{
		int vectorSize = 0;

		parallelVector.combine_each([&](const std::vector<StorageType>& localVector)
		{
			vectorSize += localVector.size();
		});

		combinedVector.reserve(combinedVector.size() + vectorSize);

		parallelVector.combine_each([&](const std::vector<StorageType>& localVector)
		{
			combinedVector.insert(combinedVector.end(), localVector.begin(), localVector.end());
		});
	}

	//
	// BFS markers
	//

	enum class VisitedCellLabels { UNVISITED_CELL, VISITED_CELL, FINISHED_CELL };


	///////////////////////////////////
	//
	// Borrowed and modified from
	// Robert Bridson's sample code.
	//
	////////////////////////////////////

#ifdef _MSC_VER
	#undef min
	#undef max
#endif

	using std::min;
	using std::max;
	using std::swap;

	template<typename T>
	T sqr(T x)
	{
		return x * x;
	}

	template<typename T>
	T cube(T x)
	{
		return x * sqr(x);
	}

	template<typename... Args>
	decltype(auto) min(const Args&... values)
	{
		return std::min({ values... });
	}

	template<typename... Args>
	decltype(auto) max(const Args&... values)
	{
		return std::max({ values... });
	}

	template<typename T>
	void minAndMax(T& minValue, T& maxValue, const T& value0, const T& value1)
	{
		minValue = std::min(value0, value1);
		maxValue = std::max(value0, value1);
	}

	template<typename T>
	void updateMinOrMax(T& minValue, T& maxValue, const T& value)
	{
		if (value < minValue) minValue = value;
		else if (value > maxValue) maxValue = value;
	}

	template<typename T>
	void updateMinAndMax(T& minValue, T& maxValue, const T& value)
	{
		if (value < minValue) minValue = value;
		if (value > maxValue) maxValue = value;
	}

	template<typename T>
	T clamp(const T& value, const T& lower, const T& upper)
	{
		if (value < lower) return lower;
		else if (value > upper) return upper;
		else return value;
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

	constexpr unsigned maxInteger = std::numeric_limits<unsigned>::max();

	// returns repeatable stateless pseudo-random number in [0,1]
	inline double randhashd(unsigned seed)
	{
		return randhash(seed) / double(maxInteger);
	}

	inline float randhashf(unsigned seed)
	{
		return randhash(seed) / float(maxInteger);
	}

	// returns repeatable stateless pseudo-random number in [a,b]
	inline double randhashd(unsigned seed, double a, double b)
	{
		return (b - a) * randhash(seed) / double(maxInteger) + a;
	}

	inline float randhashf(unsigned seed, float a, float b)
	{
		return (b - a) * randhash(seed) / float(maxInteger) + a;
	}

	template<typename S, typename T>
	S lerp(const S& value0, const S& value1, const T& f)
	{
		return (1. - f) * value0 + f * value1;
	}

	template<typename S, typename T>
	S bilerp(const S& value00, const S& value10,
				const S& value01, const S& value11,
				const T& fx, const T& fy)
	{
		return lerp(lerp(value00, value10, fx),
			lerp(value01, value11, fx),
			fy);
	}

	template<typename S, typename T>
	S trilerp(const S& value000, const S& value100,
		const S& value010, const S& value110,
		const S& value001, const S& value101,
		const S& value011, const S& value111,
		const T& fx, const T& fy, const T& fz)
	{
		return lerp(bilerp(value000, value100, value010, value110, fx, fy),
			bilerp(value001, value101, value011, value111, fx, fy),
			fz);
	}

	// Catmull-Rom cubic interpolation (see https://en.wikipedia.org/wiki/Cubic_Hermite_spline).
	template<typename S, typename T>
	S cubicInterp(const S& value_1, const S& value0, const S& value1, const S& value2, const T& fx)
	{
		T sqrfx = sqr(fx), cubefx = cube(fx);
		return T(0.5) * ((-cubefx + T(2.0) * sqrfx - fx) * value_1
			+ (T(3.0) * cubefx - T(5.0) * sqrfx + T(2.0)) * value0
			+ (-T(3.0) * cubefx + T(4.0) * sqrfx + fx) * value1
			+ (cubefx - sqrfx) * value2);
	};
}

#endif