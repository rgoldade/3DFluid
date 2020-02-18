#ifndef LIBRARY_VECTORGRID_H
#define LIBRARY_VECTORGRID_H

#include "ScalarGrid.h"
#include "Transform.h"
#include "Utilities.h"
#include "Vec.h"

///////////////////////////////////
//
// VectorGrid.h
// Ryan Goldade 2017
//
// Container class of three ScalarGrids
// to represent a vector field.
// Provides control for cell-centered
// or staggered formation.
//
////////////////////////////////////

namespace FluidSim3D::Utilities
{

// Separating the grid settings out from the grid class
// means that we don't have to deal with templating when
// referencing these settings.
namespace VectorGridSettings
{
	enum class SampleType { CENTER, STAGGERED, NODE, EDGE };
}

template<typename T>
class VectorGrid
{
	using ScalarGrid = ScalarGrid<T>;
	using ScalarSampleType = ScalarGridSettings::SampleType;
	using ScalarBorderType = ScalarGridSettings::BorderType;
	using SampleType = VectorGridSettings::SampleType;

public:

	VectorGrid() : myXform(1., Vec3f(0.)), myGridSize(0) {}

	VectorGrid(const Transform& xform, const Vec3i& size,
				SampleType sampleType = SampleType::CENTER,
				ScalarBorderType borderType = ScalarBorderType::CLAMP)
	: VectorGrid(xform, size, T(0), sampleType, borderType)
	{}

	VectorGrid(const Transform& xform, const Vec3i& size, T initialValue,
				SampleType sampleType = SampleType::CENTER,
				ScalarBorderType borderType = ScalarBorderType::CLAMP)
		: myXform(xform)
		, myGridSize(size)
		, mySampleType(sampleType)
	{
		switch (sampleType)
		{
		case SampleType::CENTER:
			myGrids[0] = ScalarGrid(xform, size, initialValue, ScalarSampleType::CENTER, borderType);
			myGrids[1] = ScalarGrid(xform, size, initialValue, ScalarSampleType::CENTER, borderType);
			myGrids[2] = ScalarGrid(xform, size, initialValue, ScalarSampleType::CENTER, borderType);
			break;
		// If the grid is 2x2x2, it has 3x2x2 x-aligned faces.
		// This is handled inside of the ScalarGrid
		case SampleType::STAGGERED:
			myGrids[0] = ScalarGrid(xform, size, initialValue, ScalarSampleType::XFACE, borderType);
			myGrids[1] = ScalarGrid(xform, size, initialValue, ScalarSampleType::YFACE, borderType);
			myGrids[2] = ScalarGrid(xform, size, initialValue, ScalarSampleType::ZFACE, borderType);
			break;
		// If the grid is 2x2x2, it has 3x3x3 nodes. This is handled inside of the ScalarGrid
		case SampleType::NODE:
			myGrids[0] = ScalarGrid(xform, size, initialValue, ScalarSampleType::NODE, borderType);
			myGrids[1] = ScalarGrid(xform, size, initialValue, ScalarSampleType::NODE, borderType);
			myGrids[2] = ScalarGrid(xform, size, initialValue, ScalarSampleType::NODE, borderType);
			break;
		case SampleType::EDGE:
			myGrids[0] = ScalarGrid(xform, size, initialValue, ScalarSampleType::XEDGE, borderType);
			myGrids[1] = ScalarGrid(xform, size, initialValue, ScalarSampleType::YEDGE, borderType);
			myGrids[2] = ScalarGrid(xform, size, initialValue, ScalarSampleType::ZEDGE, borderType);
		}
	}

	// Check that the two grids are of the same size, 
	// positioned at the same spot, have the same grid
	// spacing and the same sampling sceme
	template<typename S>
	bool isGridMatched(const VectorGrid<S>& grid) const
	{
		for (int axis : {0, 1, 2})
			if (size(axis) != grid.size(axis)) return false;

		if (myXform != grid.xform()) return false;
		if (mySampleType != grid.sampleType()) return false;

		return true;
	}

	ScalarGrid& grid(int axis)
	{
		return myGrids[axis];
	}

	const ScalarGrid& grid(int axis) const
	{
		return myGrids[axis];
	}

	// write renderers to see the sample points, vector values (averaged and offset for staggered)
	T& operator()(int i, int j, int k, int axis) { return (*this)(Vec3i(i, j, k), axis); }

	T& operator()(const Vec3i& coord, int axis)
	{
		// Uniform grid checks that coord indices are valid.
		return myGrids[axis](coord);
	}

	const T& operator()(int i, int j, int k, int axis) const { return (*this)(Vec3i(i, j, k), axis); }

	const T& operator()(const Vec3i& coord, int axis) const
	{
		// Uniform grid checks that coord indices are valid.
		return myGrids[axis](coord);
	}

	T maxMagnitude() const;

	Vec<3, T> interp(float x, float y, float z) const { return interp(Vec3f(x, y, z)); }

	Vec<3, T> interp(const Vec3f& pos) const
	{
		return Vec<3, T>(interp(pos, 0), interp(pos, 1), interp(pos, 2));
	}

	T interp(float x, float y, float z, int axis) const { return interp(Vec3f(x, y, z), axis); }
	T interp(const Vec3f& pos, int axis) const
	{
		return myGrids[axis].interp(pos);
	}

	// World space vs. index space converters need to be done at the 
	// underlying scalar grid level because the alignment of the three 
	// grids are different depending on the SampleType.
	Vec3f indexToWorld(const Vec3f& indexPoint, int axis) const
	{
		return myGrids[axis].indexToWorld(indexPoint);
	}
	Vec3f worldToIndex(const Vec3f& worldPoint, int axis) const
	{
		return myGrids[axis].worldToIndex(worldPoint);
	}

	float dx() const { return myXform.dx(); }
	float offset() const { return myXform.offset(); }
	Transform xform() const { return myXform; }

	Vec3i size(int axis) const { return myGrids[axis].size(); }
	SampleType sampleType() const { return mySampleType; }

	// Rendering methods
	void drawGrid(Renderer& renderer) const;
	void drawSamplePoints(Renderer& renderer,
							const Vec3f& colour0 = Vec3f(1, 0, 0),
							const Vec3f& colour1 = Vec3f(0, 1, 0),
							const Vec3f& colour2 = Vec3f(0, 0, 1),
							const Vec3f& sizes = Vec3f(5.)) const;

	void drawSamplePointCell(Renderer& renderer, const Vec3i& cell,
								const Vec3f& colour0 = Vec3f(1, 0, 0),
								const Vec3f& colour1 = Vec3f(0, 1, 0),
								const Vec3f& colour2 = Vec3f(0, 0, 1),
								const Vec3f& sampleSizes = Vec3R(5.)) const;

	void drawSamplePointVectors(Renderer& renderer, Axis planeAxis, float position,
									const Vec3f& colour = Vec3f(0,0,1), float length = .25) const;

	void drawSuperSampledValuesPlane(Renderer& renderer, Axis gridAxis, Axis planeAxis, float position,
										float sampleRadius = .5, int samples = 5, float sampleSize = 1) const;

	void drawGridCell(Renderer& renderer, const Vec3i& coord) const;

private:

	// This method is private to prevent future mistakes between this transform
	// and the staggered scalar grids
	Vec3f indexToWorld(const Vec3f& point) const
	{
		return myXform.indexToWorld(point);
	}

	std::array<ScalarGrid, 3> myGrids;

	Transform myXform;

	Vec3i myGridSize;

	SampleType mySampleType;
};

// Magnitude is useful for CFL conditions
template<typename T>
T VectorGrid<T>::maxMagnitude() const
{
	T magnitude(0);

	if (mySampleType == SampleType::CENTER || mySampleType == SampleType::NODE)
	{
		T magnitude = tbb::parallel_reduce(tbb::blocked_range<int>(0, myGrids[0].voxelCount(), tbbLightGrainSize), 0,
			[&](const tbb::blocked_range<int>& range, T maxMagnitude) -> T
		{
			for (int index = range.begin(); index != range.end(); ++index)
			{
				Vec3i coord = myGrids[0].unflatten(index);
				T localMagnitude = mag2(Vec<3, T>(myGrids[0](coord), myGrids[1](coord), myGrids[2](coord)));

				maxMagnitude = std::max(maxMagnitude, localMagnitude);
			}

			return maxMagnitude;
		},
			[](T x, T y) -> T
		{
			return std::max(x, y);
		});
	}
	else if (mySampleType == SampleType::STAGGERED)
	{
		auto blocked_range = tbb::blocked_range3d<int>(0, myGridSize[0], std::cbrt(tbbLightGrainSize), 0, myGridSize[1], std::cbrt(tbbLightGrainSize), 0, myGridSize[2], std::cbrt(tbbLightGrainSize));
		T magnitude = tbb::parallel_reduce(blocked_range, 0,
			[&](const tbb::blocked_range3d<int>& range, T maxMagnitude) -> T
		{
			Vec3i cell;

			for (cell[0] = range.pages().begin(); cell[0] != range.pages().end(); ++cell[0])
				for (cell[1] = range.rows().begin(); cell[1] != range.rows().end(); ++cell[1])
					for (cell[2] = range.cols().begin(); cell[2] != range.cols().end(); ++cell[2])
					{
						Vec3f averageVector(0);

						for (int axis : {0, 1, 2})
							for (int direction : {0, 1})
							{
								Vec3i face = cellToFace(cell, axis, direction);
								averageVector[axis] += .5 * myGrids[axis](face);
							}

						T localMagnitude = mag2(averageVector);
						maxMagnitude = std::max(maxMagnitude, localMagnitude);
					}

			return maxMagnitude;
		},
			[](T x, T y) -> T
		{
			return std::max(x, y);
		});
	}
	else assert(false); // TODO: support edges

	return std::sqrt(magnitude);
}

template<typename T>
void VectorGrid<T>::drawGrid(Renderer& renderer) const
{
	myGrids[0].drawGrid(renderer);
}

template<typename T>
void VectorGrid<T>::drawGridCell(Renderer& renderer, const Vec3i& cell) const
{
	myGrids[0].drawGridCell(renderer, cell);
}

template<typename T>
void VectorGrid<T>::drawSamplePoints(Renderer& renderer,
										const Vec3f& colour0,
										const Vec3f& colour1,
										const Vec3f& colour2,
										const Vec3f& sampleSizes) const
{
	myGrids[0].drawSamplePoints(renderer, colour0, sampleSizes[0]);
	myGrids[1].drawSamplePoints(renderer, colour1, sampleSizes[1]);
	myGrids[2].drawSamplePoints(renderer, colour2, sampleSizes[2]);
}

template<typename T>
void VectorGrid<T>::drawSamplePointCell(Renderer& renderer, const Vec3i& cell,
										const Vec3f& colour0,
										const Vec3f& colour1,
										const Vec3f& colour2,
										const Vec3f& sampleSizes) const
{
	myGrids[0].drawSamplePoints(renderer, cell, colour0, sampleSizes[0]);
	myGrids[1].drawSamplePoints(renderer, cell, colour1, sampleSizes[1]);
	myGrids[2].drawSamplePoints(renderer, cell, colour2, sampleSizes[2]);
}

template<typename T>
void VectorGrid<T>::drawSamplePointVectors(Renderer& renderer, Axis planeAxis, float position,
												const Vec3f& colour, float length) const
{
	position = clamp(position, float(0), float(1));

	Vec3i start(0);
	Vec3i end = myGridSize - Vec3i(1);

	if (planeAxis == Axis::XAXIS)
	{
		start[0] = std::floor(position * float(myGridSize[0] - 1));
		end[0] = start[0] + 1;
	}
	else if (planeAxis == Axis::YAXIS)
	{
		start[1] = std::floor(position * float(myGridSize[1] - 1));
		end[1] = start[1] + 1;
	}
	else if (planeAxis == Axis::ZAXIS)
	{
		start[2] = std::floor(position * float(myGridSize[2] - 1));
		end[2] = start[2] + 1;
	}

	std::vector<Vec3R> startPoints;
	std::vector<Vec3R> endPoints;

	forEachVoxelRange(start, end, [&](const Vec3i& cell)
	{
		Vec3R worldPoint = indexToWorld(Vec3R(cell) + Vec3R(.5));
		startPoints.push_back(worldPoint);

		Vec<3, T> sampleVector = interp(worldPoint);
		Vec3f vectorEnd = worldPoint + length * sampleVector;
		endPoints.push_back(vectorEnd);
	});

	renderer.addLines(startPoints, endPoints, colour);
}

template<typename T>
void VectorGrid<T>::drawSuperSampledValuesPlane(Renderer& renderer, Axis gridAxis, Axis planeAxis, float position,
												float sampleRadius, int samples, float sampleSize) const
{
	int gridAxisInt;
	if (gridAxis == Axis::XAXIS) gridAxisInt = 0;
	else if (gridAxis == Axis::XAXIS) gridAxisInt = 1;
	else gridAxisInt = 2;

	myGrids[gridAxisInt].drawSuperSampledValuesPlane(renderer, planeAxis, position, sampleRadius, samples, sampleSize);
}

}
#endif