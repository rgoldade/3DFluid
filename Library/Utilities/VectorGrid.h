#ifndef FLUIDSIM3D_VECTOR_GRID_H
#define FLUIDSIM3D_VECTOR_GRID_H

#include "tbb/blocked_range.h"
#include "tbb/blocked_range3d.h"
#include "tbb/parallel_reduce.h"

#include <array>

#include "ScalarGrid.h"
#include "Transform.h"
#include "Utilities.h"

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

namespace FluidSim3D
{
// Separating the grid settings out from the grid class
// means that we don't have to deal with templating when
// referencing these settings.
namespace VectorGridSettings
{
enum class SampleType
{
    CENTER,
    STAGGERED,
    NODE,
    EDGE
};
}

template <typename T>
class VectorGrid
{
    using ScalarSampleType = ScalarGridSettings::SampleType;
    using ScalarBorderType = ScalarGridSettings::BorderType;
    using SampleType = VectorGridSettings::SampleType;

public:
    VectorGrid() : myXform(1., Vec3d::Zero()), myGridSize(Vec3i::Zero()) {}

    VectorGrid(const Transform& xform, const Vec3i& size, SampleType sampleType = SampleType::CENTER,
               ScalarBorderType borderType = ScalarBorderType::CLAMP)
        : VectorGrid(xform, size, Vec3t<T>::Zero(), sampleType, borderType)
    {}

    VectorGrid(const Transform& xform, const Vec3i& size, Vec3t<T> initialValue, SampleType sampleType = SampleType::CENTER,
               ScalarBorderType borderType = ScalarBorderType::CLAMP)
        : myXform(xform), myGridSize(size), mySampleType(sampleType)
    {
        switch (sampleType)
        {
            case SampleType::CENTER:
                myGrids[0] = ScalarGrid<T>(xform, size, initialValue[0], ScalarSampleType::CENTER, borderType);
                myGrids[1] = ScalarGrid<T>(xform, size, initialValue[1], ScalarSampleType::CENTER, borderType);
                myGrids[2] = ScalarGrid<T>(xform, size, initialValue[2], ScalarSampleType::CENTER, borderType);
                break;
            // If the grid is 2x2x2, it has 3x2x2 x-aligned faces.
            // This is handled inside of the ScalarGrid
            case SampleType::STAGGERED:
                myGrids[0] = ScalarGrid<T>(xform, size, initialValue[0], ScalarSampleType::XFACE, borderType);
                myGrids[1] = ScalarGrid<T>(xform, size, initialValue[1], ScalarSampleType::YFACE, borderType);
                myGrids[2] = ScalarGrid<T>(xform, size, initialValue[2], ScalarSampleType::ZFACE, borderType);
                break;
            // If the grid is 2x2x2, it has 3x3x3 nodes. This is handled inside of the ScalarGrid
            case SampleType::NODE:
                myGrids[0] = ScalarGrid<T>(xform, size, initialValue[0], ScalarSampleType::NODE, borderType);
                myGrids[1] = ScalarGrid<T>(xform, size, initialValue[1], ScalarSampleType::NODE, borderType);
                myGrids[2] = ScalarGrid<T>(xform, size, initialValue[2], ScalarSampleType::NODE, borderType);
                break;
            case SampleType::EDGE:
                myGrids[0] = ScalarGrid<T>(xform, size, initialValue[0], ScalarSampleType::XEDGE, borderType);
                myGrids[1] = ScalarGrid<T>(xform, size, initialValue[1], ScalarSampleType::YEDGE, borderType);
                myGrids[2] = ScalarGrid<T>(xform, size, initialValue[2], ScalarSampleType::ZEDGE, borderType);
        }
    }

    // Check that the two grids are of the same size,
    // positioned at the same spot, have the same grid
    // spacing and the same sampling sceme
    template <typename S>
    bool isGridMatched(const VectorGrid<S>& grid) const
    {
        for (int axis : {0, 1, 2})
            if (size(axis) != grid.size(axis)) return false;

        if (myXform != grid.xform()) return false;
        if (mySampleType != grid.sampleType()) return false;

        return true;
    }

    ScalarGrid<T>& grid(int axis) { return myGrids[axis]; }

    const ScalarGrid<T>& grid(int axis) const { return myGrids[axis]; }

    T& operator()(int i, int j, int k, int axis) { return (*this)(Vec3i(i, j, k), axis); }

    T& operator()(const Vec3i& coord, int axis)
    {
        return myGrids[axis](coord);
    }

    const T& operator()(int i, int j, int k, int axis) const { return (*this)(Vec3i(i, j, k), axis); }

    const T& operator()(const Vec3i& coord, int axis) const
    {
        return myGrids[axis](coord);
    }

    T maxMagnitude() const;

    Vec3t<T> triLerp(double x, double y, double z) const { return triLerp(Vec3d(x, y, z)); }

    Vec3t<T> triLerp(const Vec3d& samplePoint) const
    {
        return Vec3t<T>(triLerp(samplePoint, 0), triLerp(samplePoint, 1), triLerp(samplePoint, 2));
    }

    T triLerp(double x, double y, double z, int axis) const { return triLerp(Vec3d(x, y, z), axis); }
    T triLerp(const Vec3d& samplePoint, int axis) const { return myGrids[axis].triLerp(samplePoint); }

    // World space vs. index space converters need to be done at the
    // underlying scalar grid level because the alignment of the three
    // grids are different depending on the SampleType.
    Vec3d indexToWorld(const Vec3d& indexPoint, int axis) const { return myGrids[axis].indexToWorld(indexPoint); }
    Vec3d worldToIndex(const Vec3d& worldPoint, int axis) const { return myGrids[axis].worldToIndex(worldPoint); }

    double dx() const { return myXform.dx(); }
    const Vec3d& offset() const { return myXform.offset(); }
    Transform xform() const { return myXform; }

    Vec3i size(int axis) const { return myGrids[axis].size(); }
    Vec3i gridSize() const { return myGridSize; }
    SampleType sampleType() const { return mySampleType; }

    // Rendering methods
    void drawGrid(Renderer& renderer) const;
    void drawSamplePoints(Renderer& renderer, const Vec3d& colour0 = Vec3d(1, 0, 0),
                          const Vec3d& colour1 = Vec3d(0, 1, 0), const Vec3d& colour2 = Vec3d(0, 0, 1),
                          const Vec3d& sampleSizes = Vec3d::Constant(5.)) const;

    void drawSamplePointCell(Renderer& renderer, const Vec3i& cell, const Vec3d& colour0 = Vec3d(1, 0, 0),
                             const Vec3d& colour1 = Vec3d(0, 1, 0), const Vec3d& colour2 = Vec3d(0, 0, 1),
                             const Vec3d& sampleSizes = Vec3d::Constant(5.)) const;

    void drawSamplePointVectors(Renderer& renderer, Axis planeAxis, double position,
                                const Vec3d& colour = Vec3d(0, 0, 1), double length = .25) const;

    void drawSuperSampledValuesPlane(Renderer& renderer, Axis gridAxis, Axis planeAxis, double position,
                                     double sampleRadius = .5, int samples = 5, double sampleSize = 1) const;

    void drawGridCell(Renderer& renderer, const Vec3i& coord) const;

private:
    // This method is private to prevent future mistakes between this transform
    // and the staggered scalar grids
    Vec3d indexToWorld(const Vec3d& point) const { return myXform.indexToWorld(point); }

    std::array<ScalarGrid<T>, 3> myGrids;

    Transform myXform;

    Vec3i myGridSize;

    SampleType mySampleType;
};

// Magnitude is useful for CFL conditions
template <typename T>
T VectorGrid<T>::maxMagnitude() const
{
    T magnitude(0);

    if (mySampleType == SampleType::CENTER || mySampleType == SampleType::NODE)
    {
        magnitude = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, myGrids[0].voxelCount(), tbbLightGrainSize), T(0),
            [&](const tbb::blocked_range<int>& range, T maxMagnitude) -> T {
                for (int index = range.begin(); index != range.end(); ++index)
                {
                    Vec3i coord = myGrids[0].unflatten(index);
                    T localMagnitude = Vec3t<T>(myGrids[0](coord), myGrids[1](coord), myGrids[2](coord)).squaredNorm();

                    maxMagnitude = std::max(maxMagnitude, localMagnitude);
                }

                return maxMagnitude;
            },
            [](T x, T y) -> T { return std::max(x, y); });
    }
    else if (mySampleType == SampleType::STAGGERED)
    {
        auto blocked_range =
            tbb::blocked_range3d<int>(0, myGridSize[0], int(std::cbrt(tbbLightGrainSize)), 0, myGridSize[1],
                                      int(std::cbrt(tbbLightGrainSize)), 0, myGridSize[2], int(std::cbrt(tbbLightGrainSize)));
        magnitude = tbb::parallel_reduce(blocked_range, T(0),
            [&](const tbb::blocked_range3d<int>& range, T maxMagnitude) -> T
            {
                Vec3i cell;

                for (cell[0] = range.pages().begin(); cell[0] != range.pages().end(); ++cell[0])
                    for (cell[1] = range.rows().begin(); cell[1] != range.rows().end(); ++cell[1])
                        for (cell[2] = range.cols().begin(); cell[2] != range.cols().end(); ++cell[2])
                        {
                            Vec3d averageVector = Vec3d::Zero();

                            for (int axis : {0, 1, 2})
                                for (int direction : {0, 1})
                                {
                                    Vec3i face = cellToFace(cell, axis, direction);
                                    averageVector[axis] += .5 * myGrids[axis](face);
                                }

                            T localMagnitude = averageVector.squaredNorm();
                            maxMagnitude = std::max(maxMagnitude, localMagnitude);
                        }

                return maxMagnitude;
            },
            [](T x, T y) -> T { return std::max(x, y); });
    }
    else
        assert(false);  // TODO: support edges

    return std::sqrt(magnitude);
}

template <typename T>
void VectorGrid<T>::drawGrid(Renderer& renderer) const
{
    myGrids[0].drawGrid(renderer);
}

template <typename T>
void VectorGrid<T>::drawGridCell(Renderer& renderer, const Vec3i& cell) const
{
    myGrids[0].drawGridCell(renderer, cell);
}

template <typename T>
void VectorGrid<T>::drawSamplePoints(Renderer& renderer, const Vec3d& colour0, const Vec3d& colour1,
                                     const Vec3d& colour2, const Vec3d& sampleSizes) const
{
    myGrids[0].drawSamplePoints(renderer, colour0, sampleSizes[0]);
    myGrids[1].drawSamplePoints(renderer, colour1, sampleSizes[1]);
    myGrids[2].drawSamplePoints(renderer, colour2, sampleSizes[2]);
}

template <typename T>
void VectorGrid<T>::drawSamplePointCell(Renderer& renderer, const Vec3i& cell, const Vec3d& colour0,
                                        const Vec3d& colour1, const Vec3d& colour2, const Vec3d& sampleSizes) const
{
    myGrids[0].drawSamplePoints(renderer, cell, colour0, sampleSizes[0]);
    myGrids[1].drawSamplePoints(renderer, cell, colour1, sampleSizes[1]);
    myGrids[2].drawSamplePoints(renderer, cell, colour2, sampleSizes[2]);
}

template <typename T>
void VectorGrid<T>::drawSamplePointVectors(Renderer& renderer, Axis planeAxis, double position, const Vec3d& colour,
                                           double length) const
{
    position = std::clamp(position, double(0), double(1));

    Vec3i start = Vec3i::Zero();
    Vec3i end = myGridSize - Vec3i::Ones();

    if (planeAxis == Axis::XAXIS)
    {
        start[0] = int(std::floor(position * double(myGridSize[0] - 1)));
        end[0] = start[0] + 1;
    }
    else if (planeAxis == Axis::YAXIS)
    {
        start[1] = int(std::floor(position * double(myGridSize[1] - 1)));
        end[1] = start[1] + 1;
    }
    else if (planeAxis == Axis::ZAXIS)
    {
        start[2] = int(std::floor(position * double(myGridSize[2] - 1)));
        end[2] = start[2] + 1;
    }

    VecVec3d startPoints;
    VecVec3d endPoints;

    forEachVoxelRange(start, end, [&](const Vec3i& cell)
    {
        Vec3d worldPoint = indexToWorld(cell.cast<double>() + Vec3d::Constant(.5));
        startPoints.push_back(worldPoint);

        Vec3t<T> sampleVector = triLerp(worldPoint);
        Vec3d vectorEnd = worldPoint + length * sampleVector;
        endPoints.push_back(vectorEnd);
    });

    renderer.addLines(startPoints, endPoints, colour);
}

template <typename T>
void VectorGrid<T>::drawSuperSampledValuesPlane(Renderer& renderer, Axis gridAxis, Axis planeAxis, double position,
                                                double sampleRadius, int samples, double sampleSize) const
{
    int gridAxisInt;
    if (gridAxis == Axis::XAXIS)
        gridAxisInt = 0;
    else if (gridAxis == Axis::XAXIS)
        gridAxisInt = 1;
    else
        gridAxisInt = 2;

    myGrids[gridAxisInt].drawSuperSampledValuesPlane(renderer, planeAxis, position, sampleRadius, samples, sampleSize);
}

}  // namespace FluidSim3D::Utilities

#endif