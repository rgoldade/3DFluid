#ifndef LIBRARY_SCALAR_GRID_H
#define LIBRARY_SCALAR_GRID_H

#include "GridUtilities.h"
#include "Renderer.h"
#include "Transform.h"
#include "UniformGrid.h"
#include "Utilities.h"
#include "Vec.h"

///////////////////////////////////
//
// ScalarGrid.h
// Ryan Goldade 2017
//
// Thin wrapper around UniformGrid
// for scalar data types. ScalarGrid
// can be non-uniform and offset
// from the origin. It also allows
// for interpolation, etc. that would
// too specific for a generic templated
// grid.
//
////////////////////////////////////

namespace FluidSim3D::Utilities
{
using namespace RenderTools;

// Separating the grid settings out from the grid class
// means that we don't have to deal with templating when
// referencing these settings.
namespace ScalarGridSettings
{
enum class BorderType
{
    CLAMP,
    ZERO,
    ASSERT
};
enum class SampleType
{
    CENTER,
    XFACE,
    YFACE,
    ZFACE,
    XEDGE,
    YEDGE,
    ZEDGE,
    NODE
};
}  // namespace ScalarGridSettings

template <typename T>
class ScalarGrid : public UniformGrid<T>
{
    using BorderType = ScalarGridSettings::BorderType;
    using SampleType = ScalarGridSettings::SampleType;

public:
    ScalarGrid() : myXform(1., Vec3f(0.)), myGridSize(Vec3i(0)), UniformGrid<T>() {}

    ScalarGrid(const Transform& xform, const Vec3i& size, SampleType sampleType = SampleType::CENTER,
               BorderType borderType = BorderType::CLAMP)
        : ScalarGrid(xform, size, T(0), sampleType, borderType)
    {
    }

    // The grid size is the number of actual grid cells to be created. This means that a 2x2 grid
    // will have 3x3 nodes, 3x2 x-aligned faces, 2x3 y-aligned faces and 2x2 cell centers. The size of
    // the underlying storage container is reflected accordingly based the sample type to give the outside
    // caller the structure of a real grid.
    ScalarGrid(const Transform& xform, const Vec3i& size, const T& initialValue,
               SampleType sampleType = SampleType::CENTER, BorderType borderType = BorderType::CLAMP)
        : myXform(xform), mySampleType(sampleType), myBorderType(borderType), myGridSize(size)
    {
        switch (sampleType)
        {
            case SampleType::CENTER:
                myCellOffset = Vec3f(.5);
                this->resize(size, initialValue);
                break;
            case SampleType::XFACE:
                myCellOffset = Vec3f(.0, .5, .5);
                this->resize(size + Vec3i(1, 0, 0), initialValue);
                break;
            case SampleType::YFACE:
                myCellOffset = Vec3f(.5, .0, .5);
                this->resize(size + Vec3i(0, 1, 0), initialValue);
                break;
            case SampleType::ZFACE:
                myCellOffset = Vec3f(.5, .5, .0);
                this->resize(size + Vec3i(0, 0, 1), initialValue);
                break;
            case SampleType::XEDGE:
                myCellOffset = Vec3f(.5, .0, .0);
                this->resize(size + Vec3i(0, 1, 1), initialValue);
                break;
            case SampleType::YEDGE:
                myCellOffset = Vec3f(.0, .5, .0);
                this->resize(size + Vec3i(1, 0, 1), initialValue);
                break;
            case SampleType::ZEDGE:
                myCellOffset = Vec3f(.0, .0, .5);
                this->resize(size + Vec3i(1, 1, 0), initialValue);
                break;
            case SampleType::NODE:
                myCellOffset = Vec3f(.0);
                this->resize(size + Vec3i(1), initialValue);
        }
    }

    SampleType sampleType() const { return mySampleType; }

    // Check that the two grids are of the same size,
    // positioned at the same spot, have the same grid
    // spacing and the same sampling sceme
    template <typename S>
    bool isGridMatched(const ScalarGrid<S>& grid) const
    {
        if (this->mySize != grid.size()) return false;
        if (this->myXform != grid.xform()) return false;
        if (this->mySampleType != grid.sampleType()) return false;
        return true;
    }

    // Global multiply operator
    void operator*(const T& scalar)
    {
        tbb::parallel_for(tbb::blocked_range<int>(0, this->myGrid.size(), tbbLightGrainSize),
                          [&](const tbb::blocked_range<int>& range) {
                              for (int index = range.begin(); index != range.end(); ++index)
                                  this->myGrid[index] *= scalar;
                          });
    }

    // Global add operator
    void operator+(const T& scalar)
    {
        tbb::parallel_for(tbb::blocked_range<int>(0, this->myGrid.size(), tbbLightGrainSize),
                          [&](const tbb::blocked_range<int>& range) {
                              for (int index = range.begin(); index != range.end(); ++index)
                                  this->myGrid[index] += scalar;
                          });
    }

    T maxValue() const
    {
        return tbb::parallel_reduce(
            tbb::blocked_range<int>(0, this->myGrid.size(), tbbLightGrainSize), std::numeric_limits<T>::lowest(),
            [&](const tbb::blocked_range<int>& range, T maxValue) -> T {
                for (int index = range.begin(); index != range.end(); ++index)
                    maxValue = std::max(maxValue, this->myGrid[index]);
                return maxValue;
            },
            [](T x, T y) -> T { return std::max(x, y); });
    }

    T minValue() const
    {
        return tbb::parallel_reduce(
            tbb::blocked_range<int>(0, this->myGrid.size(), tbbLightGrainSize), std::numeric_limits<T>::max(),
            [&](const tbb::blocked_range<int>& range, T minValue) -> T {
                for (int index = range.begin(); index != range.end(); ++index)
                    minValue = std::min(minValue, this->myGrid[index]);
                return minValue;
            },
            [](T x, T y) -> T { return std::min(x, y); });
    }

    std::pair<T, T> minAndMaxValue() const
    {
        using MinMaxPair = std::pair<T, T>;

        MinMaxPair result = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, this->voxelCount(), tbbLightGrainSize),
            MinMaxPair(std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest()),
            [&](const tbb::blocked_range<int>& range, MinMaxPair valuePair) -> MinMaxPair {
                T localMin = valuePair.first;
                T localMax = valuePair.second;

                for (int index = range.begin(); index != range.end(); ++index)
                {
                    localMin = std::min(localMin, this->myGrid[index]);
                    localMax = std::max(localMax, this->myGrid[index]);
                }
                return MinMaxPair(localMin, localMax);
            },
            [](MinMaxPair x, MinMaxPair y) -> MinMaxPair {
                return MinMaxPair(std::min(x.first, y.first), std::max(x.second, y.second));
            });

        return std::pair<T, T>(result.first, result.second);
    }

    T interp(float x, float y, float z, bool isIndexSpace = false) const
    {
        return interp(Vec3f(x, y, z), isIndexSpace);
    }
    T interp(const Vec3f& samplePoint, bool isIndexSpace = false) const;

    // Converters between world space and local index space
    Vec3f indexToWorld(const Vec3f& indexPoint) const { return myXform.indexToWorld(indexPoint + myCellOffset); }

    Vec3f worldToIndex(const Vec3f& worldPoint) const { return myXform.worldToIndex(worldPoint) - myCellOffset; }

    // Gradient operators
    Vec<3, T> gradient(const Vec3f& worldPoint, bool isIndexSpace = false) const
    {
        constexpr float indexSpaceOffset(1E-1);

        float offset = isIndexSpace ? indexSpaceOffset : indexSpaceOffset * myXform.dx();

        T dTdx = interp(worldPoint + Vec3f(offset, 0., 0.), isIndexSpace) -
                 interp(worldPoint - Vec3f(offset, 0., 0.), isIndexSpace);
        T dTdy = interp(worldPoint + Vec3f(0., offset, 0.), isIndexSpace) -
                 interp(worldPoint - Vec3f(0., offset, 0.), isIndexSpace);
        T dTdz = interp(worldPoint + Vec3f(0., 0., offset), isIndexSpace) -
                 interp(worldPoint - Vec3f(0., 0., offset), isIndexSpace);

        Vec<3, T> grad(dTdx, dTdy, dTdz);
        return grad / (2. * offset);
    }

    float dx() const { return myXform.dx(); }
    Vec3f offset() const { return myXform.offset(); }
    Transform xform() const { return myXform; }

    // Render methods
    void drawGrid(Renderer& renderer) const;
    void drawGridCell(Renderer& renderer, const Vec3i& cell, const Vec3f& colour = Vec3f(0)) const;
    void drawGridPlane(Renderer& renderer, Axis planeAxis, float position) const;

    void drawSamplePoints(Renderer& renderer, const Vec3f& colour = Vec3f(1, 0, 0), float sampleSize = 5.) const;
    void drawCellSamplePoint(Renderer& renderer, const Vec3i& cell, const Vec3f& colour = Vec3f(1, 0, 0),
                             float sampleSize = 5.) const;

    void drawSupersampledValues(Renderer& renderer, const Vec3f& start, const Vec3f& end, const Vec3f& sampleRadius,
                                int samples, float sampleSize = 5) const;
    void drawSupersampledValuesVolume(Renderer& renderer, float sampleRadius = .5, int samples = 5,
                                      float sampleSize = 1) const;

    // Display a supersampled slice of the grid. The plane will have a normal in the planeAxis direction.
    // The position is from [0,1] where 0 is at the grid origin and 1 is at the origin + size * dx.
    void drawSupersampledValuesPlane(Renderer& renderer, Axis planeAxis, float position, float sampleRadius = .5,
                                     int samples = 5, float sampleSize = 1) const;
    void drawSampleGradientsPlane(Renderer& renderer, Axis planeAxis, float position, const Vec3f& colour = Vec3f(.5),
                                  float length = .25) const;

private:
    // The main interpolation call after the template specialized clamping passes
    T interpLocal(const Vec3f& pos) const;

    // Store the actual grid size. The mySize member of UniformGrid represents the
    // underlying sample grid. The actual grid doesn't change based on sample
    // type but the underlying array of sample points do.
    Vec3i myGridSize;

    // The transform accounts for the grid spacings and the grid offset.
    // It includes transforms to and from index space.
    // The offset specifies the location of the lowest, left-most corner
    // of the grid. The actual sample point is offset (in index space)
    // from this point based on the SampleType
    Transform myXform;

    SampleType mySampleType;
    BorderType myBorderType;

    // The local offset (in index space) associated with the sample type
    Vec3f myCellOffset;
};

template <typename T>
T ScalarGrid<T>::interp(const Vec3f& samplePoint, bool isIndexSpace) const
{
    Vec3f indexPoint = isIndexSpace ? samplePoint : worldToIndex(samplePoint);

    switch (myBorderType)
    {
        case BorderType::ZERO:

            for (int axis : {0, 1, 2})
            {
                if (indexPoint[axis] < 0 || indexPoint[axis] > float(this->mySize[axis] - 1)) return T(0);
            }

            break;

        case BorderType::CLAMP:

            for (int axis : {0, 1, 2})
                indexPoint[axis] = clamp(indexPoint[axis], float(0), float(this->mySize[axis] - 1));

            break;

        // Useful debug check. Equivalent to "NONE" for release mode
        case BorderType::ASSERT:
            for (int axis : {0, 1, 2}) assert(indexPoint[axis] >= 0 && indexPoint[axis] <= float(this->mySize[0] - 1));
            break;
    }

    return interpLocal(indexPoint);
}

// The local interp applies tri-linear interpolation on the UniformGrid. The
// templated type must have operators for basic add/mult arithmetic.
template <typename T>
T ScalarGrid<T>::interpLocal(const Vec3f& indexPoint) const
{
    Vec3f floorPoint = floor(indexPoint);
    Vec3i baseSampleCell = Vec3i(floorPoint);

    for (int axis : {0, 1, 2})
    {
        if (baseSampleCell[axis] == this->mySize[axis] - 1) --baseSampleCell[axis];
    }

    // Use base grid class operator
    T v000 = (*this)(baseSampleCell[0], baseSampleCell[1], baseSampleCell[2]);
    T v100 = (*this)(baseSampleCell[0] + 1, baseSampleCell[1], baseSampleCell[2]);

    T v010 = (*this)(baseSampleCell[0], baseSampleCell[1] + 1, baseSampleCell[2]);
    T v110 = (*this)(baseSampleCell[0] + 1, baseSampleCell[1] + 1, baseSampleCell[2]);

    T v001 = (*this)(baseSampleCell[0], baseSampleCell[1], baseSampleCell[2] + 1);
    T v101 = (*this)(baseSampleCell[0] + 1, baseSampleCell[1], baseSampleCell[2] + 1);

    T v011 = (*this)(baseSampleCell[0], baseSampleCell[1] + 1, baseSampleCell[2] + 1);
    T v111 = (*this)(baseSampleCell[0] + 1, baseSampleCell[1] + 1, baseSampleCell[2] + 1);

    Vec3f dx = indexPoint - floorPoint;

    for (int axis : {0, 1, 2}) assert(dx[axis] >= 0 && dx[axis] <= 1);

    return trilerp(v000, v100, v010, v110, v001, v101, v011, v111, dx[0], dx[1], dx[2]);
}

template <typename T>
void ScalarGrid<T>::drawGrid(Renderer& renderer) const
{
    std::vector<Vec3f> startPoints;
    std::vector<Vec3f> endPoints;

    for (int axis : {0, 1, 2})
    {
        Vec3i start(0);
        Vec3i end = myGridSize;
        end[axis] = 1;

        forEachVoxelRange(start, end, [&](const Vec3i& cell) {
            Vec3f gridStart(cell);

            Vec3f startPoint = indexToWorld(gridStart - myCellOffset);
            startPoints.push_back(startPoint);

            Vec3f gridEnd(cell);
            gridEnd[axis] = myGridSize[axis];

            Vec3f endPoint = indexToWorld(gridEnd - myCellOffset);
            endPoints.push_back(endPoint);
        });
    }

    renderer.addLines(startPoints, endPoints, Vec3f(0));
}

template <typename T>
void ScalarGrid<T>::drawGridCell(Renderer& renderer, const Vec3i& cell, const Vec3f& colour) const
{
    std::vector<Vec3f> startPoints;
    std::vector<Vec3f> endPoints;

    const Vec3f edgeToNodeOffset[12][2] = {
        {Vec3f(0, 0, 0), Vec3f(1, 0, 0)},  // x-axis edges
        {Vec3f(0, 0, 1), Vec3f(1, 0, 1)}, {Vec3f(0, 1, 0), Vec3f(1, 1, 0)}, {Vec3f(0, 1, 1), Vec3f(1, 1, 1)},

        {Vec3f(0, 0, 0), Vec3f(0, 1, 0)},  // y-axis edges
        {Vec3f(1, 0, 0), Vec3f(1, 1, 0)}, {Vec3f(0, 0, 1), Vec3f(0, 1, 1)}, {Vec3f(1, 0, 1), Vec3f(1, 1, 1)},

        {Vec3f(0, 0, 0), Vec3f(0, 0, 1)},  // z-axis edges
        {Vec3f(1, 0, 0), Vec3f(1, 0, 1)}, {Vec3f(0, 1, 0), Vec3f(0, 1, 1)}, {Vec3f(1, 1, 0), Vec3f(1, 1, 1)}};

    for (int edgeIndex = 0; edgeIndex < 12; ++edgeIndex)
    {
        Vec3f startNode = indexToWorld(Vec3f(cell) - myCellOffset + edgeToNodeOffset[edgeIndex][0]);
        Vec3f endNode = indexToWorld(Vec3f(cell) - myCellOffset + edgeToNodeOffset[edgeIndex][1]);

        startPoints.push_back(startNode);
        endPoints.push_back(endNode);
    }

    renderer.addLines(startPoints, endPoints, colour);
}

template <typename T>
void ScalarGrid<T>::drawGridPlane(Renderer& renderer, Axis planeAxis, float position) const
{
    position = clamp(position, float(0), float(1));

    Vec3i start(0);
    Vec3i end(this->mySize - Vec3i(1));

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

    forEachVoxelRange(start, end, [&](const Vec3i& cell) { drawGridCell(renderer, cell); });
}

template <typename T>
void ScalarGrid<T>::drawSamplePoints(Renderer& renderer, const Vec3f& colour, float sampleSize) const
{
    tbb::enumerable_thread_specific<std::vector<Vec3f>> parallelSamplePoints;

    tbb::parallel_for(tbb::blocked_range<int>(0, this->myGrid.size(), tbbLightGrainSize),
                      [&](const tbb::blocked_range<int>& range) {
                          auto& localSamplePoints = parallelSamplePoints.local();

                          for (int sampleIndex = range.begin(); sampleIndex != range.end(); ++sampleIndex)
                          {
                              Vec3i sampleCoord = this->unflatten(sampleIndex);

                              Vec3f worldPoint = indexToWorld(Vec3f(sampleCoord));

                              localSamplePoints.push_back(worldPoint);
                          }
                      });

    std::vector<Vec3f> samplePoints;
    mergeLocalThreadVectors(samplePoints, parallelSamplePoints);

    renderer.addPoints(samplePoints, colour, sampleSize);
}

template <typename T>
void ScalarGrid<T>::drawCellSamplePoint(Renderer& renderer, const Vec3i& cell, const Vec3f& colour,
                                        float sampleSize) const
{
    std::vector<Vec3f> samplePoints;

    Vec3f indexPoint(cell);
    Vec3f worldPoint;

    switch (mySampleType)
    {
        case SampleType::CENTER:
            worldPoint = indexToWorld(indexPoint);
            samplePoints.push_back(worldPoint);

            break;

        case SampleType::XFACE:
            worldPoint = indexToWorld(Vec3f(cellToFace(cell, 0 /*face axis*/, 0 /*direction*/)));
            samplePoints.push_back(worldPoint);

            worldPoint = indexToWorld(Vec3f(cellToFace(cell, 0 /*face axis*/, 1 /*direction*/)));
            samplePoints.push_back(worldPoint);

            break;

        case SampleType::YFACE:
            worldPoint = indexToWorld(Vec3f(cellToFace(cell, 1 /*axis*/, 0 /*direction*/)));
            samplePoints.push_back(worldPoint);

            worldPoint = indexToWorld(Vec3f(cellToFace(cell, 1 /*axis*/, 1 /*direction*/)));
            samplePoints.push_back(worldPoint);

            break;

        case SampleType::ZFACE:
            worldPoint = indexToWorld(Vec3f(cellToFace(cell, 2 /*axis*/, 0 /*direction*/)));
            samplePoints.push_back(worldPoint);

            worldPoint = indexToWorld(Vec3f(cellToFace(cell, 2 /*axis*/, 1 /*direction*/)));
            samplePoints.push_back(worldPoint);

            break;

        case SampleType::XEDGE:
            for (int edgeIndex = 0; edgeIndex < 4; ++edgeIndex)
            {
                worldPoint = indexToWorld(Vec3f(cellToEdge(cell, 0 /*edge axis*/, edgeIndex)));
                samplePoints.push_back(worldPoint);
            }

            break;

        case SampleType::YEDGE:
            for (int edgeIndex = 0; edgeIndex < 4; ++edgeIndex)
            {
                worldPoint = indexToWorld(Vec3f(cellToEdge(cell, 1 /*edge axis*/, edgeIndex)));
                samplePoints.push_back(worldPoint);
            }

            break;

        case SampleType::ZEDGE:
            for (int edgeIndex = 0; edgeIndex < 4; ++edgeIndex)
            {
                worldPoint = indexToWorld(Vec3f(cellToEdge(cell, 2 /*edge axis*/, edgeIndex)));
                samplePoints.push_back(worldPoint);
            }

            break;

        case SampleType::NODE:
            for (int nodeIndex = 0; nodeIndex < 8; ++nodeIndex)
            {
                worldPoint = indexToWorld(Vec3f(cellToNode(cell, nodeIndex)));
                samplePoints.push_back(worldPoint);
            }

            break;
    }

    renderer.addPoints(samplePoints, colour, sampleSize);
}

template <typename T>
void ScalarGrid<T>::drawSupersampledValues(Renderer& renderer, const Vec3f& start, const Vec3f& end,
                                           const Vec3f& sampleRadius, int samples, float sampleSize) const
{
    std::pair<T, T> minMaxPair = minAndMaxValue();
    T minSample = minMaxPair.first;
    T maxSample = minMaxPair.second;

    Vec3i startIndex(ceil(start));
    Vec3i endIndex(floor(end));

    forEachVoxelRange(startIndex, endIndex, [&](const Vec3i& cell) {
        // Supersample
        float dx = 2. * max(sampleRadius) / float(samples);
        Vec3f indexPoint(cell);
        Vec3f sampleOffset;
        for (sampleOffset[0] = -sampleRadius[0]; sampleOffset[0] <= sampleRadius[0]; sampleOffset[0] += dx)
            for (sampleOffset[1] = -sampleRadius[1]; sampleOffset[1] <= sampleRadius[1]; sampleOffset[1] += dx)
                for (sampleOffset[2] = -sampleRadius[2]; sampleOffset[2] <= sampleRadius[2]; sampleOffset[2] += dx)
                {
                    Vec3f samplePoint = indexPoint + sampleOffset;
                    Vec3f worldPoint = indexToWorld(samplePoint);

                    T value = (interp(worldPoint) - minSample) / (maxSample - minSample);

                    renderer.addPoint(worldPoint, Vec3f(value, value, 0), sampleSize);
                }
    });
}

// Warning: there is no protection here for ASSERT border types
template <typename T>
void ScalarGrid<T>::drawSupersampledValuesVolume(Renderer& renderer, float sampleRadius, int samples,
                                                 float sampleSize) const
{
    Vec3f start(0);
    Vec3f end = Vec3f(this->mySize) - Vec3f(1);

    drawSupersampledValues(renderer, start, end, Vec3f(sampleRadius), samples, sampleSize);
}

// Warning: there is no protection here for ASSERT border types
template <typename T>
void ScalarGrid<T>::drawSupersampledValuesPlane(Renderer& renderer, Axis planeAxis, float position, float sampleRadius,
                                                int samples, float sampleSize) const
{
    position = clamp(position, float(0), float(1));

    Vec3f start(0);
    Vec3f end(this->mySize - Vec3i(1));
    Vec3f localRadius(sampleRadius);

    if (planeAxis == Axis::XAXIS)
    {
        start[0] = std::floor(position * float(myGridSize[0] - 1));
        end[0] = start[0] + 1;

        localRadius[0] = 0;
    }
    else if (planeAxis == Axis::YAXIS)
    {
        start[1] = std::floor(position * float(myGridSize[1] - 1));
        end[1] = start[1] + 1;

        localRadius[1] = 0;
    }
    else if (planeAxis == Axis::ZAXIS)
    {
        start[2] = std::floor(position * float(myGridSize[2] - 1));
        end[2] = start[2] + 1;

        localRadius[2] = 0;
    }

    drawSupersampledValues(renderer, start, end, localRadius, samples, sampleSize);
}

template <typename T>
void ScalarGrid<T>::drawSampleGradientsPlane(Renderer& renderer, Axis planeAxis, float position, const Vec3f& colour,
                                             float length) const
{
    position = clamp(position, float(0), float(1));

    Vec3i start(0);
    Vec3i end = this->mySize - Vec3i(1);

    if (planeAxis == Axis::XAXIS)
    {
        int planeIndex = std::floor(position * float(myGridSize[0] - 1));
        start[0] = planeIndex;
        end[0] = planeIndex;
    }
    else if (planeAxis == Axis::YAXIS)
    {
        int planeIndex = std::floor(position * float(myGridSize[1] - 1));
        start[1] = planeIndex;
        end[1] = planeIndex;
    }
    else if (planeAxis == Axis::ZAXIS)
    {
        int planeIndex = std::floor(position * float(myGridSize[2] - 1));
        start[2] = planeIndex;
        end[2] = planeIndex;
    }

    std::vector<Vec3f> samplePoints;
    std::vector<Vec3f> gradientPoints;

    forEachVoxelRange(start, end, [&](const Vec3i& cell) {
        Vec3f worldPoint = indexToWorld(Vec3f(cell));
        samplePoints.push_back(worldPoint);

        Vec<3, T> gradVector = gradient(worldPoint);
        Vec3f vectorEnd = worldPoint + length * gradVector;
        gradientPoints.push_back(vectorEnd);
    });

    renderer.addLines(samplePoints, gradientPoints, colour);
}

}  // namespace FluidSim3D::Utilities

#endif