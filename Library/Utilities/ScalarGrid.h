#ifndef FLUIDSIM3D_SCALAR_GRID_H
#define FLUIDSIM3D_SCALAR_GRID_H

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

#include "GridUtilities.h"
#include "Transform.h"
#include "UniformGrid.h"
#include "Utilities.h"

#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"

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

namespace FluidSim3D
{

// Separating the grid settings out from the grid class
// means that we don't have to deal with templating when
// referencing these settings.
namespace ScalarGridSettings
{
enum class BorderType
{
    CLAMP,
    ZERO
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
}

template <typename T>
class ScalarGrid : public UniformGrid<T>
{
    using BorderType = ScalarGridSettings::BorderType;
    using SampleType = ScalarGridSettings::SampleType;

public:
    ScalarGrid() : myXform(1., Vec3d::Zero()), myGridSize(Vec3i::Zero()), UniformGrid<T>() {}

    ScalarGrid(const Transform& xform, const Vec3i& size, SampleType sampleType = SampleType::CENTER,
               BorderType borderType = BorderType::CLAMP)
        : ScalarGrid(xform, size, T(0), sampleType, borderType)
    {}

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
                myCellOffset = Vec3d::Constant(.5);
                this->resize(size, initialValue);
                break;
            case SampleType::XFACE:
                myCellOffset = Vec3d(.0, .5, .5);
                this->resize(size + Vec3i(1, 0, 0), initialValue);
                break;
            case SampleType::YFACE:
                myCellOffset = Vec3d(.5, .0, .5);
                this->resize(size + Vec3i(0, 1, 0), initialValue);
                break;
            case SampleType::ZFACE:
                myCellOffset = Vec3d(.5, .5, .0);
                this->resize(size + Vec3i(0, 0, 1), initialValue);
                break;
            case SampleType::XEDGE:
                myCellOffset = Vec3d(.5, .0, .0);
                this->resize(size + Vec3i(0, 1, 1), initialValue);
                break;
            case SampleType::YEDGE:
                myCellOffset = Vec3d(.0, .5, .0);
                this->resize(size + Vec3i(1, 0, 1), initialValue);
                break;
            case SampleType::ZEDGE:
                myCellOffset = Vec3d(.0, .0, .5);
                this->resize(size + Vec3i(1, 1, 0), initialValue);
                break;
            case SampleType::NODE:
                myCellOffset = Vec3d::Zero();
                this->resize(size + Vec3i::Ones(), initialValue);
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
    void operator*=(const T& scalar)
    {
        this->myGrid.array() *= scalar;
    }

    // Global add operator
    void operator+=(const T& scalar)
    {
        this->myGrid.array() += scalar;
    }

    T maxValue() const
    {
        return this->myGrid.maxCoeff();
    }

    T minValue() const
    {
        return this->myGrid.minCoeff();
    }

    std::pair<T, T> minAndMaxValue() const
    {
        return std::make_pair<T, T>(minValue(), maxValue());
    }

    T triLerp(double x, double y, double z, bool isIndexSpace = false) const
    {
        return triLerp(Vec3d(x, y, z), isIndexSpace);
    }
    T triLerp(const Vec3d& samplePoint, bool isIndexSpace = false) const;

    Vec3t<T> triLerpGradient(const Vec3d& worldPoint, bool isIndexSpace = false) const;

    // Converters between world space and local index space
    Vec3d indexToWorld(const Vec3d& indexPoint) const { return myXform.indexToWorld(indexPoint + myCellOffset); }

    Vec3d worldToIndex(const Vec3d& worldPoint) const { return myXform.worldToIndex(worldPoint) - myCellOffset; }

    double dx() const { return myXform.dx(); }
    const Vec3d& offset() const { return myXform.offset(); }
    const Transform& xform() const { return myXform; }

    // Polyscope render methods
    void drawGrid(const std::string& label) const;
    void drawGridCell(const std::string& label, const Vec3i& cell) const;
    void drawGridCellList(const std::string& label, const VecVec3i& cells) const;
    void drawGridPlane(const std::string& label, Axis planeAxis, double position) const;

    void drawSamplePoints(const std::string& label, const Vec3d& colour = Vec3d(1, 0, 0), double sampleSize = .001) const;
    void drawCellSamplePoint(const std::string& label, const Vec3i& cell, const Vec3d& colour = Vec3d(1, 0, 0),
                             double sampleSize = .001) const;

    void drawSupersampledValues(const std::string& label, const Vec3d& start, const Vec3d& end, const Vec3d& sampleRadius,
                                int samples, double sampleSize = .001) const;
    void drawSupersampledValuesVolume(const std::string& label, double sampleRadius = .5, int samples = 5,
                                      double sampleSize = .001) const;

    void drawSupersampledValuesPlane(const std::string& label, Axis planeAxis, double position, double sampleRadius = .5,
                                     int samples = 5, double sampleSize = .001) const;
    void drawSampleGradientsPlane(const std::string& label, Axis planeAxis, double position, const Vec3d& colour = Vec3d::Constant(.5),
                                  double length = .25) const;

private:
    // The main interpolation call after the template specialized clamping passes
    T triLerpLocal(const Vec3d& pos) const;

    Vec3t<T> triLerpGradientLocal(const Vec3d& pos) const;

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
    Vec3d myCellOffset;
};

template <typename T>
T ScalarGrid<T>::triLerp(const Vec3d& samplePoint, bool isIndexSpace) const
{
    Vec3d indexPoint = isIndexSpace ? samplePoint : worldToIndex(samplePoint);

    switch (myBorderType)
    {
        case BorderType::ZERO:

            for (int axis : {0, 1, 2})
            {
                if (indexPoint[axis] < 0 || indexPoint[axis] > double(this->mySize[axis] - 1)) return T(0);
            }

            break;

        case BorderType::CLAMP:

            for (int axis : {0, 1, 2})
                indexPoint[axis] = std::clamp(indexPoint[axis], double(0), double(this->mySize[axis] - 1));

            break;
    }

    return triLerpLocal(indexPoint);
}

// The local interp applies tri-linear interpolation on the UniformGrid. The
// templated type must have operators for basic add/mult arithmetic.
template <typename T>
T ScalarGrid<T>::triLerpLocal(const Vec3d& indexPoint) const
{
    Vec3d floorPoint = indexPoint.array().floor();
    Vec3i baseSampleCell = floorPoint.cast<int>();

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

    Vec3d dx = indexPoint - floorPoint;

    for (int axis : {0, 1, 2}) assert(dx[axis] >= 0 && dx[axis] <= 1);

    return trilerp<T, double>(v000, v100, v010, v110, v001, v101, v011, v111, dx[0], dx[1], dx[2]);
}

template<typename T>
Vec3t<T> ScalarGrid<T>::triLerpGradient(const Vec3d& samplePoint, bool isIndexSpace) const
{
	Vec3d indexPoint = isIndexSpace ? samplePoint : worldToIndex(samplePoint);

	switch (myBorderType)
	{
		case BorderType::ZERO:
		{
			for (int axis : {0, 1, 1})
			{
				if (indexPoint[axis] < 0 || indexPoint[axis] > double(this->mySize[axis] - 1))
					return Vec3t<T>::Zero();
			}

			break;
		}
		case BorderType::CLAMP:
		{
			for (int axis : {0, 1, 2})
				indexPoint[axis] = std::clamp(indexPoint[axis], double(0), double(this->mySize[axis] - 1));

			break;
		}
	}

	return triLerpGradientLocal(indexPoint);
}

template<typename T>
Vec3t<T> ScalarGrid<T>::triLerpGradientLocal(const Vec3d& samplePoint) const
{
	Vec3d floorPoint = samplePoint.array().floor();
	Vec3i baseSampleCell = floorPoint.cast<int>();

	for (int axis : {0, 1, 2})
	{
		if (baseSampleCell[axis] == this->mySize[axis] - 1)
			--baseSampleCell[axis];
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

	Vec3d deltaX = samplePoint - floorPoint;

	for (int axis : {0, 1, 2})
		assert(deltaX[axis] >= 0 && deltaX[axis] <= 1);
	
	return trilerpGradient<T, double>(v000, v100, v010, v110, v001, v101, v011, v111, deltaX[0], deltaX[1], deltaX[2]) / dx();
}

template <typename T>
void ScalarGrid<T>::drawGrid(const std::string& label) const
{
    VecVec3d segments;

    for (int axis : {0, 1, 2})
    {
        Vec3i start = Vec3i::Zero();
        Vec3i end = myGridSize;
        end[axis] = 1;

        forEachVoxelRange(start, end, [&](const Vec3i& cell)
        {
            Vec3d gridStart = cell.cast<double>();
            segments.push_back(indexToWorld(gridStart - myCellOffset));

            Vec3d gridEnd = cell.cast<double>();
            gridEnd[axis] = myGridSize[axis];
            segments.push_back(indexToWorld(gridEnd - myCellOffset));
        });
    }

    auto* cn = polyscope::registerCurveNetworkSegments(label + " grid", segments);
    cn->setColor(glm::vec3(0.f, 0.f, 0.f));
    cn->setRadius(.0005);
}

template <typename T>
void ScalarGrid<T>::drawGridCell(const std::string& label, const Vec3i& cell) const
{
    drawGridCellList(label, {cell});
}

template <typename T>
void ScalarGrid<T>::drawGridCellList(const std::string& label, const VecVec3i& cells) const
{
    static const Vec3d edgeToNodeOffset[12][2] = {
        {Vec3d(0, 0, 0), Vec3d(1, 0, 0)},
        {Vec3d(0, 0, 1), Vec3d(1, 0, 1)}, {Vec3d(0, 1, 0), Vec3d(1, 1, 0)}, {Vec3d(0, 1, 1), Vec3d(1, 1, 1)},
        {Vec3d(0, 0, 0), Vec3d(0, 1, 0)},
        {Vec3d(1, 0, 0), Vec3d(1, 1, 0)}, {Vec3d(0, 0, 1), Vec3d(0, 1, 1)}, {Vec3d(1, 0, 1), Vec3d(1, 1, 1)},
        {Vec3d(0, 0, 0), Vec3d(0, 0, 1)},
        {Vec3d(1, 0, 0), Vec3d(1, 0, 1)}, {Vec3d(0, 1, 0), Vec3d(0, 1, 1)}, {Vec3d(1, 1, 0), Vec3d(1, 1, 1)}};

    VecVec3d segments;

    for (const Vec3i& cell : cells)
    {
        for (int edgeIndex = 0; edgeIndex < 12; ++edgeIndex)
        {
            segments.push_back(indexToWorld(cell.cast<double>() - myCellOffset + edgeToNodeOffset[edgeIndex][0]));
            segments.push_back(indexToWorld(cell.cast<double>() - myCellOffset + edgeToNodeOffset[edgeIndex][1]));
        }
    }

    if (!segments.empty())
    {
        auto* cn = polyscope::registerCurveNetworkSegments(label, segments);
        cn->setColor(glm::vec3(0.f, 0.f, 0.f));
        cn->setRadius(.00025);
    }
}

template <typename T>
void ScalarGrid<T>::drawGridPlane(const std::string& label, Axis planeAxis, double position) const
{
    position = std::clamp(position, double(0), double(1));

    Vec3i start = Vec3i::Zero();
    Vec3i end(this->mySize - Vec3i::Ones());

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

    VecVec3i cells;
    forEachVoxelRange(start, end, [&](const Vec3i& cell) { cells.push_back(cell); });

    drawGridCellList(label + " grid plane", cells);
}

template <typename T>
void ScalarGrid<T>::drawSamplePoints(const std::string& label, const Vec3d& colour, double sampleSize) const
{
    tbb::enumerable_thread_specific<VecVec3d> parallelSamplePoints;

    tbb::parallel_for(tbb::blocked_range<int>(0, this->myGrid.size(), tbbLightGrainSize),
                      [&](const tbb::blocked_range<int>& range) {
                          auto& localSamplePoints = parallelSamplePoints.local();

                          for (int sampleIndex = range.begin(); sampleIndex != range.end(); ++sampleIndex)
                          {
                              Vec3i sampleCoord = this->unflatten(sampleIndex);
                              Vec3d worldPoint = indexToWorld(sampleCoord.cast<double>());
                              localSamplePoints.push_back(worldPoint);
                          }
                      });

    VecVec3d samplePoints;
    mergeLocalThreadVectors(samplePoints, parallelSamplePoints);

    auto* pc = polyscope::registerPointCloud(label + " sample points", samplePoints);
    pc->setPointColor(glm::vec3((float)colour[0], (float)colour[1], (float)colour[2]));
    pc->setPointRadius(sampleSize);
}

template <typename T>
void ScalarGrid<T>::drawCellSamplePoint(const std::string& label, const Vec3i& cell, const Vec3d& colour,
                                        double sampleSize) const
{
    VecVec3d samplePoints;
    Vec3d worldPoint;

    switch (mySampleType)
    {
        case SampleType::CENTER:
            samplePoints.push_back(indexToWorld(cell.cast<double>()));
            break;
        case SampleType::XFACE:
            samplePoints.push_back(indexToWorld(cellToFace(cell, 0, 0).cast<double>()));
            samplePoints.push_back(indexToWorld(cellToFace(cell, 0, 1).cast<double>()));
            break;
        case SampleType::YFACE:
            samplePoints.push_back(indexToWorld(cellToFace(cell, 1, 0).cast<double>()));
            samplePoints.push_back(indexToWorld(cellToFace(cell, 1, 1).cast<double>()));
            break;
        case SampleType::ZFACE:
            samplePoints.push_back(indexToWorld(cellToFace(cell, 2, 0).cast<double>()));
            samplePoints.push_back(indexToWorld(cellToFace(cell, 2, 1).cast<double>()));
            break;
        case SampleType::XEDGE:
            for (int edgeIndex = 0; edgeIndex < 4; ++edgeIndex)
                samplePoints.push_back(indexToWorld(cellToEdge(cell, 0, edgeIndex).cast<double>()));
            break;
        case SampleType::YEDGE:
            for (int edgeIndex = 0; edgeIndex < 4; ++edgeIndex)
                samplePoints.push_back(indexToWorld(cellToEdge(cell, 1, edgeIndex).cast<double>()));
            break;
        case SampleType::ZEDGE:
            for (int edgeIndex = 0; edgeIndex < 4; ++edgeIndex)
                samplePoints.push_back(indexToWorld(cellToEdge(cell, 2, edgeIndex).cast<double>()));
            break;
        case SampleType::NODE:
            for (int nodeIndex = 0; nodeIndex < 8; ++nodeIndex)
                samplePoints.push_back(indexToWorld(cellToNode(cell, nodeIndex).cast<double>()));
            break;
    }

    auto* pc = polyscope::registerPointCloud(label + " cell sample", samplePoints);
    pc->setPointColor(glm::vec3((float)colour[0], (float)colour[1], (float)colour[2]));
    pc->setPointRadius(sampleSize);
}

template <typename T>
void ScalarGrid<T>::drawSupersampledValues(const std::string& label, const Vec3d& start, const Vec3d& end,
                                           const Vec3d& sampleRadius, int samples, double sampleSize) const
{
    std::pair<T, T> minMaxPair = minAndMaxValue();
    T minSample = minMaxPair.first;
    T maxSample = minMaxPair.second;

    Vec3i startIndex(ceil(start).cast<int>());
    Vec3i endIndex(floor(end).cast<int>());

    VecVec3d pts;
    std::vector<glm::vec3> colors;

    forEachVoxelRange(startIndex, endIndex, [&](const Vec3i& cell)
    {
        double dx = 2. * sampleRadius.maxCoeff() / double(samples);
        Vec3d indexPoint = cell.cast<double>();
        Vec3d sampleOffset;
        for (sampleOffset[0] = -sampleRadius[0]; sampleOffset[0] <= sampleRadius[0]; sampleOffset[0] += dx)
            for (sampleOffset[1] = -sampleRadius[1]; sampleOffset[1] <= sampleRadius[1]; sampleOffset[1] += dx)
                for (sampleOffset[2] = -sampleRadius[2]; sampleOffset[2] <= sampleRadius[2]; sampleOffset[2] += dx)
                {
                    Vec3d samplePoint = indexPoint + sampleOffset;
                    Vec3d worldPoint = indexToWorld(samplePoint);

                    T value = (triLerp(worldPoint) - minSample) / (maxSample - minSample);

                    pts.push_back(worldPoint);
                    colors.push_back(glm::vec3((float)value, (float)value, 0.f));
                }
    });

    auto* pc = polyscope::registerPointCloud(label + " supersampled values", pts);
    pc->setPointRadius(sampleSize);
    pc->addColorQuantity("values", colors)->setEnabled(true);
}

template <typename T>
void ScalarGrid<T>::drawSupersampledValuesVolume(const std::string& label, double sampleRadius, int samples,
                                                 double sampleSize) const
{
    Vec3d start = Vec3d::Zero();
    Vec3d end = this->mySize.template cast<double>() - Vec3d::Ones();

    drawSupersampledValues(label, start, end, Vec3d::Constant(sampleRadius), samples, sampleSize);
}

template <typename T>
void ScalarGrid<T>::drawSupersampledValuesPlane(const std::string& label, Axis planeAxis, double position, double sampleRadius,
                                                int samples, double sampleSize) const
{
    position = std::clamp(position, double(0), double(1));

    Vec3d start = Vec3d::Zero();
    Vec3d end = (this->mySize - Vec3i::Ones()).template cast<double>();
    Vec3d localRadius = Vec3d::Constant(sampleRadius);

    if (planeAxis == Axis::XAXIS)
    {
        start[0] = std::floor(position * double(myGridSize[0] - 1));
        end[0] = start[0] + 1;
        localRadius[0] = 0;
    }
    else if (planeAxis == Axis::YAXIS)
    {
        start[1] = std::floor(position * double(myGridSize[1] - 1));
        end[1] = start[1] + 1;
        localRadius[1] = 0;
    }
    else if (planeAxis == Axis::ZAXIS)
    {
        start[2] = std::floor(position * double(myGridSize[2] - 1));
        end[2] = start[2] + 1;
        localRadius[2] = 0;
    }

    drawSupersampledValues(label + " plane", start, end, localRadius, samples, sampleSize);
}

template <typename T>
void ScalarGrid<T>::drawSampleGradientsPlane(const std::string& label, Axis planeAxis, double position, const Vec3d& colour,
                                             double length) const
{
    position = std::clamp(position, double(0), double(1));

    Vec3i start = Vec3i::Zero();
    Vec3i end = this->mySize - Vec3i::Ones();

    if (planeAxis == Axis::XAXIS)
    {
        int planeIndex = int(std::floor(position * double(myGridSize[0] - 1)));
        start[0] = planeIndex;
        end[0] = planeIndex;
    }
    else if (planeAxis == Axis::YAXIS)
    {
        int planeIndex = int(std::floor(position * double(myGridSize[1] - 1)));
        start[1] = planeIndex;
        end[1] = planeIndex;
    }
    else if (planeAxis == Axis::ZAXIS)
    {
        int planeIndex = int(std::floor(position * double(myGridSize[2] - 1)));
        start[2] = planeIndex;
        end[2] = planeIndex;
    }

    VecVec3d segments;

    forEachVoxelRange(start, end, [&](const Vec3i& cell)
    {
        Vec3d worldPoint = indexToWorld(cell.cast<double>());
        Vec3t<T> gradVector = triLerpGradient(worldPoint);
        Vec3d vectorEnd = worldPoint + length * gradVector;

        segments.push_back(worldPoint);
        segments.push_back(vectorEnd);
    });

    auto* cn = polyscope::registerCurveNetworkSegments(label + " gradients", segments);
    cn->setColor(glm::vec3((float)colour[0], (float)colour[1], (float)colour[2]));
    cn->setRadius(.0005);
}

}

#endif