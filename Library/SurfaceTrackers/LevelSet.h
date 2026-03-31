#ifndef FLUIDSIM3D_LEVELSET_H
#define FLUIDSIM3D_LEVELSET_H

#include <string>

#include "FieldAdvector.h"
#include "Predicates.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "TriMesh.h"
#include "Utilities.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// LevelSet.h/cpp
// Ryan Goldade 2017
//
// 3-D level set surface tracker.
// Uses a simple dense grid but with
// a narrow band for most purposes.
// Redistancing performs an interface
// search for nodes near the zero crossing
// and then fast marching to update the
// remaining grid (w.r.t. narrow band).
//
////////////////////////////////////

namespace FluidSim3D
{

class LevelSet
{
public:
    LevelSet();
    LevelSet(const Transform& xform, const Vec3i& size);
    LevelSet(const Transform& xform, const Vec3i& size, double bandwidth, bool isBoundaryNegative = false);

    void initFromMesh(const TriMesh& initialMesh, bool resizeGrid = true);

    void reinit();

    bool isGridMatched(const LevelSet& grid) const;
    bool isGridMatched(const ScalarGrid<double>& grid) const;

    void unionSurface(const LevelSet& unionPhi, bool doReinit = true);

    bool isBackgroundNegative() const;
    void setBackgroundNegative();

    TriMesh buildMesh() const;

    template <typename VelocityField>
    void advectSurface(double dt, const VelocityField& velocity, IntegrationOrder order);

    FORCE_INLINE Vec3d normal(const Vec3d& worldPoint, bool useLinearInterp = true) const
    {
        Vec3d normal;

        if (useLinearInterp)
        {
            normal = myPhiGrid.triLerpGradient(worldPoint);
        }
        else
        {
            normal = myPhiGrid.triCubicGradient(worldPoint);
        }

		if ((normal.array() == Vec3d::Zero().array()).all()) return Vec3d::Zero();

		return normal.normalized();
    }

    void clear();
    void resize(const Vec3i& size);

    FORCE_INLINE double narrowBand() { return myNarrowBand; }

    // There's no way to change the grid spacing inside the class.
    // The best way is to build a new grid and sample this one
    FORCE_INLINE double dx() const { return myPhiGrid.dx(); }
    FORCE_INLINE const Vec3d& offset() const { return myPhiGrid.offset(); }
    FORCE_INLINE const Transform& xform() const { return myPhiGrid.xform(); }
    FORCE_INLINE const Vec3i& size() const { return myPhiGrid.size(); }

    FORCE_INLINE Vec3d indexToWorld(const Vec3d& indexPoint) const { return myPhiGrid.indexToWorld(indexPoint); }
    FORCE_INLINE Vec3d worldToIndex(const Vec3d& worldPoint) const { return myPhiGrid.worldToIndex(worldPoint); }

    FORCE_INLINE double triLerp(const Vec3d& worldPoint) const { return myPhiGrid.triLerp(worldPoint); }

    FORCE_INLINE double triCubicInterp(const Vec3d& worldPoint) const { return myPhiGrid.triCubicInterp(worldPoint); }

    FORCE_INLINE double& operator()(int i, int j, int k) { return myPhiGrid(i, j, k); }
    FORCE_INLINE double& operator()(const Vec3i& cell) { return myPhiGrid(cell); }

    FORCE_INLINE const double& operator()(int i, int j, int k) const { return myPhiGrid(i, j, k); }
    FORCE_INLINE const double& operator()(const Vec3i& cell) const { return myPhiGrid(cell); }

    FORCE_INLINE int voxelCount() const { return myPhiGrid.voxelCount(); }
    FORCE_INLINE Vec3i unflatten(int cellIndex) const { return myPhiGrid.unflatten(cellIndex); }

    Vec3d findSurface(const Vec3d& worldPoint, int iterationLimit, double tolerance) const;

    // Interpolate the interface position between two nodes. This assumes
    // the caller has verified an interface (sign change) between the two.
    Vec3d interpolateInterface(const Vec3i& startPoint, const Vec3i& endPoint) const;

    void drawGrid(const std::string& label, bool doOnlyNarrowBand) const;

    void drawGridPlane(const std::string& label, Axis planeAxis, double position, bool doOnlyNarrowBand) const;

    void drawSupersampledValuesPlane(const std::string& label, Axis planeAxis, double position, double radius = .5,
                                     int samples = 5, double sampleSize = .001) const;
    void drawSampleNormalsPlane(const std::string& label, Axis planeAxis, double position, const Vec3d& colour = Vec3d::Constant(.5),
                                double length = .25) const;

    void drawSurface(const std::string& label, const Vec3d& colour = Vec3d::Zero()) const;

private:

    void initFromMeshImpl(const TriMesh& initialMesh, bool doResizeGrid);

    void reinitFastMarching(UniformGrid<VisitedCellLabels>& interfaceCells);

    Vec3d findSurfaceIndex(const Vec3d& indexPoint, int iterationLimit, double tolerance) const;

    // The narrow band of signed distances around the interface
    double myNarrowBand;

    bool myIsBackgroundNegative;

    ScalarGrid<double> myPhiGrid;
};

template<typename VelocityField>
void LevelSet::advectSurface(double dt, const VelocityField& velocity, IntegrationOrder order)
{
	ScalarGrid<double> tempPhiGrid = myPhiGrid;
	advectField(dt, tempPhiGrid, myPhiGrid, velocity, order, InterpolationOrder::CUBIC);

	std::swap(tempPhiGrid, myPhiGrid);
}

}

#endif