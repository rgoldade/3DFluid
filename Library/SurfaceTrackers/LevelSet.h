#ifndef LIBRARY_LEVEL_SET_H
#define LIBRARY_LEVEL_SET_H

#include "tbb\tbb.h"

#include "FieldAdvector.h"
#include "Predicates.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "TriMesh.h"
#include "Utilities.h"
#include "Vec.h"
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

namespace FluidSim3D::SurfaceTrackers
{

using namespace Utilities;

class LevelSet
{
public:
	LevelSet() : myPhiGrid() {}

	LevelSet(const Transform& xform, const Vec3i& size) : LevelSet(xform, size, size[0] * size[1] * size[2]) {}
	LevelSet(const Transform& xform, const Vec3i& size, int bandwidth, bool isBoundaryNegative = false)
		: myNarrowBand(float(bandwidth) * xform.dx())
		, myPhiGrid(xform, size, isBoundaryNegative ? -float(bandwidth) * xform.dx() : float(bandwidth) * xform.dx())
		, myIsBackgroundNegative(isBoundaryNegative)
	{
		for (int axis : {0,1,2})
			assert(size[axis] >= 0);

		// In order to deal with triangle meshes, we need to initialize
		// the geometric predicate library.
		exactinit();
	}

	void initFromMesh(const TriMesh& initialMesh, bool resizeGrid = true);

	void reinit();
	void reinitFIM();
	void reinitMesh()
	{
		TriMesh tempMesh = buildMesh();
		initFromMesh(tempMesh, false);
	}

	bool isGridMatched(const LevelSet& grid) const
	{
		if (size() != grid.size()) return false;
		if (xform() != grid.xform()) return false;
		return true;
	}

	bool isGridMatched(const ScalarGrid<float>& grid) const
	{
		if (grid.sampleType() != ScalarGridSettings::SampleType::CENTER) return false;
		if (size() != grid.size()) return false;
		if (xform() != grid.xform()) return false;
		return true;
	}

	void unionSurface(const LevelSet& unionPhi);

	bool isBackgroundNegative() const { return myIsBackgroundNegative; }
	void setBackgroundNegative() { myIsBackgroundNegative = true; }
	
	TriMesh buildMesh() const;

	template<typename VelocityField>
	void advectSurface(float dt, const VelocityField& velocity, IntegrationOrder order);

	Vec3f normal(const Vec3f& worldPoint) const
	{
		Vec3f normal = myPhiGrid.gradient(worldPoint);

		if (normal == Vec3f(0)) return Vec3f(0);

		return normalize(normal);
	}

	void clear() { myPhiGrid.clear(); }
	void resize(const Vec3i& size) { myPhiGrid.resize(size); }

	float narrowBand() { return myNarrowBand / dx(); }

	// There's no way to change the grid spacing inside the class.
	// The best way is to build a new grid and sample this one
	float dx() const { return myPhiGrid.dx(); }
	Vec3f offset() const { return myPhiGrid.offset(); }
	Transform xform() const { return myPhiGrid.xform(); }
	Vec3i size() const { return myPhiGrid.size(); }

	Vec3f indexToWorld(const Vec3f& indexPoint) const { return myPhiGrid.indexToWorld(indexPoint); }
	Vec3f worldToIndex(const Vec3f& worldPoint) const { return myPhiGrid.worldToIndex(worldPoint); }

	float interp(const Vec3f& worldPoint) const { return myPhiGrid.interp(worldPoint); }

	float& operator()(int i, int j, int k) { return myPhiGrid(i, j, k); }
	float& operator()(const Vec3i& cell) { return myPhiGrid(cell); }

	const float& operator()(int i, int j, int k) const { return myPhiGrid(i, j, k); }
	const float& operator()(const Vec3i& cell) const { return myPhiGrid(cell); }

	int voxelCount() const { return myPhiGrid.voxelCount(); }
	Vec3i unflatten(int cellIndex) const { return myPhiGrid.unflatten(cellIndex); }
	
	Vec3f findSurface(const Vec3f& worldPoint, int iterationLimit) const;

	// Interpolate the interface position between two nodes. This assumes
	// the caller has verified an interface (sign change) between the two.
	Vec3f interpolateInterface(const Vec3i& startPoint, const Vec3i& endPoint) const;

	void drawGrid(Renderer& renderer, bool doOnlyNarrowBand) const;

	void drawGridPlane(Renderer& renderer, Axis planeAxis, float position, bool doOnlyNarrowBand) const;

	// Display a supersampled slice of the grid. The plane will have a normal in the plane_axis direction.
	// The position is from [0,1] where 0 is at the grid origin and 1 is at the origin + size * dx.
	void drawSupersampledValuesPlane(Renderer& renderer, Axis planeAxis, float position, float radius = .5, int samples = 5, float sampleSize = 1) const;
	void drawSampleNormalsPlane(Renderer& renderer, Axis planeAxis, float position, const Vec3f& colour = Vec3f(.5), float length = .25) const;
	
	void drawSurface(Renderer& renderer, const Vec3f& colour = Vec3f(0.), float lineWidth = 1) const;

private:

	void reinitFastMarching(UniformGrid<VisitedCellLabels>& interfaceCells);
	void reinitFastIterative(UniformGrid<VisitedCellLabels>& interfaceCells);

	Vec3f findSurfaceIndex(const Vec3f& indexPoint, int iterationLimit = 10) const;

	ScalarGrid<float> myPhiGrid;

	// The narrow band of signed distances around the interface
	float myNarrowBand;

	bool myIsBackgroundNegative;
};



}

#endif