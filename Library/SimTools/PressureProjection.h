#ifndef LIBRARY_PRESSURE_PROJECTION_H
#define LIBRARY_PRESSURE_PROJECTION_H

#include <Eigen/Sparse>

#include "LevelSet.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "Utilities.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// PressureProjection.h/cpp
// Ryan Goldade 2017
//
// Variational pressure solve. Allows
// for moving solids.
//
////////////////////////////////////

namespace FluidSim3D::SimTools
{

using namespace SurfaceTrackers;
using namespace Utilities;

class PressureProjection
{
	using SolveReal = double;
	using Vector = Eigen::VectorXd;

public:

	PressureProjection(const LevelSet& surface,
						const VectorGrid<float>& cutCellWeights,
						const VectorGrid<float>& ghostFluidWeights,
						const VectorGrid<float>& solidVelocity);

	void project(VectorGrid<float>& velocity);

	void setInitialGuess(const ScalarGrid<float>& initialGuessPressure)
	{
		assert(mySurface.isGridMatched(initialGuessPressure));
		myUseInitialGuessPressure = true;
		myInitialGuessPressure = &initialGuessPressure;
	}

	void disableInitialGuess()
	{
		myUseInitialGuessPressure = false;
	}

	ScalarGrid<float> getPressureGrid()
	{
		return myPressure;
	}

	const VectorGrid<VisitedCellLabels>& getValidFaces()
	{
		return myValidFaces;
	}

private:

	const VectorGrid<float>& mySolidVelocity;
	const VectorGrid<float>& myGhostFluidWeights;
	const VectorGrid<float>& myCutCellWeights;

	// Store flags for solved faces
	VectorGrid<VisitedCellLabels> myValidFaces;

	const LevelSet& mySurface;

	ScalarGrid<float> myPressure;

	const ScalarGrid<float> *myInitialGuessPressure;
	bool myUseInitialGuessPressure;
};

}

#endif