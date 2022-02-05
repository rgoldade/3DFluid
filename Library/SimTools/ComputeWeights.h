#ifndef FLUIDSIM3D_COMPUTE_WEIGHTS_H
#define FLUIDSIM3D_COMPUTE_WEIGHTS_H

#include "LevelSet.h"
#include "ScalarGrid.h"
#include "Utilities.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// ComputeWeights.h/cpp
// Ryan Goldade 2017
//
// Useful collection of tools to compute
// control volume weights for use in both
// pressure projection and viscosity solves.
//
////////////////////////////////////

namespace FluidSim3D
{

VectorGrid<double> computeGhostFluidWeights(const LevelSet& surface);

VectorGrid<double> computeCutCellWeights(const LevelSet& surface, bool invertWeights = false);

void computeSupersampleVolumes(ScalarGrid<double>& volumes, const LevelSet& surface, int samples);

VectorGrid<double> computeSupersampledFaceVolumes(const LevelSet& surface, int samples);

}

#endif