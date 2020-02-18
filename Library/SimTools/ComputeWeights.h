#ifndef LIBRARY_COMPUTEWEIGHTS_H
#define LIBRARY_COMPUTEWEIGHTS_H

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

namespace FluidSim3D::SimTools
{

VectorGrid<float> computeGhostFluidWeights(const LevelSet& surface);

VectorGrid<float> computeCutCellWeights(const LevelSet& surface, bool invertWeights = false);

void computeSupersampleVolumes(ScalarGrid<float>& volumes, const LevelSet& surface, int samples);

VectorGrid<float> computeSupersampledFaceVolumes(const LevelSet& surface, int samples);

}

#endif