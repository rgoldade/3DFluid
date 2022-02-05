#ifndef FLUIDSIM3D_VISCOSITY_SOLVER_H
#define FLUIDSIM3D_VISCOSITY_SOLVER_H

#include "LevelSet.h"
#include "ScalarGrid.h"
#include "Utilities.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// ViscositySolver.h/cpp
// Ryan Goldade 2017
//
// Variational viscosity solver.
// Uses ghost fluid weights for moving
// collisions. Uses volume control
// weights for the various tensor and
// velocity sample positions.
//
////////////////////////////////////

namespace FluidSim3D
{

void ViscositySolver(double dt,
                        const LevelSet& surface,
                        VectorGrid<double>& velocity,
                        const LevelSet& solidSurface,
                        const VectorGrid<double>& solidVelocity,
                        const ScalarGrid<double>& viscosity);

}

#endif