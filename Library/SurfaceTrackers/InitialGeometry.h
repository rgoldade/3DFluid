#ifndef FLUIDSIM3D_INITIAL_GEOMETRY_H
#define FLUIDSIM3D_INITIAL_GEOMETRY_H

#include "TriMesh.h"
#include "Utilities.h"

///////////////////////////////////
//
// InitialGeometry.h
// Ryan Goldade 2017
//
// List of initial surface configurations
// to speed up scene creation.
//
////////////////////////////////////

namespace FluidSim3D
{

TriMesh makeDiamondMesh(const Vec3d& center = Vec3d::Zero(), double scale = 1.);
TriMesh makeCubeMesh(const Vec3d& center = Vec3d::Zero(), const Vec3d& scale = Vec3d::Ones());
TriMesh makeIcosahedronMesh();
TriMesh makeSphereMesh(const Vec3d& center = Vec3d::Zero(), double radius = 1., int subdivisions = 5);

}

#endif