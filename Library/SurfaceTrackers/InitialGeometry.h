#ifndef LIBRARY_INITIAL_GEOMETRY_H
#define LIBRARY_INITIAL_GEOMETRY_H

#include "TriMesh.h"
#include "Utilities.h"
#include "Vec.h"

///////////////////////////////////
//
// InitialGeometry.h
// Ryan Goldade 2017
//
// List of initial surface configurations
// to speed up scene creation.
//
////////////////////////////////////

namespace FluidSim3D::SurfaceTrackers
{

using namespace Utilities;

TriMesh makeDiamondMesh(const Vec3f& center = Vec3f(0.), float scale = 1.);
TriMesh makeCubeMesh(const Vec3f& center = Vec3f(0), const Vec3f& scale = Vec3f(1.));
TriMesh makeSphereMesh(const Vec3f& center = Vec3f(0), float radius = 1., float dx = .1);

}

#endif