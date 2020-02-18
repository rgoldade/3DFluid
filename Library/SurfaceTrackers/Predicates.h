#ifndef LIBRARY_PREDICATES_H
#define LIBRARY_PREDICATES_H

#include "GridUtilities.h"
#include "Vec.h"

namespace FluidSim3D::SurfaceTrackers
{
using namespace Utilities;

using REAL = float;
using Vec3R = Vec<3, REAL>;

REAL
exactinit(); // call this before anything else

REAL
orient2d(const REAL *pa,
		 const REAL *pb,
		 const REAL *pc);

REAL
orient3d(const REAL *pa,
		 const REAL *pb,
		 const REAL *pc,
		 const REAL *pd);

REAL
incircle(const REAL *pa,
		 const REAL *pb,
		 const REAL *pc,
		 const REAL *pd);

REAL
insphere(const REAL *pa,
		 const REAL *pb,
		 const REAL *pc,
		 const REAL *pd,
		 const REAL *pe);

enum class IntersectionLabels { YES, ON, NO };

// An inner helper function that tests for an intersection between the z-axis grid edge
// (a ray cast in the positive z-direction from "a") and the triangle "qrs". The triangle
// is assumed to wind CCW on its projected xy-plane. The idea behind this helper is to catch 
// when there is a definite intersection as the outer loop has to handle an additional degeneracy
// before returning a true/false for the entire intersection test
bool zCastIntersection(const Vec3R& a, const Vec3R& q, const Vec3R& r, const Vec3R& s);

IntersectionLabels exactTriIntersect(Vec3R a, Vec3R q, Vec3R r, Vec3R s, Axis axis);

}

#endif
