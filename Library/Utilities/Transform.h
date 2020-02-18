#ifndef LIBRARY_TRANSFORM_H
#define LIBRARY_TRANSFORM_H

#include "Utilities.h"
#include "Vec.h"

///////////////////////////////////
//
// Transform.h
// Ryan Goldade 2017
//
// Simple transform container to simplify
// implementation for various transforms
// between grids, etc.
//
////////////////////////////////////

namespace FluidSim3D::Utilities
{

class Transform
{
public:
	Transform(float dx = 1., const Vec3f& offset = Vec3f(0))
		: myDx(dx)
		, myOffset(offset)
		{}

	Vec3f indexToWorld(const Vec3f& indexPoint) const
	{
		return indexPoint * myDx + myOffset;
	}

	Vec3f worldToIndex(const Vec3f& worldPoint) const
	{
		return (worldPoint - myOffset) / myDx;
	}

	float dx() const { return myDx; }
	Vec3f offset() const { return myOffset; }

	bool operator==(const Transform& rhs) const
	{
		if (myDx != rhs.myDx) return false;
		if (myOffset != rhs.myOffset) return false;
		return true;
	}

	bool operator!=(const Transform& rhs) const
	{
		if (myDx == rhs.myDx) return false;
		if (myOffset == rhs.myOffset) return false;
		return true;
	}

private:
	float myDx;
	Vec3f myOffset;
};
}
#endif