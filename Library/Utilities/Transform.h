#ifndef FLUIDSIM3D_TRANSFORM_H
#define FLUIDSIM3D_TRANSFORM_H

#include "Utilities.h"

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

namespace FluidSim3D
{
class Transform
{
public:
    Transform(double dx = 1., const Vec3d& offset = Vec3d::Zero()) : myDx(dx), myOffset(offset) {}

    Vec3d indexToWorld(const Vec3d& indexPoint) const { return indexPoint * myDx + myOffset; }

    Vec3d worldToIndex(const Vec3d& worldPoint) const { return (worldPoint - myOffset) / myDx; }

    double dx() const { return myDx; }
    const Vec3d& offset() const { return myOffset; }

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
    double myDx;
    Vec3d myOffset;
};

}
#endif