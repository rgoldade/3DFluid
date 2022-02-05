#ifndef FLUIDSIM3D_UNIFORM_GRID_H
#define FLUIDSIM3D_UNIFORM_GRID_H

#include <vector>

#include "Utilities.h"

///////////////////////////////////
//
// UniformGrid.h/cpp
// Ryan Goldade 2017
//
// Uniform 3-D grid class that stores templated
// values at grid centers. Any positioned-based
// storage here must be accounted for by the
// caller.
//
////////////////////////////////////

namespace FluidSim3D
{
template <typename T>
class UniformGrid
{
public:
    UniformGrid() : mySize(Vec3i::Zero()) {}

    UniformGrid(const Vec3i& size) : mySize(size)
    {
        for (int axis : {0, 1, 2}) assert(size[axis] >= 0);

        myGrid.resize(mySize[0] * mySize[1] * mySize[2]);
    }

    UniformGrid(const Vec3i& size, const T& value) : mySize(size)
    {
        for (int axis : {0, 1, 2}) assert(size[axis] >= 0);

        myGrid.resize(mySize[0] * mySize[1] * mySize[2]);
        myGrid.array() = value;
    }

    // Accessor is z-major because the inside loop for most processes is naturally z. Should give better cache
    // coherence. Clamping should only occur for interpolation. Direct index access that's outside of the grid should be
    // a sign of an error.
    T& operator()(int i, int j, int k) { return (*this)(Vec3i(i, j, k)); }

    T& operator()(const Vec3i& coord)
    {
        for (int axis : {0, 1, 2}) assert(coord[axis] >= 0 && coord[axis] < mySize[axis]);

        return myGrid[flatten(coord)];
    }

    const T& operator()(int i, int j, int k) const { return (*this)(Vec3i(i, j, k)); }

    const T& operator()(const Vec3i& coord) const
    {
        for (int axis : {0, 1, 2}) assert(coord[axis] >= 0 && coord[axis] < mySize[axis]);

        return myGrid[flatten(coord)];
    }

    void clear()
    {
        mySize.setZero();
        myGrid.resize(0);
    }

    bool empty() const { return myGrid.rows() == 0; }

    void resize(const Vec3i& newSize)
    {
        for (int axis : {0, 1, 2}) assert(newSize[axis] >= 0);

        mySize = newSize;
        myGrid.resize(mySize[0] * mySize[1] * mySize[2]);
    }

    void resize(const Vec3i& newSize, const T& value)
    {
        for (int axis : {0, 1, 2}) assert(newSize[axis] >= 0);

        mySize = newSize;
        myGrid.resize(mySize[0] * mySize[1] * mySize[2]);
        myGrid.array() = value;
    }

    void reset(const T& value)
    {
        myGrid.array() = value;;
    }

    const Vec3i& size() const { return mySize; }
    int voxelCount() const { return mySize[0] * mySize[1] * mySize[2]; }

    FORCE_INLINE int flatten(const Vec3i& coord) const { return coord[2] + mySize[2] * coord[1] + mySize[2] * mySize[1] * coord[0]; }

    FORCE_INLINE Vec3i unflatten(int index) const
    {
        assert(index >= 0 && index < voxelCount());

        Vec3i coord;
        coord[2] = index % mySize[2];

        index -= coord[2];
        index /= mySize[2];

        coord[1] = index % mySize[1];

        index -= coord[1];
        index /= mySize[1];

        coord[0] = index;

        return coord;
    }

protected:
    VectorXt<T> myGrid;
    Vec3i mySize;
};

}

#endif