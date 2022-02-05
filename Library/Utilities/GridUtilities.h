#ifndef FLUIDSIM3D_GRID_UTILITIES_H
#define FLUIDSIM3D_GRID_UTILITIES_H

#include "Utilities.h"

namespace FluidSim3D
{
enum class Axis
{
    XAXIS,
    YAXIS,
    ZAXIS
};

FORCE_INLINE Vec3i cellToCell(const Vec3i& cell, int axis, int direction)
{
    Vec3i adjacentCell(cell);

    if (direction == 0)
        --adjacentCell[axis];
    else
    {
        assert(direction == 1);
        ++adjacentCell[axis];
    }

    return adjacentCell;
}

FORCE_INLINE Vec3i cellToFace(const Vec3i& cell, int axis, int direction)
{
    Vec3i face(cell);

    if (direction == 1)
        ++face[axis];
    else
        assert(direction == 0);

    return face;
}

FORCE_INLINE Vec3i cellToEdge(const Vec3i& cell, int edgeAxis, int edgeIndex)
{
    assert(edgeAxis >= 0 && edgeAxis < 3);
    assert(edgeIndex >= 0 && edgeIndex < 4);

    Vec3i edge(cell);

    for (int axisOffset : {0, 1})
    {
        if (edgeIndex & (1 << axisOffset))
        {
            int localAxis = (edgeAxis + 1 + axisOffset) % 3;
            ++edge[localAxis];
        }
    }

    return edge;
}

FORCE_INLINE Vec3i cellToNode(const Vec3i& cell, int nodeIndex)
{
    assert(nodeIndex >= 0 && nodeIndex < 8);

    Vec3i node(cell);

    for (int axis : {0, 1, 2})
    {
        if (nodeIndex & (1 << axis)) ++node[axis];
    }

    return node;
}

FORCE_INLINE Vec3i faceToCell(const Vec3i& face, int axis, int direction)
{
    Vec3i cell(face);

    if (direction == 0)
        --cell[axis];
    else
        assert(direction == 1);

    return cell;
}

FORCE_INLINE Vec3i faceToEdge(const Vec3i& face, int faceAxis, int edgeAxis, int direction)
{
    assert(faceAxis >= 0 && faceAxis < 3 && edgeAxis >= 0 && edgeAxis < 3);
    assert(faceAxis != edgeAxis);

    Vec3i edge(face);
    if (direction == 1)
    {
        int offsetAxis = 3 - faceAxis - edgeAxis;
        ++edge[offsetAxis];
    }
    else
        assert(direction == 0);

    return edge;
}

FORCE_INLINE Vec3i faceToNode(const Vec3i& face, int faceAxis, int nodeIndex)
{
    assert(faceAxis >= 0 && faceAxis < 3);
    assert(nodeIndex >= 0 && nodeIndex < 4);

    Vec3i node(face);
    for (int axisOffset : {0, 1})
    {
        if (nodeIndex & (1 << axisOffset))
        {
            int localAxis = (faceAxis + 1 + axisOffset) % 3;
            ++node[localAxis];
        }
    }

    return node;
}

FORCE_INLINE Vec3i faceToNodeCCW(const Vec3i& face, int faceAxis, int nodeIndex)
{
    const Vec3i faceToNodeOffsets[3][4] = {{Vec3i(0, 0, 0), Vec3i(0, 1, 0), Vec3i(0, 1, 1), Vec3i(0, 0, 1)},
                                           {Vec3i(0, 0, 0), Vec3i(0, 0, 1), Vec3i(1, 0, 1), Vec3i(1, 0, 0)},
                                           {Vec3i(0, 0, 0), Vec3i(1, 0, 0), Vec3i(1, 1, 0), Vec3i(0, 1, 0)}};

    assert(faceAxis >= 0 && faceAxis < 3);
    assert(nodeIndex >= 0 && nodeIndex < 4);

    Vec3i node(face);
    node += faceToNodeOffsets[faceAxis][nodeIndex];

    return node;
}

FORCE_INLINE Vec3i edgeToFace(const Vec3i& edge, int edgeAxis, int faceAxis, int direction)
{
    assert(faceAxis >= 0 && faceAxis < 3 && edgeAxis >= 0 && edgeAxis < 3);
    assert(faceAxis != edgeAxis);

    Vec3i face(edge);
    if (direction == 0)
    {
        int offsetAxis = 3 - faceAxis - edgeAxis;
        --face[offsetAxis];
    }
    else
        assert(direction == 1);

    return face;
}

FORCE_INLINE Vec3i edgeToCell(const Vec3i& edge, int edgeAxis, int cellIndex)
{
    assert(edgeAxis >= 0 && edgeAxis < 3);
    assert(cellIndex >= 0 && cellIndex < 4);

    Vec3i cell(edge);
    for (int axisOffset : {0, 1})
    {
        if (!(cellIndex & (1 << axisOffset)))
        {
            int localAxis = (edgeAxis + 1 + axisOffset) % 3;
            --cell[localAxis];
        }
    }

    return cell;
}

FORCE_INLINE Vec3i edgeToCellCCW(const Vec3i& edge, int edgeAxis, int cellIndex)
{
    const Vec3i edgeToCellOffsets[3][4] = {{Vec3i(0, -1, -1), Vec3i(0, 0, -1), Vec3i(0, 0, 0), Vec3i(0, -1, 0)},
                                           {Vec3i(-1, 0, -1), Vec3i(-1, 0, 0), Vec3i(0, 0, 0), Vec3i(0, 0, -1)},
                                           {Vec3i(-1, -1, 0), Vec3i(0, -1, 0), Vec3i(0, 0, 0), Vec3i(-1, 0, 0)}};

    assert(edgeAxis >= 0 && edgeAxis < 3);
    assert(cellIndex >= 0 && cellIndex < 4);

    Vec3i cell(edge);
    cell += edgeToCellOffsets[edgeAxis][cellIndex];

    return cell;
}

FORCE_INLINE Vec3i edgeToNode(const Vec3i& edge, int axis, int direction)
{
    Vec3i node(edge);
    if (direction == 1)
        ++node[axis];
    else
        assert(direction == 0);

    return node;
}

FORCE_INLINE Vec3i nodeToFace(const Vec3i& node, int faceAxis, int faceIndex)
{
    assert(faceAxis >= 0 && faceAxis < 3);
    assert(faceIndex >= 0 && faceIndex < 4);

    Vec3i face(node);
    for (int axisOffset : {0, 1})
    {
        if (!(faceIndex & (1 << axisOffset)))
        {
            int localAxis = (faceAxis + 1 + axisOffset) % 3;
            --face[localAxis];
        }
    }

    return face;
}

FORCE_INLINE Vec3i nodeToCell(const Vec3i& node, int cellIndex)
{
    assert(cellIndex >= 0 && cellIndex < 8);

    Vec3i cell(node);
    for (int axis : {0, 1, 2})
    {
        if (!(cellIndex & (1 << axis))) --cell[axis];
    }

    return cell;
}

const Vec3d colours[] = {Vec3d(1, 0, 0), Vec3d(0, 1, 0), Vec3d(0, 0, 1),
                         Vec3d(1, 1, 0), Vec3d(1, 0, 1), Vec3d(0, 1, 1)};

template <typename Real>
Real lengthFraction(Real phi0, Real phi1)
{
    Real theta = 0.;

    if (phi0 <= 0)
    {
        if (phi1 <= 0)
            theta = 1;
        else// if (phi1 > 0)
            theta = phi0 / (phi0 - phi1);
    }
    else if (phi1 <= 0)
        theta = phi1 / (phi1 - phi0);

    return theta;
}

// Execute function "f" over range [start, end)
template <typename Function>
void forEachVoxelRange(const Vec3i& start, const Vec3i& end, const Function& f)
{
    Vec3i cell;
    for (cell[0] = start[0]; cell[0] != end[0]; ++cell[0])
        for (cell[1] = start[1]; cell[1] != end[1]; ++cell[1])
            for (cell[2] = start[2]; cell[2] != end[2]; ++cell[2])
                f(cell);
}

template <typename Function>
void forEachVoxelRangeReverse(const Vec3i& start, const Vec3i& end, const Function& f)
{
    Vec3i cell;
    for (cell[0] = end[0] - 1; cell[0] >= start[0]; --cell[0])
        for (cell[1] = end[1] - 1; cell[1] >= start[1]; --cell[1])
            for (cell[2] = end[2] - 1; cell[2] >= start[2]; --cell[2])
                f(cell);
}

}

#endif