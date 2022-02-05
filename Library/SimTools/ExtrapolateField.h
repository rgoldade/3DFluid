#ifndef FLUIDSIM3D_EXTRAPOLATE_FIELD_H
#define FLUIDSIM3D_EXTRAPOLATE_FIELD_H

#include "tbb/blocked_range.h"
#include "tbb/parallel_sort.h"

#include "UniformGrid.h"
#include "Utilities.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// ExtrapolateField.h/cpp
// Ryan Goldade 2017
//
// Extrapolates field from the boundary
// of a mask outward based on a simple
// BFS flood fill approach. Values
// are averaged from FINISHED neighbours.
// Note that because this process happens
// "in order" there could be some bias
// based on which boundary locations
// are inserted into the queue first.
//
//
////////////////////////////////////

namespace FluidSim3D
{

template <typename Field>
void extrapolateField(Field& field, UniformGrid<VisitedCellLabels> finishedCellMask, int bandwidth)
{
    assert(bandwidth > 0);
    assert(field.size() == finishedCellMask.size());

    // Build an initial list of cells adjacent to finished cells in the provided mask grid
    VecVec3i toVisitCells;

    tbb::enumerable_thread_specific<VecVec3i> parallelToVisitCells;

    tbb::parallel_for(tbb::blocked_range<int>(0, finishedCellMask.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range) 
    {
        auto& localToVisitCells = parallelToVisitCells.local();
        for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
        {
            Vec3i cell = finishedCellMask.unflatten(cellIndex);

            // Load up adjacent unfinished cells
            if (finishedCellMask(cell) == VisitedCellLabels::FINISHED_CELL)
            {
                for (int axis : {0, 1, 2})
                    for (int direction : {0, 1})
                    {
                        Vec3i adjacentCell = cellToCell(cell, axis, direction);

                        if (adjacentCell[axis] < 0 || adjacentCell[axis] >= finishedCellMask.size()[axis]) continue;

                        if (finishedCellMask(adjacentCell) != VisitedCellLabels::FINISHED_CELL)
                            localToVisitCells.push_back(adjacentCell);
                    }
            }
        }
    });

    mergeLocalThreadVectors(toVisitCells, parallelToVisitCells);

    auto vecCompare = [](const Vec3i& vec0, const Vec3i& vec1)
    {
        for (int i = 0; i < 3; ++i)
        {
            if (vec0[i] < vec1[i])
                return true;
            else if (vec0[i] > vec1[i])
                return false;
        }

        return false;
    };

    // Now flood outwards layer-by-layer
    for (int layer = 0; layer < bandwidth; ++layer)
    {
        // First sort the list because there could be duplicates
        tbb::parallel_sort(toVisitCells.begin(), toVisitCells.end(), vecCompare);

        // Compute values from adjacent finished cells
        tbb::parallel_for(tbb::blocked_range<int>(0, int(toVisitCells.size()), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
        {
            // Because the list could contain duplicates, we need to advance forward through possible duplicates
            int cellIndex = range.begin();

            if (cellIndex > 0)
            {
                while (cellIndex < toVisitCells.size() && toVisitCells[cellIndex] == toVisitCells[cellIndex - 1])
                    ++cellIndex;
            }

            Vec3i oldCell = Vec3i::Constant(-1);

            for (; cellIndex < range.end(); ++cellIndex)
            {
                Vec3i cell = toVisitCells[cellIndex];

                if (cell == oldCell) continue;

                oldCell = cell;

                assert(finishedCellMask(cell) != VisitedCellLabels::FINISHED_CELL);

                // TODO: get template type from field instead of assuming double is valid
                double accumulatedValue = 0;
                double accumulatedCount = 0;

                for (int axis : {0, 1, 2})
                    for (int direction : {0, 1})
                    {
                        Vec3i adjacentCell = cellToCell(cell, axis, direction);

                        if (adjacentCell[axis] < 0 || adjacentCell[axis] >= finishedCellMask.size()[axis]) continue;

                        if (finishedCellMask(adjacentCell) == VisitedCellLabels::FINISHED_CELL)
                        {
                            accumulatedValue += field(adjacentCell);
                            ++accumulatedCount;
                        }
                    }

                assert(accumulatedCount > 0);

                field(cell) = accumulatedValue / accumulatedCount;
            }
        });

        // Set visited cells to finished
        tbb::parallel_for(tbb::blocked_range<int>(0, int(toVisitCells.size()), tbbLightGrainSize),
            [&](const tbb::blocked_range<int>& range)
            {
                // Because the list could contain duplicates, we need to advance forward through possible duplicates
                int cellIndex = range.begin();

                if (cellIndex > 0)
                {
                    while (cellIndex < toVisitCells.size() && toVisitCells[cellIndex] == toVisitCells[cellIndex - 1])
                        ++cellIndex;
                }

                Vec3i oldCell = Vec3i::Constant(-1);

                for (; cellIndex < range.end(); ++cellIndex)
                {
                    Vec3i cell = toVisitCells[cellIndex];

                    if (cell == oldCell) continue;

                    oldCell = cell;

                    assert(finishedCellMask(cell) != VisitedCellLabels::FINISHED_CELL);
                    finishedCellMask(cell) = VisitedCellLabels::FINISHED_CELL;
                }
            });

        // Build new layer of cells
        if (layer < bandwidth - 1)
        {
            parallelToVisitCells.clear();

            tbb::parallel_for(tbb::blocked_range<int>(0, int(toVisitCells.size()), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
            {
                auto& localToVisitCells = parallelToVisitCells.local();

                // Because the list could contain duplicates, we need to advance forward through possible duplicates
                int cellIndex = range.begin();

                if (cellIndex > 0)
                {
                    while (cellIndex < toVisitCells.size() && toVisitCells[cellIndex] == toVisitCells[cellIndex - 1])
                        ++cellIndex;
                }

                Vec3i oldCell = Vec3i::Constant(-1);

                for (; cellIndex < range.end(); ++cellIndex)
                {
                    Vec3i cell = toVisitCells[cellIndex];

                    if (cell == oldCell) continue;

                    oldCell = cell;

                    assert(finishedCellMask(cell) == VisitedCellLabels::FINISHED_CELL);

                    for (int axis : {0, 1, 2})
                        for (int direction : {0, 1})
                        {
                            Vec3i adjacentCell = cellToCell(cell, axis, direction);

                            if (adjacentCell[axis] < 0 || adjacentCell[axis] >= finishedCellMask.size()[axis])
                                continue;

                            if (finishedCellMask(adjacentCell) != VisitedCellLabels::FINISHED_CELL)
                                localToVisitCells.push_back(adjacentCell);
                        }
                }
            });

            toVisitCells.clear();
            mergeLocalThreadVectors(toVisitCells, parallelToVisitCells);
        }
    }
}

}

#endif