#ifndef TESTS_ANALYTICAL_POISSON_H
#define TESTS_ANALYTICAL_POISSON_H

#include <iostream>

#include <Eigen/Sparse>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

#include "Integrator.h"
#include "LevelSet.h"
#include "Renderer.h"
#include "ScalarGrid.h"
#include "Timer.h"
#include "Transform.h"
#include "Utilities.h"
#include "VectorGrid.h"

using namespace FluidSim3D;

class AnalyticalPoissonSolver
{
public:
    AnalyticalPoissonSolver(const Transform& xform, const Vec3i& size) : myXform(xform) { myPoissonGrid = ScalarGrid<double>(myXform, size, 0); }

    template <typename RHS, typename Solution>
    double solve(const RHS& rhsFunction, const Solution& solutionFunction);

    void drawGrid(Renderer& renderer) const;
    void drawValues(Renderer& renderer) const;

private:
    Transform myXform;
    ScalarGrid<double> myPoissonGrid;
};

template <typename RHS, typename Solution>
double AnalyticalPoissonSolver::solve(const RHS& rhsFuction, const Solution& solutionFunction)
{
    UniformGrid<int> solvableCells(myPoissonGrid.size(), -1);

    int solutionDOFCount = 0;

    Vec3i gridSize = myPoissonGrid.size();

    forEachVoxelRange(Vec3i::Zero(), gridSize, [&](const Vec3i& cell) { solvableCells(cell) = solutionDOFCount++; });

    std::vector<Eigen::Triplet<double>> sparseMatrixElements;
    sparseMatrixElements.reserve(gridSize[0] * gridSize[1] * gridSize[2]);

    VectorXd rhsVector = VectorXd::Zero(solutionDOFCount);

    double dx = myPoissonGrid.dx();
	double coeff = dx * dx;

    forEachVoxelRange(Vec3i::Zero(), gridSize, [&](const Vec3i& cell)
    {
        int row = solvableCells(cell);

        assert(row >= 0);

        // Build RHS
        Vec3d gridPoint = myPoissonGrid.indexToWorld(cell.cast<double>());

        rhsVector(row) = -coeff * rhsFuction(gridPoint);

        for (auto axis : {0, 1, 2})
            for (auto direction : {0, 1})
            {
                Vec3i adjacentCell = cellToCell(cell, axis, direction);

                // Bounds check. Use analytical solution for Dirichlet condition.
                if ((direction == 0 && adjacentCell[axis] < 0) || (direction == 1 && adjacentCell[axis] >= gridSize[axis]))
                {
                    Vec3d adjacentPoint = myPoissonGrid.indexToWorld(adjacentCell.cast<double>());
                    rhsVector(row) += solutionFunction(adjacentPoint);
                }
                else
                {
                    // If neighbouring cell is solvable, it should have an entry in the system
                    int adjacentRow = solvableCells(adjacentCell);
                    assert(adjacentRow >= 0);

                    sparseMatrixElements.emplace_back(row, adjacentRow, -1);
                }
            }
        sparseMatrixElements.emplace_back(row, row, 6);
    });

    SparseMatrix sparseMatrix(solutionDOFCount, solutionDOFCount);
    sparseMatrix.setFromTriplets(sparseMatrixElements.begin(), sparseMatrixElements.end());

	const double tolerance = 1E-6;

    Eigen::ConjugateGradient<SparseMatrix, Eigen::Upper | Eigen::Lower> cg_solver(sparseMatrix);

    cg_solver.setTolerance(tolerance);

    VectorXd solutionVector = cg_solver.solve(rhsVector);

    double error = 0;

    forEachVoxelRange(Vec3i::Zero(), gridSize, [&](const Vec3i& cell)
    {
        int row = solvableCells(cell);

        assert(row >= 0);

        Vec3d gridPoint = myPoissonGrid.indexToWorld(cell.cast<double>());
		double localError = fabs(solutionVector(row) - solutionFunction(gridPoint));

        if (error < localError) error = localError;

        myPoissonGrid(cell) = solutionVector(row);
    });

    return error;
}

#endif