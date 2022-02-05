#include <iostream>

#include <Eigen/Sparse>

#include <gtest/gtest.h>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"

#include "Integrator.h"
#include "LevelSet.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "Utilities.h"
#include "VectorGrid.h"

using namespace FluidSim3D;

class AnalyticalPoissonSolver
{
public:
    AnalyticalPoissonSolver(const Transform& xform, const Vec3i& size) 
		: myXform(xform) 
	{
		myPoissonGrid = ScalarGrid<float>(myXform, size, 0);
	}

    template <typename RHS, typename Solution>
	double solve(const RHS& rhsFunction, const Solution& solutionFunction);

private:
    Transform myXform;
    ScalarGrid<float> myPoissonGrid;
};

template <typename RHS, typename Solution>
double AnalyticalPoissonSolver::solve(const RHS& rhsFuction, const Solution& solutionFunction)
{
    UniformGrid<int> solvableCells(myPoissonGrid.size(), -1);

    int solutionDOFCount = 0;

    Vec3i gridSize = myPoissonGrid.size();

    forEachVoxelRange(Vec3i::Zero(), gridSize, [&](const Vec3i& cell) { solvableCells(cell) = solutionDOFCount++; });

    std::vector<Eigen::Triplet<double>> sparseMatrixElements;

    VectorXd rhsVector = VectorXd::Zero(solutionDOFCount);

    double dx = myPoissonGrid.dx();
	double coeff = std::pow(dx, 2);

	{
		tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<double>>> parallelSparseElements;
		tbb::parallel_for(tbb::blocked_range<int>(0, solvableCells.voxelCount()), [&](const tbb::blocked_range<int>& range)
		{
        	auto& localSparseElements = parallelSparseElements.local();

			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec3i cell = solvableCells.unflatten(cellIndex);
				int row = solvableCells(cell);

				assert(row >= 0);

				// Build RHS
				Vec3d gridPoint = Vec3d(myPoissonGrid.indexToWorld(cell.cast<double>()));

				rhsVector(row) = -coeff * rhsFuction(gridPoint);

				for (auto axis : {0, 1, 2})
					for (auto direction : {0, 1})
					{
						Vec3i adjacentCell = cellToCell(cell, axis, direction);

						// Bounds check. Use analytical solution for Dirichlet condition.
						if ((direction == 0 && adjacentCell[axis] < 0) || (direction == 1 && adjacentCell[axis] >= gridSize[axis]))
						{
							Vec3d adjacentPoint = Vec3d(myPoissonGrid.indexToWorld(adjacentCell.cast<double>()));
							rhsVector(row) += solutionFunction(adjacentPoint);
						}
						else
						{
							// If neighbouring cell is solvable, it should have an entry in the system
							int adjacentRow = solvableCells(adjacentCell);
							assert(adjacentRow >= 0);

							localSparseElements.emplace_back(row, adjacentRow, -1);
						}
					}
				localSparseElements.emplace_back(row, row, 6);
			}
		});

		mergeLocalThreadVectors(sparseMatrixElements, parallelSparseElements);
	}

    SparseMatrix sparseMatrix(solutionDOFCount, solutionDOFCount);
    sparseMatrix.setFromTriplets(sparseMatrixElements.begin(), sparseMatrixElements.end());

    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Upper | Eigen::Lower> solver;
    solver.compute(sparseMatrix);

    if (solver.info() != Eigen::Success)
    {
        return -1;
    }

    VectorXd solutionVector = solver.solve(rhsVector);

    if (solver.info() != Eigen::Success)
    {
        return -1;
    }

	double error = tbb::parallel_reduce(tbb::blocked_range<int>(0, solvableCells.voxelCount()), double(0),
					[&](const tbb::blocked_range<int>& range, double error) -> double
					{
						for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
						{
							Vec3i cell = solvableCells.unflatten(cellIndex);
							
							int row = solvableCells(cell);

							assert(row >= 0);
							
							Vec3d gridPoint = myPoissonGrid.indexToWorld(cell.cast<double>());
							double localError = fabs(solutionVector(row) - solutionFunction(gridPoint));
							error = std::max(error, localError);
						}

						return error;
					},
					[](double a, double b) -> double {
						return std::max(a, b);
					});

    return error;
}

TEST(ANALYTICAL_POISSON_SOLVER_TEST, CONVERGENCE_TEST)
{
	auto rhs = [](const Vec3d& pos) -> double
	{
		return 3. * std::exp(-pos[0] - pos[1] - pos[2]);
	};

	auto solution = [](const Vec3d& pos) -> double
	{
		return std::exp(-pos[0] - pos[1] - pos[2]);
	};

    const int startGrid = 4;
    const int endGrid = startGrid * int(pow(2, 4));
	
    std::vector<double> errors;
    for (int gridSize = startGrid; gridSize < endGrid; gridSize *= 2)
    {
        double dx = PI / double(gridSize);
		Vec3d origin = Vec3d::Zero();
		Vec3i size = Vec3i::Constant(int(std::round(PI / dx)));
		Transform xform(dx, origin);

		AnalyticalPoissonSolver solver(xform, size);
		double error = solver.solve(rhs, solution);

	    errors.push_back(error);
        EXPECT_GT(error, 0.);
	}

    for (int errorIndex = 1; errorIndex < errors.size(); ++errorIndex)
    {
        double errorRatio = errors[errorIndex - 1] / errors[errorIndex];
        EXPECT_GT(errorRatio, 4.);
    }
}