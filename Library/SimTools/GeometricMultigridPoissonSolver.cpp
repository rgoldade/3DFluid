#include "GeometricMultigridPoissonSolver.h"

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

#include "GeometricMultigridOperators.h"

namespace FluidSim3D
{

using MGCellLabels = GeometricMultigridOperators::CellLabels;

void copyGridToVector(VectorXd& vector,
						const UniformGrid<double>& vectorGrid,
						const UniformGrid<int>& solverIndices,
						const UniformGrid<MGCellLabels>& cellLabels)
{
	assert(vectorGrid.size() == solverIndices.size() && solverIndices.size() == cellLabels.size());

	tbb::parallel_for(tbb::blocked_range<int>(0, cellLabels.voxelCount()), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec3i cell = cellLabels.unflatten(cellIndex);

			int index = solverIndices(cell);

			if (index >= 0)
			{
				assert(cellLabels(cell) == MGCellLabels::INTERIOR_CELL ||
					cellLabels(cell) == MGCellLabels::BOUNDARY_CELL);

				vector(index) = vectorGrid(cell);
			}
			else
			{
				assert(cellLabels(cell) != MGCellLabels::INTERIOR_CELL &&
					cellLabels(cell) != MGCellLabels::BOUNDARY_CELL);
			}
		}
	});
}

void copyVectorToGrid(UniformGrid<double>& vectorGrid,
						const VectorXd& vector,
						const UniformGrid<int>& solverIndices,
						const UniformGrid<MGCellLabels>& cellLabels)
{
	assert(vectorGrid.size() == solverIndices.size() &&
		solverIndices.size() == cellLabels.size());

	tbb::parallel_for(tbb::blocked_range<int>(0, cellLabels.voxelCount()), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec3i cell = cellLabels.unflatten(cellIndex);

			int index = solverIndices(cell);

			if (index >= 0)
			{
				assert(cellLabels(cell) == MGCellLabels::INTERIOR_CELL ||
					cellLabels(cell) == MGCellLabels::BOUNDARY_CELL);

				vectorGrid(cell) = vector(index);
			}
			else
			{
				assert(cellLabels(cell) != MGCellLabels::INTERIOR_CELL &&
					cellLabels(cell) != MGCellLabels::BOUNDARY_CELL);
			}
		}
	});
}

GeometricMultigridPoissonSolver::GeometricMultigridPoissonSolver(const UniformGrid<MGCellLabels>& initialDomainLabels,
																	const VectorGrid<double>& boundaryWeights,
																	int mgLevels,
																	double dx)
																	: myMGLevels(mgLevels)
																	, myBoundarySmootherWidth(3)
																	, myBoundarySmootherIterations(3)
{
	assert(mgLevels > 0 && dx > 0);

	assert(initialDomainLabels.size()[0] % 2 == 0 &&
		initialDomainLabels.size()[1] % 2 == 0 &&
		initialDomainLabels.size()[2] % 2 == 0);

	assert(int(std::log2(initialDomainLabels.size()[0])) + 1 >= mgLevels && 
		int(std::log2(initialDomainLabels.size()[1])) + 1 >= mgLevels &&
		int(std::log2(initialDomainLabels.size()[2])) + 1 >= mgLevels);

	myDomainLabels.resize(myMGLevels);
	myDomainLabels[0] = initialDomainLabels;

	assert(boundaryWeights.size(0)[0] - 1 == myDomainLabels[0].size()[0] &&
		boundaryWeights.size(0)[1] == myDomainLabels[0].size()[1] &&
		boundaryWeights.size(0)[2] == myDomainLabels[0].size()[2] &&

		boundaryWeights.size(1)[0] == myDomainLabels[0].size()[0] &&
		boundaryWeights.size(1)[1] - 1 == myDomainLabels[0].size()[1] &&
		boundaryWeights.size(1)[2] == myDomainLabels[0].size()[2] &&
	
		boundaryWeights.size(2)[0] == myDomainLabels[0].size()[0] &&
		boundaryWeights.size(2)[1] == myDomainLabels[0].size()[1] &&
		boundaryWeights.size(2)[2] - 1 == myDomainLabels[0].size()[2]);

	myFineBoundaryWeights = boundaryWeights;

	auto checkSolvableCell = [](const UniformGrid<MGCellLabels>& testGrid) -> bool
	{
		bool hasSolvableCell = false;

		tbb::parallel_for(tbb::blocked_range<int>(0, testGrid.voxelCount()), [&](const tbb::blocked_range<int>& range)
		{
			if (hasSolvableCell)
			{
				return;
			}

			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec3i cell = testGrid.unflatten(cellIndex);
			
				if (testGrid(cell) == MGCellLabels::INTERIOR_CELL ||
					testGrid(cell) == MGCellLabels::BOUNDARY_CELL)
				{
					hasSolvableCell = true;
				}
			}
		});

		return hasSolvableCell;
	};

	assert(checkSolvableCell(myDomainLabels[0]));
	assert(GeometricMultigridOperators::unitTestBoundaryCells(myDomainLabels[0], &myFineBoundaryWeights));
	assert(GeometricMultigridOperators::unitTestExteriorCells(myDomainLabels[0]));

	// Precompute the coarsening strategy. Cap level if there are no longer interior cells
	for (int level = 1; level < myMGLevels; ++level)
	{
		myDomainLabels[level] = buildCoarseCellLabels(myDomainLabels[level - 1]);

		if (!checkSolvableCell(myDomainLabels[level]))
		{
			myMGLevels = level - 1;
			myDomainLabels.resize(myMGLevels);
			break;
		}
		assert(GeometricMultigridOperators::unitTestCoarsening(myDomainLabels[level], myDomainLabels[level - 1]));
		assert(GeometricMultigridOperators::unitTestBoundaryCells(myDomainLabels[level]));
		assert(GeometricMultigridOperators::unitTestExteriorCells(myDomainLabels[0]));
	}

	myDx.resize(myMGLevels);
	myDx[0] = dx;

	for (int level = 1; level < myMGLevels; ++level)
	{
		myDx[level] = 2. * myDx[level - 1];
	}

	// Initialize solution vectors
	mySolutionGrids.resize(myMGLevels);
	myRHSGrids.resize(myMGLevels);
	myResidualGrids.resize(myMGLevels);

	for (int level = 0; level < myMGLevels; ++level)
	{
		mySolutionGrids[level] = UniformGrid<double>(myDomainLabels[level].size());
		myRHSGrids[level] = UniformGrid<double>(myDomainLabels[level].size());
		myResidualGrids[level] = UniformGrid<double>(myDomainLabels[level].size());
	}

	myBoundaryCells.resize(myMGLevels);
	
	for (int level = 0; level < myMGLevels; ++level)
	{
		myBoundaryCells[level] = buildBoundaryCells(myDomainLabels[level], myBoundarySmootherWidth);
	}

	// Pre-build matrix at the coarsest level
	{
		int interiorCellCount = 0;
		Vec3i coarsestSize = myDomainLabels[myMGLevels - 1].size();

		myDirectSolverIndices = UniformGrid<int>(coarsestSize, -1 /* unlabelled cell marker */);

		forEachVoxelRange(Vec3i::Zero(), coarsestSize, [&](const Vec3i& cell)
		{
			if (myDomainLabels[myMGLevels - 1](cell) == MGCellLabels::INTERIOR_CELL ||
				myDomainLabels[myMGLevels - 1](cell) == MGCellLabels::BOUNDARY_CELL)
			{
				myDirectSolverIndices(cell) = interiorCellCount++;
			}
		});

		// Build rows
		std::vector<Eigen::Triplet<double>> sparseElements;

		double gridScale = 1. / std::pow(myDx[myMGLevels - 1], 2);
		forEachVoxelRange(Vec3i::Zero(), coarsestSize, [&](const Vec3i& cell)
		{
			if (myDomainLabels[myMGLevels - 1](cell) == MGCellLabels::INTERIOR_CELL ||
				myDomainLabels[myMGLevels - 1](cell) == MGCellLabels::BOUNDARY_CELL)
			{
				int diagonal = 0;
				int index = myDirectSolverIndices(cell);
				assert(index >= 0);

				for (int axis : {0, 1, 2})
				{
					for (int direction : {0, 1})
					{
						Vec3i adjacentCell = cellToCell(cell, axis, direction);

						auto cellLabels = myDomainLabels[myMGLevels - 1](adjacentCell);
						if (cellLabels == MGCellLabels::INTERIOR_CELL ||
							cellLabels == MGCellLabels::BOUNDARY_CELL)
						{
							int adjacentIndex = myDirectSolverIndices(adjacentCell);
							assert(adjacentIndex >= 0);

							sparseElements.emplace_back(index, adjacentIndex, -gridScale);
							++diagonal;
						}
						else if (cellLabels == MGCellLabels::DIRICHLET_CELL)
						{
							++diagonal;
						}
					}
				}

				sparseElements.emplace_back(index, index, gridScale * diagonal);
			}
		});

		// Solve system
		mySparseMatrix = Eigen::SparseMatrix<double>(interiorCellCount, interiorCellCount);
		mySparseMatrix.setFromTriplets(sparseElements.begin(), sparseElements.end());
		mySparseMatrix.makeCompressed();

		myCoarseSolver.compute(mySparseMatrix);
		assert(myCoarseSolver.info() == Eigen::Success);
	}
}

void GeometricMultigridPoissonSolver::applyMGVCycle(UniformGrid<double>& fineSolutionGrid,
													const UniformGrid<double>& fineRHSGrid,
													bool useInitialGuess)
{
	using namespace GeometricMultigridOperators;

	assert(fineSolutionGrid.size() == fineRHSGrid.size() && fineRHSGrid.size() == myDomainLabels[0].size());

	// If there is an initial guess in the solution vector, copy it locally
	if (!useInitialGuess)
	{
		fineSolutionGrid.reset(0);
	}

	// Apply fine-level smoothing pass
	{
		for (int boundaryIteration = 0; boundaryIteration < myBoundarySmootherIterations; ++boundaryIteration)
		{
			boundaryJacobiPoissonSmoother(fineSolutionGrid,
											fineRHSGrid,
											myDomainLabels[0],
											myBoundaryCells[0],
											myDx[0],
											&myFineBoundaryWeights);
		}

		// Interior smoothing
		interiorJacobiPoissonSmoother(fineSolutionGrid,
										fineRHSGrid,
										myDomainLabels[0],
										myDx[0],
										&myFineBoundaryWeights);

		for (int boundaryIteration = 0; boundaryIteration < myBoundarySmootherIterations; ++boundaryIteration)
		{
			boundaryJacobiPoissonSmoother(fineSolutionGrid,
											fineRHSGrid,
											myDomainLabels[0],
											myBoundaryCells[0],
											myDx[0],
											&myFineBoundaryWeights);
		}

		myResidualGrids[0].reset(0);

		computePoissonResidual(myResidualGrids[0],
								fineSolutionGrid,
								fineRHSGrid,
								myDomainLabels[0],
								myDx[0],
								&myFineBoundaryWeights);

		myRHSGrids[1].reset(0);

		downsample(myRHSGrids[1],
					myResidualGrids[0],
					myDomainLabels[1],
					myDomainLabels[0]);
	}

	// Down-stroke of the v-cycle
	for (int level = 1; level < myMGLevels - 1; ++level)
	{
		mySolutionGrids[level].reset(0);

		for (int boundaryIteration = 0; boundaryIteration < myBoundarySmootherIterations * (1 << level); ++boundaryIteration)
		{
			boundaryJacobiPoissonSmoother(mySolutionGrids[level],
											myRHSGrids[level],
											myDomainLabels[level],
											myBoundaryCells[level],
											myDx[level]);
		}

		// Interior smoothing
		interiorJacobiPoissonSmoother(mySolutionGrids[level],
										myRHSGrids[level],
										myDomainLabels[level],
										myDx[level]);

		for (int boundaryIteration = 0; boundaryIteration < myBoundarySmootherIterations * (1 << level); ++boundaryIteration)
		{
			boundaryJacobiPoissonSmoother(mySolutionGrids[level],
											myRHSGrids[level],
											myDomainLabels[level],
											myBoundaryCells[level],
											myDx[level]);
		}

		myResidualGrids[level].reset(0);

		// Compute residual to restrict to the next level
		computePoissonResidual(myResidualGrids[level],
								mySolutionGrids[level],
								myRHSGrids[level],
								myDomainLabels[level],
								myDx[level]);

		myRHSGrids[level + 1].reset(0);

		downsample(myRHSGrids[level + 1],
					myResidualGrids[level],
					myDomainLabels[level + 1],
					myDomainLabels[level]);
	}

	VectorXd coarseRHSVector = VectorXd::Zero(mySparseMatrix.rows());
	
	copyGridToVector(coarseRHSVector,
						myRHSGrids[myMGLevels - 1],
						myDirectSolverIndices,
						myDomainLabels[myMGLevels - 1]);

	VectorXd directSolution = myCoarseSolver.solve(coarseRHSVector);

	mySolutionGrids[myMGLevels - 1].reset(0);
	
	copyVectorToGrid(mySolutionGrids[myMGLevels - 1],
						directSolution,
						myDirectSolverIndices,
						myDomainLabels[myMGLevels - 1]);

	// Up-stroke of the v-cycle
	for (int level = myMGLevels - 2; level >= 1; --level)
	{
		upsampleAndAdd(mySolutionGrids[level],
						mySolutionGrids[level + 1],
						myDomainLabels[level],
						myDomainLabels[level + 1]);

		for (int boundaryIteration = 0; boundaryIteration < myBoundarySmootherIterations * (1 << level); ++boundaryIteration)
		{
			boundaryJacobiPoissonSmoother(mySolutionGrids[level],
											myRHSGrids[level],
											myDomainLabels[level],
											myBoundaryCells[level],
											myDx[level]);
		}

		interiorJacobiPoissonSmoother(mySolutionGrids[level],
										myRHSGrids[level],
										myDomainLabels[level],
										myDx[level]);

		for (int boundaryIteration = 0; boundaryIteration < myBoundarySmootherIterations * (1 << level); ++boundaryIteration)
		{
			boundaryJacobiPoissonSmoother(mySolutionGrids[level],
											myRHSGrids[level],
											myDomainLabels[level],
											myBoundaryCells[level],
											myDx[level]);
		}
	}

	// Apply fine-level smoother
	{
		upsampleAndAdd(fineSolutionGrid,
						mySolutionGrids[1],
						myDomainLabels[0],
						myDomainLabels[1]);

		for (int boundaryIteration = 0; boundaryIteration < myBoundarySmootherIterations; ++boundaryIteration)
		{
			boundaryJacobiPoissonSmoother(fineSolutionGrid,
											fineRHSGrid,
											myDomainLabels[0],
											myBoundaryCells[0],
											myDx[0],
											&myFineBoundaryWeights);
		}

		interiorJacobiPoissonSmoother(fineSolutionGrid,
										fineRHSGrid,
										myDomainLabels[0],
										myDx[0],
										&myFineBoundaryWeights);

		for (int boundaryIteration = 0; boundaryIteration < myBoundarySmootherIterations; ++boundaryIteration)
		{
			boundaryJacobiPoissonSmoother(fineSolutionGrid,
											fineRHSGrid,
											myDomainLabels[0],
											myBoundaryCells[0],
											myDx[0],
											&myFineBoundaryWeights);
		}
	}
}

}