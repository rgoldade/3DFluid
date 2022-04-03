#ifndef FLUIDSIM3D_GEOMETRIC_MULTIGRID_POISSONSOLVER_H
#define FLUIDSIM3D_GEOMETRIC_MULTIGRID_POISSONSOLVER_H

#include "Eigen/Sparse"

#include "GeometricMultigridOperators.h"
#include "UniformGrid.h"
#include "Utilities.h"
#include "VectorGrid.h"

namespace FluidSim3D
{

class GeometricMultigridPoissonSolver
{
	using MGCellLabels = GeometricMultigridOperators::CellLabels;

public:
	GeometricMultigridPoissonSolver() : myMGLevels(0) {};
	GeometricMultigridPoissonSolver(const UniformGrid<MGCellLabels>& initialDomainLabels,
									const VectorGrid<double>& boundaryWeights,
									int mgLevels,
									double dx);

	void applyMGVCycle(UniformGrid<double>& solutionVector,
						const UniformGrid<double>& rhsVector,
						bool useInitialGuess = false);

private:

	std::vector<UniformGrid<MGCellLabels>> myDomainLabels;
	std::vector<UniformGrid<double>> mySolutionGrids, myRHSGrids, myResidualGrids;

	std::vector<VecVec3i> myBoundaryCells;

	int myMGLevels;

	VectorGrid<double> myFineBoundaryWeights;

	std::vector<double> myDx;

	int myBoundarySmootherIterations;
	int myBoundarySmootherWidth;

	UniformGrid<int> myDirectSolverIndices;

	Eigen::SparseMatrix<double> mySparseMatrix;
	Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> myCoarseSolver;
};

}

#endif