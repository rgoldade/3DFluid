#include <Eigen/Core>

#include "tbb/tbb.h"

#include "LevelSet.h"
#include "PressureProjection.h"

namespace FluidSim3D::SimTools
{

PressureProjection::PressureProjection(const LevelSet& surface,
										const VectorGrid<float>& cutCellWeights,
										const VectorGrid<float>& ghostFluidWeights,
										const VectorGrid<float>& solidVelocity)
: mySurface(surface)
, myCutCellWeights(cutCellWeights)
, myGhostFluidWeights(ghostFluidWeights)
, mySolidVelocity(solidVelocity)
, myUseInitialGuessPressure(false)
, myInitialGuessPressure(nullptr)
{
	// For efficiency sake, this should only take in velocity on a staggered grid
	// that matches the center sampled surface and collision

	assert(solidVelocity.sampleType() == VectorGridSettings::SampleType::STAGGERED);

#if !defined(NDEBUG)
	for (int axis : {0, 1, 2})
	{
		Vec3i faceCount = solidVelocity.size(axis);

		Vec3i cellSize = faceCount;
		--cellSize[axis];

		assert(cellSize == surface.size());
	}
#endif

	assert(solidVelocity.isGridMatched(cutCellWeights) &&
			solidVelocity.isGridMatched(ghostFluidWeights));

	myPressure = ScalarGrid<float>(surface.xform(), surface.size(), 0);
	myValidFaces = VectorGrid<VisitedCellLabels>(surface.xform(), surface.size(), VisitedCellLabels::UNVISITED_CELL, VectorGridSettings::SampleType::STAGGERED);
}

void PressureProjection::project(VectorGrid<float>& velocity)
{
	assert(velocity.isGridMatched(mySolidVelocity));

	enum class MaterialLabels { SOLID_CELL, AIR_CELL, LIQUID_CELL };

	UniformGrid<MaterialLabels> materialCellLabels(mySurface.size(), MaterialLabels::SOLID_CELL);

	tbb::parallel_for(tbb::blocked_range<int>(0, materialCellLabels.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec3i cell = materialCellLabels.unflatten(cellIndex);

			bool isFluidCell = false;

			for (int axis = 0; axis < 3 && !isFluidCell; ++axis)
				for (int direction : {0, 1})
				{
					Vec3i face = cellToFace(cell, axis, direction);

					if (myCutCellWeights(face, axis) > 0)
					{
						isFluidCell = true;
						break;
					}
				}

			if (isFluidCell)
			{
				if (mySurface(cell) <= 0)
					materialCellLabels(cell) = MaterialLabels::LIQUID_CELL;
				else
					materialCellLabels(cell) = MaterialLabels::AIR_CELL;
			}
		}
	});

	constexpr int UNLABELLED_CELL = -1;

	UniformGrid<int> liquidCellIndices(mySurface.size(), UNLABELLED_CELL);

	int liquidCellCount = 0;

	forEachVoxelRange(Vec3i(0), liquidCellIndices.size(), [&](const Vec3i& cell)
	{
		if (materialCellLabels(cell) == MaterialLabels::LIQUID_CELL)
			liquidCellIndices(cell) = liquidCellCount++;
	});

	Vector rhsVector = Vector::Zero(liquidCellCount);
	Vector initialGuessVector = Vector::Zero(liquidCellCount);

	tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<SolveReal>>> parallelSparseMatrixElements;

	tbb::parallel_for(tbb::blocked_range<int>(0, liquidCellIndices.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		auto &localSparseMatrixElements = parallelSparseMatrixElements.local();

		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec3i cell = liquidCellIndices.unflatten(cellIndex);

			int liquidIndex = liquidCellIndices(cell);

			if (liquidIndex >= 0)
			{
				assert(materialCellLabels(cell) == MaterialLabels::LIQUID_CELL);

				// Compute divergence to add to RHS
				SolveReal divergence = 0;

				for (int axis : {0, 1, 2})
					for (int direction : {0, 1})
					{
						Vec3i face = cellToFace(cell, axis, direction);

						SolveReal weight = myCutCellWeights(face, axis);

						SolveReal sign = (direction == 0) ? 1 : -1;

						// Add divergence from faces
						if (weight > 0)
							divergence += sign * weight * velocity(face, axis);
						if (weight < 1.)
							divergence += sign * (1. - weight) * mySolidVelocity(face, axis);
					}

				rhsVector(liquidIndex) = divergence;

				SolveReal diagonal = 0;

				for (int axis : {0, 1, 2})
					for (int direction : {0, 1})
					{
						Vec3i adjacentCell = cellToCell(cell, axis, direction);

						// Bounds check. If out-of-bounds, treat like a stationary grid-aligned solid.
						if (adjacentCell[axis] < 0 || adjacentCell[axis] >= mySurface.size()[axis])
							continue;

						Vec3i face = cellToFace(cell, axis, direction);

						SolveReal weight = myCutCellWeights(face, axis);

						if (weight > 0)
						{
							int adjacentLiquidIndex = liquidCellIndices(adjacentCell);
							if (adjacentLiquidIndex >= 0)
							{
								assert(materialCellLabels(adjacentCell) == MaterialLabels::LIQUID_CELL);
								
								localSparseMatrixElements.emplace_back(liquidIndex, adjacentLiquidIndex, -weight);
								diagonal += weight;
							}
							else
							{
								assert(materialCellLabels(adjacentCell) == MaterialLabels::AIR_CELL);

								SolveReal theta = myGhostFluidWeights(face, axis);

								theta = Utilities::clamp(theta, SolveReal(.01), SolveReal(1));
								diagonal += weight / theta;
							}
						}
						else assert(materialCellLabels(adjacentCell) == MaterialLabels::SOLID_CELL);
					}

				assert(diagonal > 0);
				localSparseMatrixElements.emplace_back(liquidIndex, liquidIndex, diagonal);

				if (myUseInitialGuessPressure)
				{
					assert(myInitialGuessPressure != nullptr);
					initialGuessVector(liquidIndex) = (*myInitialGuessPressure)(cell);
				}
			}
			else assert(materialCellLabels(cell) != MaterialLabels::LIQUID_CELL);
		}
	});

	std::vector<Eigen::Triplet<SolveReal>> sparseMatrixElements;
	mergeLocalThreadVectors(sparseMatrixElements, parallelSparseMatrixElements);

	Eigen::SparseMatrix<SolveReal> sparseMatrix(liquidCellCount, liquidCellCount);
	sparseMatrix.setFromTriplets(sparseMatrixElements.begin(), sparseMatrixElements.end());

	Eigen::ConjugateGradient<Eigen::SparseMatrix<SolveReal>, Eigen::Upper | Eigen::Lower> solver;
	solver.compute(sparseMatrix);

	if (solver.info() != Eigen::Success)
	{
		std::cout << "   Solver failed to build" << std::endl;
		return;
	}

	solver.setTolerance(1E-3);

	Vector solutionVector = solver.solveWithGuess(rhsVector, initialGuessVector);

	if (solver.info() != Eigen::Success)
	{
		std::cout << "   Solver failed to converge" << std::endl;
		return;
	}
	else
	{
		std::cout << "    Solver iterations:     " << solver.iterations() << std::endl;
		std::cout << "    Solver error: " << solver.error() << std::endl;
	}

	// Copy resulting vector to pressure grid
	tbb::parallel_for(tbb::blocked_range<int>(0, liquidCellIndices.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec3i cell = liquidCellIndices.unflatten(cellIndex);

			int liquidIndex = liquidCellIndices(cell);

			if (liquidIndex >= 0)
			{
				assert(materialCellLabels(cell) == MaterialLabels::LIQUID_CELL);
				myPressure(cell) = solutionVector(liquidIndex);
			}
			else
				assert(materialCellLabels(cell) != MaterialLabels::LIQUID_CELL);
		}
	});

	// Build valid faces
	for (int axis : {0, 1, 2})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, myValidFaces.grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
			{
				Vec3i face = myValidFaces.grid(axis).unflatten(faceIndex);
				
				if (myCutCellWeights(face, axis) > 0)
				{
					Vec3i backwardCell = faceToCell(face, axis, 0);
					Vec3i forwardCell = faceToCell(face, axis, 1);

					if (!(backwardCell[axis] < 0 || forwardCell[axis] >= mySurface.size()[axis]))
					{
						if (liquidCellIndices(backwardCell) >= 0 || liquidCellIndices(forwardCell) >= 0)
						{
							assert(materialCellLabels(backwardCell) == MaterialLabels::LIQUID_CELL ||
									materialCellLabels(forwardCell) == MaterialLabels::LIQUID_CELL);

							myValidFaces(face, axis) = VisitedCellLabels::FINISHED_CELL;
						}
						else myValidFaces(face, axis) = VisitedCellLabels::UNVISITED_CELL;
					}
					else myValidFaces(face, axis) = VisitedCellLabels::UNVISITED_CELL;
				}
			}
		});
	}

	// Apply pressure update
	for (int axis : {0, 1, 2})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, myValidFaces.grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
			{
				Vec3i face = myValidFaces.grid(axis).unflatten(faceIndex);

				if (myValidFaces(face, axis) == VisitedCellLabels::FINISHED_CELL)
				{
					Vec3i backwardCell = faceToCell(face, axis, 0);
					Vec3i forwardCell = faceToCell(face, axis, 1);

					assert(myCutCellWeights(face, axis) > 0);
					assert(backwardCell[axis] >= 0 && forwardCell[axis] <= mySurface.size()[axis]);
					assert(materialCellLabels(backwardCell) == MaterialLabels::LIQUID_CELL || materialCellLabels(forwardCell) == MaterialLabels::LIQUID_CELL);

					SolveReal gradient = myPressure(forwardCell) - myPressure(backwardCell);

					if (materialCellLabels(backwardCell) == MaterialLabels::AIR_CELL ||
						materialCellLabels(forwardCell) == MaterialLabels::AIR_CELL)
					{
						SolveReal theta = myGhostFluidWeights(face, axis);
						theta = Utilities::clamp(theta, SolveReal(.01), SolveReal(1));

						gradient /= theta;
					}

					velocity(face, axis) -= gradient;
				}
			}
		});
	}
}

}