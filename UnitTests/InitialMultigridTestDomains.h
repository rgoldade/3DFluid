#ifndef FLUIDSIM3D_INITIAL_MULTIGRID_TEST_DOMAINS_H
#define FLUIDSIM3D_INITIAL_MULTIGRID_TEST_DOMAINS_H

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

#include "ComputeWeights.h"
#include "GeometricMultigridOperators.h"
#include "LevelSet.h"
#include "UniformGrid.h"
#include "Utilities.h"

namespace FluidSim3D
{
	
using GeometricMultigridOperators::CellLabels;

std::pair<Vec3i, int> buildExpandedDomain(UniformGrid<CellLabels> &expandedDomainCellLabels,
											VectorGrid<double> &expandedBoundaryWeights,
											const UniformGrid<CellLabels> &baseDomainCellLabels,
											const VectorGrid<double> &baseBoundaryWeights)
{
	std::pair<Vec3i, int> mgSettings = buildExpandedDomainLabels(expandedDomainCellLabels,
																		baseDomainCellLabels);

	Vec3i exteriorOffset = mgSettings.first;

	Transform xform(baseBoundaryWeights.dx(), baseBoundaryWeights.xform().offset() - baseBoundaryWeights.dx() * exteriorOffset.cast<double>());
	expandedBoundaryWeights = VectorGrid<double>(xform, expandedDomainCellLabels.size(), Vec3d::Zero(), VectorGridSettings::SampleType::STAGGERED);
	// Build expanded boundary weights
	for (int axis : {0, 1, 2})
	{
		buildExpandedBoundaryWeights(expandedBoundaryWeights, baseBoundaryWeights, expandedDomainCellLabels, exteriorOffset, axis);
	}

	// Build boundary cells
	setBoundaryDomainLabels(expandedDomainCellLabels, expandedBoundaryWeights);
	
	assert(unitTestBoundaryCells(expandedDomainCellLabels, &expandedBoundaryWeights));
	assert(unitTestExteriorCells(expandedDomainCellLabels));

	return mgSettings;
}

void buildComplexDomain(UniformGrid<CellLabels> &domainCellLabels,
						VectorGrid<double> &boundaryWeights,
						const int gridSize,
						const bool useSolidSphere)
{
	assert(gridSize > 0);
	domainCellLabels.resize(Vec3i::Constant(gridSize), CellLabels::EXTERIOR_CELL);

	double dx = 1. / double(gridSize);

	Transform xform(dx, Vec3d::Zero());

	auto dirichletIsoSurface = [](const Vec3d& point)
	{
		return point[0] - .5 + .25 * std::sin(2. * PI * point[1] + 4. * PI * point[2]);
	};

	LevelSet dirichletSurface(xform, Vec3i::Constant(gridSize));

	tbb::parallel_for(tbb::blocked_range<int>(0, dirichletSurface.voxelCount()), [&](const tbb::blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec3i cell = dirichletSurface.unflatten(flatIndex);

			Vec3d point = dirichletSurface.indexToWorld(cell.cast<double>());
			dirichletSurface(cell) = dirichletIsoSurface(point);
		}
	});

	boundaryWeights = VectorGrid<double>(xform, Vec3i::Constant(gridSize), Vec3d::Ones(), VectorGridSettings::SampleType::STAGGERED);

	// Compute cut-cell weights
	if (useSolidSphere)
	{
		LevelSet solidSphereSurface(xform, Vec3i::Constant(gridSize), 5);

		auto sphereIsoSurface = [](const Vec3d& point) -> double 
		{
            const Vec3d sphereCenter= Vec3d::Constant(.5);
            constexpr double sphereRadius = .125;

            return (point - sphereCenter).norm() - sphereRadius;
        };

		tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount()), [&](const tbb::blocked_range<int> &range)
		{
			for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
			{
				Vec3i cell = solidSphereSurface.unflatten(flatIndex);

				Vec3d point = solidSphereSurface.indexToWorld(cell.cast<double>());
				solidSphereSurface(cell) = sphereIsoSurface(point);
			}
		});

		VectorGrid<double> cutCellWeights = computeCutCellWeights(solidSphereSurface, true);

		for (int axis : {0, 1, 2})
		{
			tbb::parallel_for(tbb::blocked_range<int>(0, boundaryWeights.grid(axis).voxelCount()), [&](const tbb::blocked_range<int> &range)
			{
				for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
				{
					Vec3i face = boundaryWeights.grid(axis).unflatten(flatIndex);
					boundaryWeights(face, axis) = cutCellWeights(face, axis);						
				}
			});
		}
	}

	// To keep a layer of exterior cells
	for (int axis : {0, 1, 2})
	{
		for (int direction : {0, 1})
		{
			Vec3i startFace = Vec3i::Zero();
			Vec3i endFace = boundaryWeights.size(axis);

			if (direction == 0)
			{
				endFace[axis] = 1;
			}
			else
			{
				startFace[axis] = endFace[axis] - 1;
			}

			forEachVoxelRange(startFace, endFace, [&](const Vec3i& face) { boundaryWeights(face, axis) = 0; });
		}
	}

	// Set domain cell labels
	tbb::parallel_for(tbb::blocked_range<int>(0, domainCellLabels.voxelCount()), [&](const tbb::blocked_range<int> &range)
	{
		for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
		{
			Vec3i cell = domainCellLabels.unflatten(flatIndex);

			// Make sure there is an open cut-cell face
			bool hasOpenFace = false;
			for (int axis : {0, 1, 2})
			{
				for (int direction : {0, 1})
				{
					Vec3i face = cellToFace(cell, axis, direction);

					if (boundaryWeights(face, axis) > 0)
					{
						hasOpenFace = true;
					}
				}
			}

			if (hasOpenFace)
			{
				// Sine wave for air-liquid boundary.
				double sdf = dirichletSurface(cell);
				if (sdf > 0)
				{
					domainCellLabels(cell) = CellLabels::DIRICHLET_CELL;
				}
				else
				{
					domainCellLabels(cell) = CellLabels::INTERIOR_CELL;
				}
			}
			else
			{
				assert(domainCellLabels(cell) == CellLabels::EXTERIOR_CELL);
			}
		}
	});

	VectorGrid<double> ghostFluidWeights = computeGhostFluidWeights(dirichletSurface);

	// Build ghost fluid weights
	for (int axis : {0, 1, 2})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, boundaryWeights.grid(axis).voxelCount()), [&](const tbb::blocked_range<int> &range)
		{
			for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
			{
				Vec3i face = boundaryWeights.grid(axis).unflatten(flatIndex);

				if (boundaryWeights(face, axis) > 0)
				{
					Vec3i backwardCell = faceToCell(face, axis, 0);
					Vec3i forwardCell = faceToCell(face, axis, 1);

					assert(backwardCell[axis] >= 0 && forwardCell[axis] < domainCellLabels.size()[axis]);

					auto backwardLabel = domainCellLabels(backwardCell);
					auto forwardLabel = domainCellLabels(forwardCell);

					assert(backwardLabel != CellLabels::EXTERIOR_CELL && forwardLabel != CellLabels::EXTERIOR_CELL);

					if (backwardLabel == CellLabels::DIRICHLET_CELL && forwardLabel == CellLabels::DIRICHLET_CELL)
					{
						boundaryWeights(face, axis) = 0;
					}
					else
					{
						if (backwardLabel == CellLabels::DIRICHLET_CELL || forwardLabel == CellLabels::DIRICHLET_CELL)
						{
							double backwardSDF = dirichletSurface(backwardCell);
							double forwardSDF = dirichletSurface(forwardCell);

							// Sine wave for air-liquid boundary.
							assert((backwardSDF > 0 && forwardSDF <= 0) || (backwardSDF <= 0 && forwardSDF > 0));

							double theta = ghostFluidWeights(face, axis);
							theta = std::clamp(theta, double(.01), double(1));

							boundaryWeights(face, axis) /= theta;
						}
					}
				}
			}
		});
	}
}

void buildSimpleDomain(UniformGrid<CellLabels> &domainCellLabels,
						VectorGrid<double> &boundaryWeights,
						const int gridSize,
						const int dirichletBand)
{
	assert(gridSize > 0);
	assert(dirichletBand >= 0);

	domainCellLabels.resize(Vec3i::Constant(gridSize), CellLabels::EXTERIOR_CELL);

	double dx = 1. / double(gridSize);

	// Set outer layers to DIRICHLET
	for (int axis : {0, 1, 2})
	{
		for (int direction : {0, 1})
		{
			Vec3i start = Vec3i::Zero();
			Vec3i end = domainCellLabels.size();

			if (direction == 0)
			{
				end[axis] = dirichletBand;
			}
			else
			{
				start[axis] = domainCellLabels.size()[axis] - dirichletBand;
			}

			forEachVoxelRange(start, end, [&](const Vec3i& cell)
			{
				domainCellLabels(cell) = CellLabels::DIRICHLET_CELL;
			});
		}
	}

	{
        Vec3i start = Vec3i::Constant(dirichletBand);
        Vec3i end = domainCellLabels.size() - Vec3i::Constant(dirichletBand);

        forEachVoxelRange(start, end, [&](const Vec3i& cell) { domainCellLabels(cell) = CellLabels::INTERIOR_CELL; });
    }

    boundaryWeights = VectorGrid<double>(Transform(dx, Vec3d::Zero()), Vec3i::Constant(gridSize), Vec3d::Zero(), VectorGridSettings::SampleType::STAGGERED);

	// Build boundary weights
	for (int axis : {0, 1, 2})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, boundaryWeights.grid(axis).voxelCount()), [&](const tbb::blocked_range<int> &range)
		{
			for (int flatIndex = range.begin(); flatIndex != range.end(); ++flatIndex)
			{
				Vec3i face = boundaryWeights.grid(axis).unflatten(flatIndex);
				bool isInterior = false;
				bool isExterior = false;
				for (int direction : {0, 1})
				{
					Vec3i cell = faceToCell(face, axis, direction);

					if (cell[axis] < 0 || cell[axis] >= domainCellLabels.size()[axis])
					{
						isExterior = true;
						continue;
					}

					if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL)
					{
						isInterior = true;
					}
					else if (domainCellLabels(cell) == CellLabels::EXTERIOR_CELL)
					{
						isExterior = true;
					}
				}

				if (isInterior && !isExterior)
				{
					boundaryWeights(face, axis) = 1;
				}
			}
		});
	}
}

}

#endif