#include "ClosestPointComputer.h"

#include "tbb/blocked_range.h"
#include "tbb/enumerable_thread_specific.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"

#include "GridUtilities.h"

namespace FluidSim3D
{
ClosestPointComputer::ClosestPointComputer() {}

ClosestPointComputer::ClosestPointComputer(const TriMesh& mesh)
	: myMesh(mesh)
{
	// Build AABB for mesh
	AlignedBox3d bbox = tbb::parallel_reduce(tbb::blocked_range<int>(0, mesh.vertexCount()), AlignedBox3d(), [&](tbb::blocked_range<int>& range, AlignedBox3d bbox)
						{
							for (int vertIndex = range.begin(); vertIndex != range.end(); ++vertIndex)
							{
								bbox.extend(mesh.vertex(vertIndex));
							}

							return bbox;
						},
						[](const AlignedBox3d& bbox0, AlignedBox3d bbox1)
						{
							bbox1.extend(bbox0);
							return bbox1;
						});

	// Build average AABB size
	Vec3d summedBBoxSize = tbb::parallel_deterministic_reduce(tbb::blocked_range<int>(0, mesh.triangleCount()), Vec3d::Zero().eval(), [&](tbb::blocked_range<int>& range, Vec3d localSummedBBoxSize) -> Vec3d
		{
			for (int triIndex = range.begin(); triIndex != range.end(); ++triIndex)
			{
				const Vec3i& tri = mesh.triangle(triIndex);

				AlignedBox3d bbox(mesh.vertex(tri[0]));
				bbox.extend(mesh.vertex(tri[1]));
				bbox.extend(mesh.vertex(tri[2]));

				localSummedBBoxSize += bbox.max() - bbox.min();
			}

			return localSummedBBoxSize;
		},
		[](const Vec3d& bboxSize0, const Vec3d& bboxSize1) -> Vec3d
		{
			return bboxSize0 + bboxSize1;
		});

	summedBBoxSize.array() /= double(mesh.triangleCount());

	double dx = summedBBoxSize.maxCoeff();

	Vec3d origin = (bbox.min() / dx).array().floor() * dx;

	myXform = Transform(dx, origin);

	Vec3i gridSize = myXform.worldToIndex(bbox.max()).array().ceil().cast<int>();
	myMeshGrid.resize(gridSize);

	// Insert mesh indices into grid
	tbb::enumerable_thread_specific<std::vector<std::pair<Vec3i, int>>> parallelCellTrianglePairs;
	tbb::parallel_for(tbb::blocked_range<int>(0, mesh.triangleCount()), [&](const tbb::blocked_range<int>& range)
	{
		auto& localCellTrianglePairs = parallelCellTrianglePairs.local();
		for (int triIndex = range.begin(); triIndex != range.end(); ++triIndex)
		{
			// Iterate over the grid indices spanned by the mesh AABB
			const Vec3i& tri = mesh.triangle(triIndex);
			AlignedBox3d localBBox(myXform.worldToIndex(mesh.vertex(tri[0])));
			localBBox.extend(myXform.worldToIndex(mesh.vertex(tri[1])));
			localBBox.extend(myXform.worldToIndex(mesh.vertex(tri[2])));

			forEachVoxelRange(localBBox.min().array().floor().cast<int>(), localBBox.max().array().ceil().cast<int>(), [&](const Vec3i& cell)
			{
				localCellTrianglePairs.emplace_back(cell, triIndex);
			});
		}
	});

	std::vector<std::pair<Vec3i, int>> cellTrianglePairs;
	mergeLocalThreadVectors(cellTrianglePairs, parallelCellTrianglePairs);

	parallelCellTrianglePairs.combine_each([&](const std::vector<std::pair<Vec3i, int>>& cellTrianglePairs)
	{
		for (const auto& cellTrianglePair : cellTrianglePairs)
		{
			myMeshGrid(cellTrianglePair.first).push_back(cellTrianglePair.second);
		}
	});
}

std::pair<Vec3d, int> ClosestPointComputer::computeClosestPoint(const Vec3d& queryPoint, double radius)
{
	// Get voxels within radius of query point

	AlignedBox3d bbox;
	if (radius < std::numeric_limits<double>::max())
	{
		bbox.extend(myXform.worldToIndex(queryPoint + Vec3d::Constant(radius)));
		bbox.extend(myXform.worldToIndex(queryPoint - Vec3d::Constant(radius)));
	}
	else
	{
		bbox.extend(Vec3d::Zero());
		bbox.extend(myMeshGrid.size().cast<double>());
	}

	Vec3i startCell = bbox.min().array().floor().cast<int>().max(Vec3i::Zero().array());
	Vec3i endCell = bbox.max().array().ceil().cast<int>().min(myMeshGrid.size().array());

	int triCount = 0;
	forEachVoxelRange(startCell, endCell, [&](const Vec3i& cell)
	{
		triCount += int(myMeshGrid(cell).size());
	});

	std::vector<int> candidateTris;
	candidateTris.reserve(triCount);

	forEachVoxelRange(startCell, endCell, [&](const Vec3i& cell)
	{
		candidateTris.insert(candidateTris.end(), myMeshGrid(cell).begin(), myMeshGrid(cell).end());
	});

	// Find closest point
	std::tuple<double, int, Vec3d> cpTuple(radius, -1, Vec3d());
	
	for (int triIndex : candidateTris)
	{
		const Vec3i& tri = myMesh.triangle(triIndex);
		Vec3d vp = pointToTriangleProjection(queryPoint, myMesh.vertex(tri[0]), myMesh.vertex(tri[1]), myMesh.vertex(tri[2]));

		double localDist = (queryPoint - vp).norm();

		if (localDist < std::get<0>(cpTuple))
		{
			std::get<0>(cpTuple) = localDist;
			std::get<1>(cpTuple) = triIndex;
			std::get<2>(cpTuple) = vp;
		}
	}

	return std::make_pair(std::get<2>(cpTuple), std::get<1>(cpTuple));
}

}