#include "LevelSet.h"

#include "tbb/tbb.h"
#include "tbb/blocked_range3d.h"

#include <Eigen/Dense>
#include <Eigen/SVD>

namespace FluidSim3D::SurfaceTrackers
{

// Helper function to project a point to a triangle in 3-D
Vec3f pointToTriangleProjection(const Vec3f& point, const Vec3f& vertex0, const Vec3f& vertex1, const Vec3f& vertex2)
{
	// Normal form
	Vec3f E0 = vertex1 - vertex0, E1 = vertex2 - vertex0;

	Vec3f D = vertex0 - point;

	float a = dot(E0, E0), b = dot(E0, E1), c = dot(E1, E1);
	float d = dot(E0, D), e = dot(E1, D), f = dot(D, D);

	// TODO: check on abs with determinant incase of sign inversion
	float det = std::fabs(a * c - b * b);
	float s = b * e - c * d;
	float t = b * d - a * e;

	// Logic tree to account for the point projection into the various regions.
	// Method borrowed from https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
	// 	and https://www.mathworks.com/matlabcentral/fileexchange/22857-distance-between-a-point-and-a-triangle-in-3d
	//
	//  \      |
	//	 \reg2 |
	//	  \    |
	//	   \   |
	//	    \  |
	//	     \ |
	//	      *P2
	//	       |\
	//	       | \
	//	  reg3 |  \ reg1
	//	       |   \
	//	       |reg0\
	// 	       |     \
	//         |      \ P1
	//  -------*-------*------->s
	//	       | P0     \
	//    reg4 | reg5    \ reg6

	if (s + t <= det)
	{
		if (s < 0)
		{
			if (t < 0) // region 4
			{
				if (d < 0)
				{
					t = 0;
					if (-d >= a)
						s = 1;
					else
						s = -d / a;
				}
				else
				{
					s = 0;
					if (e >= 0)
						t = 0;
					else
					{
						if (-e >= c)
							t = 1;
						else
							t = -e / c;
					}
				}
			}
			else // region 3
			{
				s = 0;
				if (e >= 0)
					t = 0;
				else
				{
					if (-e >= c)
						t = 1;
					else
						t = -e / c;
				}
			}
		}
		else if (t < 0) // region 5
		{
			t = 0;
			if (d >= 0)
				s = 0;
			else
			{
				if (-d >= a)
					s = 1;
				else
					s = -d / a;
			}
		}
		else // region 0
		{
			s = s / det;
			t = t / det;
		}
	}
	else
	{
		if (s < 0) // region 2
		{
			float tmp0 = b + d;
			float tmp1 = c + e;
			if (tmp1 > tmp0)
			{
				float numer = tmp1 - tmp0;
				float denom = a - 2. * b + c;
				if (numer >= denom)
				{
					s = 1;
					t = 0;
				}
				else
				{
					s = numer / denom;
					t = 1 - s;
				}
			}
			else
			{
				s = 0;
				if (tmp1 <= 0)
					t = 1;
				else
				{
					if (e >= 0)
						t = 0;
					else
						t = -e / c;
				}
			}
		}
		else if (t < 0) // region 6
		{
			float tmp0 = b + e;
			float tmp1 = a + d;
			if (tmp1 > tmp0)
			{
				float numer = tmp1 - tmp0;
				float denom = a - 2. * b + c;
				if (numer >= denom)
				{
					t = 1;
					s = 0;
				}
				else
				{
					t = numer / denom;
					s = 1 - t;
				}
			}
			else
			{
				t = 0;
				if (tmp1 <= 0)
					s = 1;
				else
				{
					if (d >= 0)
						s = 0;
					else
						s = -d / a;
				}
			}
		}
		else // region 1
		{
			float numer = c + e - b - d;
			if (numer <= 0)
			{
				s = 0;
				t = 1;
			}
			else
			{
				float denom = a - 2. * b + c;
				if (numer >= denom)
				{
					s = 1;
					t = 0;
				}
				else
				{
					s = numer / denom;
					t = 1 - s;
				}
			}
		}
	}

	return Vec3f(vertex0 + s * E0 + t * E1);
}

void LevelSet::drawGrid(Renderer& renderer, bool doOnlyNarrowBand) const
{
	if (doOnlyNarrowBand)
	{
		forEachVoxelRange(Vec3i(0), size(), [&](const Vec3i& cell)
		{
			if (std::fabs(myPhiGrid(cell)) < myNarrowBand)
				myPhiGrid.drawGridCell(renderer, cell);
		});
	}
	else myPhiGrid.drawGrid(renderer);
}

void LevelSet::drawGridPlane(Renderer& renderer, Axis planeAxis, float position, bool doOnlyNarrowBand) const
{
	position = clamp(position, float(0), float(1));

	Vec3i start(0);
	Vec3i end(myPhiGrid.size() - Vec3i(1));

	if (planeAxis == Axis::XAXIS)
	{
		start[0] = std::floor(position * float(myPhiGrid.size()[0] - 1));
		end[0] = start[0] + 1;
	}
	else if (planeAxis == Axis::YAXIS)
	{
		start[1] = std::floor(position * float(myPhiGrid.size()[1] - 1));
		end[1] = start[1] + 1;

	}
	else if (planeAxis == Axis::ZAXIS)
	{
		start[2] = std::floor(position * float(myPhiGrid.size()[2] - 1));
		end[2] = start[2] + 1;
	}
	
	forEachVoxelRange(start, end, [&](const Vec3i& cell)
	{
		if (doOnlyNarrowBand)
		{
			if (std::fabs(myPhiGrid(cell)) < myNarrowBand)
				myPhiGrid.drawGridCell(renderer, cell);
		}
		else myPhiGrid.drawGridCell(renderer, cell);
	});
}

// Display a supersampled slice of the grid. The plane will have a normal in the plane_axis direction.
// The position is from [0,1] where 0 is at the grid origin and 1 is at the origin + size * dx.
void LevelSet::drawSupersampledValuesPlane(Renderer& renderer, Axis planeAxis, float position, float radius, int samples, float sampleSize) const
{
	myPhiGrid.drawSupersampledValuesPlane(renderer, planeAxis, position, radius, samples, sampleSize);
}
void LevelSet::drawSampleNormalsPlane(Renderer& renderer, Axis planeAxis, float position, const Vec3f& colour, float length) const
{
	myPhiGrid.drawSampleGradientsPlane(renderer, planeAxis, position, colour, length);
}

void LevelSet::drawSurface(Renderer& renderer, const Vec3f& colour, float lineWidth) const
{
	TriMesh tempMesh = buildMesh();
	tempMesh.drawMesh(renderer, true, colour, lineWidth);
}

// Find the nearest point on the interface starting from the index position.
// If the position falls outside of the narrow band, there isn't a defined gradient
// to use. In this case, the original position will be returned.

Vec3f LevelSet::findSurface(const Vec3f& worldPoint, int iterationLimit) const
{
	assert(iterationLimit >= 0);

	float phi = myPhiGrid.interp(worldPoint);

	float epsilon = 1E-2 * dx();
	Vec3f tempPoint = worldPoint;

	int iterationCount = 0;
	if (std::fabs(phi) < myNarrowBand)
	{
		while (std::fabs(phi) > epsilon && iterationCount < iterationLimit)
		{
			tempPoint -= phi * normal(tempPoint);
			phi = myPhiGrid.interp(tempPoint);
			++iterationCount;
		}
	}

	return tempPoint;
}

Vec3f LevelSet::findSurfaceIndex(const Vec3f& indexPoint, int iterationLimit) const
{
	Vec3f worldPoint = indexToWorld(indexPoint);
	worldPoint = findSurface(worldPoint, iterationLimit);
	return worldToIndex(worldPoint);
}

void LevelSet::reinit()
{
	ScalarGrid<float> tempPhiGrid = myPhiGrid;

	UniformGrid<VisitedCellLabels> reinitializedCells(size(), VisitedCellLabels::UNVISITED_CELL);

	// Find zero crossings
	tbb::parallel_for(tbb::blocked_range<int>(0, myPhiGrid.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int> &range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec3i cell = myPhiGrid.unflatten(cellIndex);

			bool isAtZeroCrossing = false;
			for (int axis = 0; axis < 3 && !isAtZeroCrossing; ++axis)
				for (int direction : {0, 1})
				{
					Vec3i adjacentCell = cellToCell(cell, axis, direction);

					if (adjacentCell[axis] < 0 || adjacentCell[axis] >= size()[axis]) continue;

					if ((myPhiGrid(cell) <= 0 && myPhiGrid(adjacentCell) > 0) ||
						(myPhiGrid(cell) > 0 && myPhiGrid(adjacentCell) <= 0))
					{
						isAtZeroCrossing = true;

						Vec3f worldPoint = indexToWorld(Vec3f(cell));
						Vec3f interfacePoint = findSurface(worldPoint, 5);

						float distance = dist(worldPoint, interfacePoint);

						tempPhiGrid(cell) = myPhiGrid(cell) < 0. ? -distance : distance;
						reinitializedCells(cell) = VisitedCellLabels::FINISHED_CELL;

						break;
					}
				}

			// Set unvisited grid cells to background value, using old grid for inside/outside sign
			if (!isAtZeroCrossing)
			{
				assert(reinitializedCells(cell) == VisitedCellLabels::UNVISITED_CELL);
				tempPhiGrid(cell) = myPhiGrid(cell) < 0. ? -myNarrowBand : myNarrowBand;
			}
		}
	});

	std::swap(myPhiGrid, tempPhiGrid);
	reinitFastMarching(reinitializedCells);
}

void LevelSet::initFromMesh(const TriMesh& initialMesh, bool doResizeGrid)
{
	// TODO: make parallel!!

	if (doResizeGrid)
	{
		// Determine the bounding box of the mesh to build the underlying grids
		Vec3f minBoundingBox(std::numeric_limits<float>::max());
		Vec3f maxBoundingBox(std::numeric_limits<float>::lowest());

		for (const auto& vertex : initialMesh.vertices())
			updateMinAndMax(minBoundingBox, maxBoundingBox, vertex.point());

		// Just for nice whole numbers, let's clamp the bounding box to be an integer
		// offset in index space then bring it back to world space
		float maxNarrowBand = 10.;
		maxNarrowBand = std::min(myNarrowBand / dx(), maxNarrowBand);

		minBoundingBox = (Vec3f(floor(minBoundingBox / dx())) - Vec3f(maxNarrowBand)) * dx();
		maxBoundingBox = (Vec3f(ceil(maxBoundingBox / dx())) + Vec3f(maxNarrowBand)) * dx();

		clear();
		Transform xform(dx(), minBoundingBox);
		// Since we know how big the mesh is, we know how big our grid needs to be (wrt to grid spacing)
		myPhiGrid = ScalarGrid<float>(xform, Vec3i((maxBoundingBox - minBoundingBox) / dx()), myNarrowBand);
	}
	else
		myPhiGrid.resize(size(), myNarrowBand);

	// We want to track which cells in the level set contain valid distance information.
	// The first pass will set cells close to the mesh as FINISHED.
	UniformGrid<VisitedCellLabels> reinitializedCells(size(), VisitedCellLabels::UNVISITED_CELL);
	UniformGrid<int> meshCellParities(size(), 0);

	for (const auto& triFace : initialMesh.triFaces())
	{
		// It's easier to work in our index space and just scale the distance later.
		std::array<Vec3f, 3> triVertices;
		for (int localVertexIndex : {0, 1, 2})
			triVertices[localVertexIndex] = worldToIndex(initialMesh.vertex(triFace.vertex(localVertexIndex)).point());

		// Degerate check
		for (int localVertexIndex : {0, 1, 2})
		{
			if (triVertices[localVertexIndex] == triVertices[(localVertexIndex + 1) % 3])
				continue;
		}

		// Record mesh-grid intersections between cell nodes (i.e. on grid edges)
		// Since we only cast rays *left-to-right* for inside/outside checking, we don't
		// need to know if the mesh intersects y-aligned grid edges
		Vec3f minVertexBB(triVertices[0]), maxVertexBB(triVertices[0]);

		for (int localVertexIndex : {1, 2})
			updateMinAndMax(minVertexBB, maxVertexBB, triVertices[localVertexIndex]);

		Vec3i ceilMin = Vec3i(ceil(minVertexBB));
		Vec3i floorMin = Vec3i(floor(minVertexBB)) - Vec3i(1);
		Vec3i floorMax = Vec3i(floor(maxVertexBB));

		// Z-axis intersection tests. Iterate along an aligned set of grid edges
		// in decsending order, checking for intersections at each edge.
		// If an intersection is found then we can stop searching along the set.
		for (int i = ceilMin[0]; i <= floorMax[0]; ++i)
			for (int j = ceilMin[1]; j <= floorMax[1]; ++j)
				for (int k = floorMax[2]; k >= floorMin[2]; --k)
				{
					Vec3f gridPoint(i, j, k);
					IntersectionLabels intersectionResult = exactTriIntersect(gridPoint, triVertices[0], triVertices[1], triVertices[2], Axis::ZAXIS);

					if (intersectionResult == IntersectionLabels::NO) continue;

					int parityChange = 0;
					float qrs = orient2d(triVertices[0].data(), triVertices[1].data(), triVertices[2].data());
					if (qrs < 0)
						parityChange = 1;
					else
					{
						assert(qrs > 0);
						parityChange = -1;
					}

					if (intersectionResult == IntersectionLabels::YES)
						meshCellParities(i, j, k + 1) += parityChange;
					// If the grid node is explicitly on the mesh-edge, set distance to zero
					// since it might not be exactly zero due to floating point error above.
					else
					{
						assert(intersectionResult == IntersectionLabels::ON);

						reinitializedCells(i, j, k) = VisitedCellLabels::FINISHED_CELL;
						myPhiGrid(i, j, k) = 0.;
						meshCellParities(i, j, k) += parityChange;
					}

					break;
				}
	}

	// Now that all the z-axis edge crossings have been found, we can compile the parity changes
	// and label grid nodes that are at the interface
	for (int i = 0; i < size()[0]; ++i)
		for (int j = 0; j < size()[1]; ++j)
		{
			int parity = myIsBackgroundNegative ? 1 : 0;

			for (int k = 0; k < size()[2]; ++k)
			{
				Vec3i cell(i, j, k);

				parity += meshCellParities(cell);
				meshCellParities(cell) = parity;

				if (parity > 0) myPhiGrid(cell) = -myPhiGrid(cell);
			}

			assert(myIsBackgroundNegative ? parity == 1 : parity == 0);
		}

	// With the parity assigned, loop over the grid once more and label nodes that have a sign change
	// with neighbouring nodes (this means parity goes from -'ve (and zero) to +'ve or vice versa).
	forEachVoxelRange(Vec3i(1), size() - Vec3i(1), [&](const Vec3i &cell)
	{
		bool isCellInside = meshCellParities(cell) > 0;

		for (int axis : {0, 1, 2})
			for (int direction : {0, 1})
			{
				Vec3i adjacentCell = cellToCell(cell, axis, direction);

				bool isAdjacentCellInside = meshCellParities(adjacentCell) > 0;

				if (isCellInside != isAdjacentCellInside)
					reinitializedCells(cell) = VisitedCellLabels::FINISHED_CELL;
			}
	});

	// Loop over all the triangles in the mesh. Level set grid cells labelled as FINISHED will be
	// updated with the distance to the surface if it happens to be shorter than the current
	// distance to the surface.
	for (const auto& triFace : initialMesh.triFaces())
	{
		std::array<Vec3f, 3> vertices;
		for (int localVertexIndex : {0, 1, 2})
			vertices[localVertexIndex] = worldToIndex(initialMesh.vertex(triFace.vertex(localVertexIndex)).point());

		Vec3i minBoundingBox = Vec3i(floor(minUnion(minUnion(vertices[0], vertices[1]), vertices[2]))) - Vec3i(2);
		minBoundingBox = maxUnion(minBoundingBox, Vec3i(0));

		Vec3i maxBoundingBox = Vec3i(ceil(maxUnion(maxUnion(vertices[0], vertices[1]), vertices[2]))) + Vec3i(2);
		Vec3i top = size() - Vec3i(1);
		maxBoundingBox = minUnion(maxBoundingBox, top);

		// Update distances to the mesh at grid cells within the bounding box
		for (int axis : {0, 1, 2})
			assert(minBoundingBox[axis] >= 0 && maxBoundingBox[axis] < size()[axis]);

		forEachVoxelRange(minBoundingBox, maxBoundingBox + Vec3i(1), [&](const Vec3i& cell)
		{
			if (reinitializedCells(cell) != VisitedCellLabels::UNVISITED_CELL)
			{
				Vec3f gridPoint(cell);

				Vec3f triProjectionPoint = pointToTriangleProjection(gridPoint, vertices[0], vertices[1], vertices[2]);

				float surfaceDistance = dist(gridPoint, triProjectionPoint) * dx();

				// If the new distance is closer than existing distance values, update cell
				if (std::fabs(myPhiGrid(cell)) > surfaceDistance)
				{
					// If the parity says the node is inside, set it to be negative
					myPhiGrid(cell) = (meshCellParities(cell) > 0) ? -surfaceDistance : surfaceDistance;
				}
			}
		});
	}

	reinitFastMarching(reinitializedCells);
}

Vec3f LevelSet::interpolateInterface(const Vec3i& startPoint, const Vec3i& endPoint) const
{
	assert((myPhiGrid(startPoint) <= 0 && myPhiGrid(endPoint) > 0) ||
			(myPhiGrid(startPoint) > 0 && myPhiGrid(endPoint) <= 0));

	if (myPhiGrid(startPoint) == 0 && myPhiGrid(endPoint) == 0)
		return Vec3f(startPoint);

	// Find weight to zero isosurface
	float s = myPhiGrid(startPoint) / (myPhiGrid(startPoint) - myPhiGrid(endPoint));
	s = Utilities::clamp(s, float(0), float(1));

	Vec3f dx = Vec3f(endPoint - startPoint);
	return Vec3f(startPoint) + s * dx;
}

void LevelSet::unionSurface(const LevelSet& unionPhi)
{
	assert(isGridMatched(unionPhi));

	tbb::parallel_for(tbb::blocked_range<int>(0, voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int index = range.begin(); index != range.end(); ++index)
		{
			Vec3i cell = myPhiGrid.unflatten(index);
			if (unionPhi(cell) < 2 * unionPhi.dx())
				myPhiGrid(cell) = std::min(myPhiGrid(cell), unionPhi(cell));
		}
	});

	reinitMesh();
}

TriMesh LevelSet::buildMesh() const
{
	// Create grid to store index to dual contouring point. Note that phi is
	// center sampled so the DC grid must be node sampled and one cell shorter
	// in each dimension
	UniformGrid<int> dcPointIndices(size() - Vec3i(1), -1);
	std::vector<std::pair<Vec3i, Vec3f>> dcPointPair;

	// Build list of dual contouring points
	{
		tbb::enumerable_thread_specific<std::vector<std::pair<Vec3i, Vec3f>>> parallelDCPoints;

		tbb::parallel_for(tbb::blocked_range<int>(0, dcPointIndices.voxelCount(), tbbHeavyGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			auto& localDCPoints = parallelDCPoints.local();

			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec3i cell = dcPointIndices.unflatten(cellIndex);

				std::vector<Vec3f> qefPoints;
				std::vector<Vec3f> qefNormals;

				for (int edgeAxis : {0, 1, 2})
					for (int edgeIndex = 0; edgeIndex < 4; ++edgeIndex)
					{
						Vec3i edge = cellToEdge(cell, edgeAxis, edgeIndex);

						Vec3i backwardNode = edgeToNode(edge, edgeAxis, 0);
						Vec3i forwardNode = edgeToNode(edge, edgeAxis, 1);

						// Look for zero crossings.
						// Note that nodes for the DC grid fall exactly on the cell centers
						// of the level set grid.
						if ((myPhiGrid(backwardNode) <= 0 && myPhiGrid(forwardNode) > 0) ||
							(myPhiGrid(backwardNode) > 0 && myPhiGrid(forwardNode) <= 0))
						{
							// Find interface point
							Vec3f interfacePoint = interpolateInterface(backwardNode, forwardNode);
							qefPoints.push_back(interfacePoint);

							// Find associated surface normal
							Vec3f surfaceNormal = normal(indexToWorld(interfacePoint));
							qefNormals.push_back(surfaceNormal);
						}
					}

				if (qefPoints.size() > 0)
				{
					assert(qefPoints.size() > 2);

					Eigen::MatrixXd A(qefPoints.size(), 3);
					Eigen::VectorXd b(qefPoints.size());
					Eigen::VectorXd pointCOM = Eigen::VectorXd::Zero(3);

					for (int pointIndex = 0; pointIndex < qefPoints.size(); ++pointIndex)
					{
						for (int axis : {0, 1, 2})
						{
							A(pointIndex, axis) = qefNormals[pointIndex][axis];
							pointCOM[axis] += qefPoints[pointIndex][axis];
						}

						b(pointIndex) = dot(qefNormals[pointIndex], qefPoints[pointIndex]);
					}

					pointCOM /= double(qefPoints.size());

					// TODO: clamp singular values?
					Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
					svd.setThreshold(1E-2);

					Eigen::VectorXd dcPoint = pointCOM + svd.solve(b - A * pointCOM);

					Vec3d vecCOM(pointCOM[0], pointCOM[1], pointCOM[2]);

					// Because we set the DC point indices is a seperate loop, the DC points must remain
					// in their own cell to properly index.
					Vec3d boundingBoxMin = Vec3d(cell);
					Vec3d boundingBoxMax = Vec3d(cell) + Vec3d(1);

					if (dcPoint[0] < boundingBoxMin[0] || dcPoint[1] < boundingBoxMin[1] || dcPoint[2] < boundingBoxMin[2] ||
						dcPoint[0] >= boundingBoxMax[0] || dcPoint[1] >= boundingBoxMax[1] || dcPoint[2] >= boundingBoxMax[2])
						dcPoint = pointCOM;

					localDCPoints.emplace_back(cell, Vec3f(dcPoint[0], dcPoint[1], dcPoint[2]));
				}
			}
		});

		mergeLocalThreadVectors(dcPointPair, parallelDCPoints);
	}
	
	std::vector<Vec3f> vertices(dcPointPair.size());

	// Set DC point index for direct look up when building mesh and
	// convert points to world space.
	tbb::parallel_for(tbb::blocked_range<int>(0, dcPointPair.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int index = range.begin(); index != range.end(); ++index)
		{
			const Vec3i& cell = dcPointPair[index].first;

			assert(dcPointIndices(cell) == -1);
			dcPointIndices(cell) = index;

			const Vec3f& point = dcPointPair[index].second;
			vertices[index] = indexToWorld(point);
		}
	});

	// Build triangle mesh using dual contouring points

	std::vector<Vec3i> triFaces;

	for (int edgeAxis : {0, 1, 2})
	{
		Vec3i start(0);
		++start[(edgeAxis + 1) % 3];
		++start[(edgeAxis + 2) % 3];

		Vec3i end(dcPointIndices.size());

		tbb::enumerable_thread_specific<std::vector<Vec3i>> parallelTriFaces;
		
		auto loopRange3d = tbb::blocked_range3d<int>(start[0], end[0], std::cbrt(tbbLightGrainSize), start[1], end[1], std::cbrt(tbbLightGrainSize), start[2], end[2], std::cbrt(tbbLightGrainSize));

		tbb::parallel_for(loopRange3d, [&](const tbb::blocked_range3d<int>& range)
		{
			auto& localTriFaces = parallelTriFaces.local();

			Vec3i edge;
			for (edge[0] = range.pages().begin(); edge[0] != range.pages().end(); ++edge[0])
				for (edge[1] = range.rows().begin(); edge[1] != range.rows().end(); ++edge[1])
					for (edge[2] = range.cols().begin(); edge[2] != range.cols().end(); ++edge[2])
					{
						Vec3i backwardNode = edgeToNode(edge, edgeAxis, 0);
						Vec3i forwardNode = edgeToNode(edge, edgeAxis, 1);

						if ((myPhiGrid(backwardNode) <= 0 && myPhiGrid(forwardNode) > 0) ||
							(myPhiGrid(backwardNode) > 0 && myPhiGrid(forwardNode) <= 0))
						{
							std::array<Vec3i, 4> dcCells;
							std::array<int, 4> vertexIndices;

							for (int cellIndex = 0; cellIndex < 4; ++cellIndex)
							{
								dcCells[cellIndex] = edgeToCellCCW(edge, edgeAxis, cellIndex);
								vertexIndices[cellIndex] = dcPointIndices(dcCells[cellIndex]);
								assert(vertexIndices[cellIndex] >= 0);
							}

							if (myPhiGrid(backwardNode) <= 0.)
							{
								localTriFaces.emplace_back(vertexIndices[0], vertexIndices[1], vertexIndices[2]);
								localTriFaces.emplace_back(vertexIndices[0], vertexIndices[2], vertexIndices[3]);
							}
							else
							{
								localTriFaces.emplace_back(vertexIndices[0], vertexIndices[2], vertexIndices[1]);
								localTriFaces.emplace_back(vertexIndices[0], vertexIndices[3], vertexIndices[2]);
							}
						}
					}		
		});

		mergeLocalThreadVectors(triFaces, parallelTriFaces);
	}

	return TriMesh(triFaces, vertices);
}

void LevelSet::reinitFastMarching(UniformGrid<VisitedCellLabels>& reinitializedCells)
{
	assert(reinitializedCells.size() == size());

	auto solveEikonal2D = [&](float Ux, float Uy) -> float
	{
		if (std::fabs(Ux - Uy) >= dx())
			return std::min(Ux, Uy) + dx();
		else
			return ((Ux + Uy) + std::sqrt(sqr(Ux + Uy) - 2. * (sqr(Ux) + sqr(Uy) - sqr(dx())))) / 2.;
	};

	auto solveEikonal = [&](const Vec3i& cell) -> float
	{
		float max = std::numeric_limits<float>::max();

		float U_bx = (cell[0] > 0) ? std::fabs(myPhiGrid(cell[0] - 1, cell[1], cell[2])) : max;
		float U_fx = (cell[0] < size()[0] - 1) ? std::fabs(myPhiGrid(cell[0] + 1, cell[1], cell[2])) : max;

		float U_by = (cell[1] > 0) ? std::fabs(myPhiGrid(cell[0], cell[1] - 1, cell[2])) : max;
		float U_fy = (cell[1] < size()[1] - 1) ? std::fabs(myPhiGrid(cell[0], cell[1] + 1, cell[2])) : max;

		float U_bz = (cell[2] > 0) ? std::fabs(myPhiGrid(cell[0], cell[1], cell[2] - 1)) : max;
		float U_fz = (cell[2] < size()[2] - 1) ? std::fabs(myPhiGrid(cell[0], cell[1], cell[2] + 1)) : max;

		float Ux = min(U_bx, U_fx);
		float Uy = min(U_by, U_fy);
		float Uz = min(U_bz, U_fz);

		float discrim = sqr(Ux + Uy + Uz) - 3. * (sqr(Ux) + sqr(Uy) + sqr(Uz) - sqr(dx()));
		float U;
		if (discrim < 0.)
			U = min(min(solveEikonal2D(Ux, Uy), solveEikonal2D(Uy, Uz)), solveEikonal2D(Ux, Uz));
		else
			// Quadratic equation from the Eikonal
			U = ((Ux + Uy + Uz) + sqrt(discrim)) / 3.;

		return U;
	};

	// Load up the BFS queue with the unvisited cells next to the finished ones
	typedef std::pair<Vec3i, float> Node;
	auto cmp = [](const Node& a, const Node& b) -> bool { return std::fabs(a.second) > std::fabs(b.second); };
	std::priority_queue<Node, std::vector<Node>, decltype(cmp)> marchingQ(cmp);

	forEachVoxelRange(Vec3i(0), reinitializedCells.size(), [&](const Vec3i& cell)
	{
		if (reinitializedCells(cell) == VisitedCellLabels::FINISHED_CELL)
		{
			for (int axis : {0, 1, 2})
				for (int direction : {0, 1})
				{
					Vec3i adjacentCell = cellToCell(cell, axis, direction);

					if (adjacentCell[axis] < 0 || adjacentCell[axis] >= reinitializedCells.size()[axis])
						continue;

					if (reinitializedCells(adjacentCell) == VisitedCellLabels::UNVISITED_CELL)
					{
						float dist = solveEikonal(adjacentCell);
						assert(dist >= 0);

						myPhiGrid(adjacentCell) = myPhiGrid(adjacentCell) <= 0 ? -dist : dist;

						Node node(adjacentCell, dist);

						marchingQ.push(node);
						reinitializedCells(adjacentCell) = VisitedCellLabels::VISITED_CELL;
					}
				}
		}
	});

	while (!marchingQ.empty())
	{
		Node localNode = marchingQ.top();
		Vec3i localCell = localNode.first;
		marchingQ.pop();

		// Since you can't just update parts of the priority queue,
		// it's possible that a cell has been solidified at a smaller distance
		// and an older insert if floating around.
		if (reinitializedCells(localCell) == VisitedCellLabels::FINISHED_CELL)
		{
			// Make sure that the distance assigned to the cell is smaller than
			// what is floating around
			assert(std::fabs(myPhiGrid(localCell)) <= std::fabs(localNode.second));
			continue;
		}
		assert(reinitializedCells(localCell) == VisitedCellLabels::VISITED_CELL);

		if (std::fabs(myPhiGrid(localCell)) < myNarrowBand)
		{
			// Debug check that there is indeed a FINISHED cell next to it
			bool foundFinishedCell = false;
			
			for (int axis : {0, 1, 2})
				for (int direction : {0, 1})
				{
					Vec3i adjacentCell = cellToCell(localCell, axis, direction);

					if (adjacentCell[axis] < 0 || adjacentCell[axis] >= reinitializedCells.size()[axis])
						continue;

					if (reinitializedCells(adjacentCell) == VisitedCellLabels::FINISHED_CELL)
						foundFinishedCell = true;
					else
					{
						float dist = solveEikonal(adjacentCell);
						assert(dist >= 0);

						if (dist > myNarrowBand) dist = myNarrowBand;

						if (reinitializedCells(adjacentCell) == VisitedCellLabels::VISITED_CELL && dist > std::fabs(myPhiGrid(adjacentCell)))
							continue;

						myPhiGrid(adjacentCell) = myPhiGrid(adjacentCell) < 0 ? -dist : dist;

						Node node(adjacentCell, dist);

						marchingQ.push(node);
						reinitializedCells(adjacentCell) = VisitedCellLabels::VISITED_CELL;
					}
				}
			assert(foundFinishedCell);
		}
		else myPhiGrid(localCell) = myPhiGrid(localCell) < 0 ? -myNarrowBand : myNarrowBand;

		reinitializedCells(localCell) = VisitedCellLabels::FINISHED_CELL;
	}
}

}