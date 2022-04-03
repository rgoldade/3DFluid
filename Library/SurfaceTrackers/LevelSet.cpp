#include "LevelSet.h"

#include <array>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "tbb/blocked_range3d.h"
#include "tbb/tbb.h"

namespace FluidSim3D
{

LevelSet::LevelSet()
    : myNarrowBand(0)
    , myPhiGrid(Transform(0, Vec3d::Zero()), Vec3i::Zero())
    , myIsBackgroundNegative(false)
{
    exactinit();
}

LevelSet::LevelSet(const Transform& xform, const Vec3i& size)
    : LevelSet(xform, size, size[0] * size[1] * size[2]) 
{}

LevelSet::LevelSet(const Transform& xform, const Vec3i& size, double bandwidth, bool isBoundaryNegative)
    : myNarrowBand(bandwidth * xform.dx())
    , myIsBackgroundNegative(isBoundaryNegative)
    , myPhiGrid(xform, size, myIsBackgroundNegative ? -myNarrowBand : myNarrowBand)
{
    for (int axis : {0, 1, 2}) assert(size[axis] >= 0);

    // In order to deal with triangle meshes, we need to initialize
    // the geometric predicate library.
    exactinit();
}

void LevelSet::initFromMesh(const TriMesh& initialMesh, bool doResizeGrid)
{
	// The internal code can't handle a mesh that falls outside of the bounds.
	// If the mesh does, we need to create a new copy and clamp it into the bounds of the grid.

    if (!doResizeGrid)
    {
		bool outOfBounds = false;
		for (const Vec3d& vertex : initialMesh.vertices())
		{
			Vec3d indexPoint = worldToIndex(vertex);

            for (int axis : {0, 1, 2})
            {
			    if (indexPoint[axis] <= 0 || indexPoint[axis] >= size()[axis])
				    outOfBounds = true;
            }
		}

		if (outOfBounds)
		{
			TriMesh clampedMesh = initialMesh;

			// Clamp to be inside grid
			for (int vertIndex = 0; vertIndex < clampedMesh.vertexCount(); ++vertIndex)
			{
				const Vec3d& vertex = clampedMesh.vertex(vertIndex);
				Vec3d indexPoint = worldToIndex(vertex);

				double offset = 1e-5 * dx();

                for (int axis : {0, 1, 2})
                {
                    indexPoint[axis] = std::clamp(indexPoint[axis], offset, size()[axis] - 1. - offset);
                }

				clampedMesh.setVertex(vertIndex, indexToWorld(indexPoint));
			}

			initFromMeshImpl(clampedMesh, doResizeGrid);
			return;
		}
    }

    initFromMeshImpl(initialMesh, doResizeGrid);
}

void LevelSet::reinit()
{
    TriMesh tempMesh = buildMesh();
    initFromMeshImpl(tempMesh, false);
}

bool LevelSet::isGridMatched(const LevelSet& grid) const
{
    if (size() != grid.size()) return false;
    if (xform() != grid.xform()) return false;
    return true;
}

bool LevelSet::isGridMatched(const ScalarGrid<double>& grid) const
{
    if (grid.sampleType() != ScalarGridSettings::SampleType::CENTER) return false;
    if (size() != grid.size()) return false;
    if (xform() != grid.xform()) return false;
    return true;
}

void LevelSet::unionSurface(const LevelSet& unionPhi)
{
    assert(isGridMatched(unionPhi));

    tbb::parallel_for(tbb::blocked_range<int>(0, voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
    {
        for (int index = range.begin(); index != range.end(); ++index)
        {
            Vec3i cell = myPhiGrid.unflatten(index);
            myPhiGrid(cell) = std::min(myPhiGrid(cell), unionPhi(cell));
        }
    });
}

bool LevelSet::isBackgroundNegative() const
{
    return myIsBackgroundNegative;
}

void LevelSet::setBackgroundNegative()
{
    myIsBackgroundNegative = true;
}

TriMesh LevelSet::buildMesh() const
{
    // Create grid to store index to dual contouring point. Note that phi is
    // center sampled so the DC grid must be node sampled and one cell shorter
    // in each dimension
    UniformGrid<int> dcPointIndices(size() - Vec3i::Ones(), -1);
    std::vector<std::pair<Vec3i, Vec3d>> dcPointPair;

    // Build list of dual contouring points
    {
        tbb::enumerable_thread_specific<std::vector<std::pair<Vec3i, Vec3d>>> parallelDCPoints;

        tbb::parallel_for(tbb::blocked_range<int>(0, dcPointIndices.voxelCount(), tbbHeavyGrainSize), [&](const tbb::blocked_range<int>& range)
        {
            auto& localDCPoints = parallelDCPoints.local();

            for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
            {
                Vec3i cell = dcPointIndices.unflatten(cellIndex);

                VecVec3d points;
                VecVec3d normals;

                Vec3d averagePoint = Vec3d::Zero();

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
                            Vec3d point = interpolateInterface(backwardNode, forwardNode);

                            for (int axis : {0, 1, 2})
                                assert(point[axis] >= backwardNode[axis] && point[axis] <= forwardNode[axis]);

                            points.push_back(point);

                            averagePoint += point;

                            // Find associated surface normal
                            Vec3d localNormal = normal(indexToWorld(point));

                            normals.push_back(localNormal);
                        }
                    }

                if (points.size() > 0)
                {
                    averagePoint.array() /= double(points.size());

                    Matrix3x3d AtA = Matrix3x3d::Zero();

                    Vec3d rhs = Vec3d::Zero();

                    for (int pointIndex = 0; pointIndex < points.size(); ++pointIndex)
                    {
                        AtA += normals[pointIndex] * normals[pointIndex].transpose();
                        rhs += normals[pointIndex] * normals[pointIndex].dot(points[pointIndex] - averagePoint);
                    }

                    Eigen::SelfAdjointEigenSolver<Matrix3x3d> eigenSolver(AtA);
                    const Vec3d& eigenvalues = eigenSolver.eigenvalues();

                    // Clamp eigenvalues
                    double tolerance = 0.01 * eigenvalues.cwiseAbs().maxCoeff();

                    Vec3d invEigenvalues;
                    int clamped = 0;
                    for (int index : {0, 1, 2})
                    {
                        if (std::fabs(eigenvalues[index] < tolerance))
                        {
                            invEigenvalues[index] = 0;
                            ++clamped;
                        }
                        else
                        {
                            invEigenvalues[index] = 1. / eigenvalues[index];
                        }
                    }

                    Vec3d qefPoint;
                    if (clamped < 3)
                    {
                        qefPoint = averagePoint + eigenSolver.eigenvectors() * invEigenvalues.asDiagonal() * eigenSolver.eigenvectors().transpose() * rhs;
                    }
                    else
                    {
                        qefPoint = averagePoint;
                    }

                    // Clamp to cell
                    if (qefPoint[0] < cell[0] || qefPoint[0] > cell[0] + 1 ||
                        qefPoint[1] < cell[1] || qefPoint[1] > cell[1] + 1 ||
                        qefPoint[2] < cell[2] || qefPoint[2] > cell[2] + 1)
                    {
                        AlignedBox3d cellBbox(cell.cast<double>());
                        cellBbox.extend((cell + Vec3i::Ones()).cast<double>());
                        Vec3d rayDirection = averagePoint - qefPoint;
                        double alpha = computeRayBBoxIntersection(cellBbox, qefPoint, rayDirection);

                        if (alpha <= 1)
                        {
                            qefPoint += alpha * rayDirection;
                        }
                        else
                        {
                            qefPoint = averagePoint;
                        }
                    }

                    localDCPoints.emplace_back(cell, qefPoint);
                }
            }
        });

        mergeLocalThreadVectors(dcPointPair, parallelDCPoints);
    }

    VecVec3d vertices(dcPointPair.size());

    // Set DC point index for direct look up when building mesh and
    // convert points to world space.
    tbb::parallel_for(tbb::blocked_range<size_t>(0, dcPointPair.size(), tbbLightGrainSize), [&](const tbb::blocked_range<size_t>& range)
    {
        for (size_t index = range.begin(); index != range.end(); ++index)
        {
            const Vec3i& cell = dcPointPair[index].first;

            assert(dcPointIndices(cell) == -1);
            dcPointIndices(cell) = int(index);

            const Vec3d& point = dcPointPair[index].second;
            vertices[index] = indexToWorld(point);
        }
    });

    // Build triangle mesh using dual contouring points

    VecVec3i triangles;

    for (int edgeAxis : {0, 1, 2})
    {
        Vec3i start = Vec3i::Zero();
        ++start[(edgeAxis + 1) % 3];
        ++start[(edgeAxis + 2) % 3];

        Vec3i end = dcPointIndices.size();

        tbb::enumerable_thread_specific<VecVec3i> parallelTriangles;

        auto loopRange3d = tbb::blocked_range3d<int>(start[0], end[0], int(std::cbrt(tbbLightGrainSize)), start[1], end[1],
                                      int(std::cbrt(tbbLightGrainSize)), start[2], end[2], int(std::cbrt(tbbLightGrainSize)));

        tbb::parallel_for(loopRange3d, [&](const tbb::blocked_range3d<int>& range)
        {
            auto& localTriFaces = parallelTriangles.local();

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

        mergeLocalThreadVectors(triangles, parallelTriangles);
    }

    return TriMesh(triangles, vertices);
}

void LevelSet::clear()
{
    myPhiGrid.clear();
}

void LevelSet::resize(const Vec3i& size)
{
    myPhiGrid.resize(size);
}

// Find the nearest point on the interface starting from the index position.
// If the position falls outside of the narrow band, there isn't a defined gradient
// to use. In this case, the original position will be returned.
Vec3d LevelSet::findSurface(const Vec3d& worldPoint, int iterationLimit, double tolerance) const
{
    assert(iterationLimit >= 0);

    double phi = myPhiGrid.triLerp(worldPoint);

    double epsilon = tolerance * dx();
    Vec3d tempPoint = worldPoint;

    int iterationCount = 0;
    if (std::fabs(phi) < myNarrowBand)
    {
        while (std::fabs(phi) > epsilon && iterationCount < iterationLimit)
        {
            tempPoint -= phi * .8 * normal(tempPoint);
            phi = myPhiGrid.triLerp(tempPoint);
            ++iterationCount;
        }
    }

    return tempPoint;
}

Vec3d LevelSet::interpolateInterface(const Vec3i& startPoint, const Vec3i& endPoint) const
{
    assert((myPhiGrid(startPoint) <= 0 && myPhiGrid(endPoint) > 0) ||
           (myPhiGrid(startPoint) > 0 && myPhiGrid(endPoint) <= 0));

    // Find weight to zero isosurface
    double theta = lengthFraction(myPhiGrid(startPoint), myPhiGrid(endPoint));

    assert(theta >= 0 && theta <= 1);

    if (myPhiGrid(startPoint) > 0)
        theta = 1. - theta;

    return startPoint.cast<double>() + theta * (endPoint - startPoint).cast<double>();
}

void LevelSet::drawGrid(Renderer& renderer, bool doOnlyNarrowBand) const
{
    if (doOnlyNarrowBand)
    {
        forEachVoxelRange(Vec3i::Zero().eval(), size(), [&](const Vec3i& cell)
        {
            if (std::fabs(myPhiGrid(cell)) < myNarrowBand) myPhiGrid.drawGridCell(renderer, cell);
        });
    }
    else
        myPhiGrid.drawGrid(renderer);
}

void LevelSet::drawGridPlane(Renderer& renderer, Axis planeAxis, double position, bool doOnlyNarrowBand) const
{
    position = std::clamp(position, double(0), double(1));

    Vec3i start = Vec3i::Zero();
    Vec3i end(myPhiGrid.size() - Vec3i::Ones());

    if (planeAxis == Axis::XAXIS)
    {
        start[0] = int(std::floor(position * double(myPhiGrid.size()[0] - 1)));
        end[0] = start[0] + 1;
    }
    else if (planeAxis == Axis::YAXIS)
    {
        start[1] = int(std::floor(position * double(myPhiGrid.size()[1] - 1)));
        end[1] = start[1] + 1;
    }
    else if (planeAxis == Axis::ZAXIS)
    {
        start[2] = int(std::floor(position * double(myPhiGrid.size()[2] - 1)));
        end[2] = start[2] + 1;
    }

    forEachVoxelRange(start, end, [&](const Vec3i& cell)
    {
        if (doOnlyNarrowBand)
        {
            if (std::fabs(myPhiGrid(cell)) < myNarrowBand) myPhiGrid.drawGridCell(renderer, cell);
        }
        else
            myPhiGrid.drawGridCell(renderer, cell);
    });
}

// Display a supersampled slice of the grid. The plane will have a normal in the plane_axis direction.
// The position is from [0,1] where 0 is at the grid origin and 1 is at the origin + size * dx.
void LevelSet::drawSupersampledValuesPlane(Renderer& renderer, Axis planeAxis, double position, double radius,
                                           int samples, double sampleSize) const
{
    myPhiGrid.drawSupersampledValuesPlane(renderer, planeAxis, position, radius, samples, sampleSize);
}
void LevelSet::drawSampleNormalsPlane(Renderer& renderer, Axis planeAxis, double position, const Vec3d& colour,
                                      double length) const
{
    myPhiGrid.drawSampleGradientsPlane(renderer, planeAxis, position, colour, length);
}

void LevelSet::drawSurface(Renderer& renderer, const Vec3d& colour, double lineWidth) const
{
    TriMesh tempMesh = buildMesh();
    tempMesh.drawMesh(renderer, true, colour, lineWidth);
}

// 
// Private methods
//

void LevelSet::initFromMeshImpl(const TriMesh& initialMesh, bool doResizeGrid)
{
    if (doResizeGrid)
    {
        // Determine the bounding box of the mesh to build the underlying grids
		AlignedBox3d bbox = initialMesh.boundingBox();

		// Expand grid beyond the narrow band of the mesh
		double maxPadding = 50. * dx();
		maxPadding = std::min(2. * myNarrowBand, maxPadding);

		bbox.extend(bbox.min() - Vec3d::Constant(maxPadding));
		bbox.extend(bbox.max() + Vec3d::Constant(maxPadding));

		Vec3d origin = indexToWorld(floor(worldToIndex(bbox.min())).eval());
		Transform xform(dx(), origin);
		Vec3d topRight = indexToWorld(ceil(worldToIndex(bbox.max())).eval());
  
		// TODO: add the ability to reset grid so we don't have ot re-allocate memory
		myPhiGrid = ScalarGrid<double>(xform, ((topRight - origin) / dx()).cast<int>(), myIsBackgroundNegative ? -myNarrowBand : myNarrowBand);
    }

    // We want to track which cells in the level set contain valid distance information.
    // The first pass will set cells close to the mesh as FINISHED.
    UniformGrid<VisitedCellLabels> reinitializedCells(size(), VisitedCellLabels::UNVISITED_CELL);
    UniformGrid<int> meshCellParities(size(), 0);

    for (const Vec3i& tri : initialMesh.triangles())
    {
        // It's easier to work in our index space and just scale the distance later.
        std::array<Vec3d, 3> triVertices;
        for (int localVertexIndex : {0, 1, 2})
            triVertices[localVertexIndex] = worldToIndex(initialMesh.vertex(tri[localVertexIndex]));

		AlignedBox3d triBbox(triVertices[0]);
		triBbox.extend(triVertices[1]);
		triBbox.extend(triVertices[2]);

        Vec3i triCeilMin = ceil(triBbox.min()).cast<int>();
        Vec3i triFloorMin = floor(triBbox.min()).cast<int>() - Vec3i::Ones();
        Vec3i triFloorMax = floor(triBbox.max()).cast<int>();

        // Z-axis intersection tests. Iterate along an aligned set of grid edges
        // in decsending order, checking for intersections at each edge.
        // If an intersection is found then we can stop searching along the set.
        for (int i = triCeilMin[0]; i <= triFloorMax[0]; ++i)
            for (int j = triFloorMin[1]; j <= triFloorMax[1]; ++j)
                for (int k = triFloorMax[2]; k >= triFloorMin[2]; --k)
                {
                    Vec3d gridPoint(i, j, k);
                    IntersectionLabels intersectionResult = exactTriIntersect(gridPoint, triVertices[0], triVertices[1], triVertices[2], Axis::ZAXIS);

                    for (int axis : {0, 1, 2})
				        assert(gridPoint[axis] >= 0 && gridPoint[axis] < myPhiGrid.size()[axis] - 1);

                    if (intersectionResult == IntersectionLabels::NO) continue;

                    int parityChange = -1;
                    double qrs = orient2d(triVertices[0].data(), triVertices[1].data(), triVertices[2].data());
                    assert(qrs != 0);
                    if (qrs < 0)
                        parityChange = 1;

                    if (intersectionResult == IntersectionLabels::YES)
                        meshCellParities(i, j, k + 1) += parityChange;
                    else
                    {
                        assert(intersectionResult == IntersectionLabels::ON);

                        if (parityChange == 1)
                            meshCellParities(i, j, k) += parityChange;
                        else
                        {
                            meshCellParities(i, j, k + 1) += parityChange;
                        }                      
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

                if (parity > 0)
                    myPhiGrid(cell) = -myNarrowBand;
                else
                    myPhiGrid(cell) = myNarrowBand;
            }

            assert(myIsBackgroundNegative ? parity == 1 : parity == 0);
        }

    // With the parity assigned, loop over the grid once more and label nodes that have a sign change
    // with neighbouring nodes (this means parity goes from -'ve (and zero) to +'ve or vice versa).
    forEachVoxelRange(Vec3i::Ones(), size() - Vec3i::Ones(), [&](const Vec3i& cell)
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
    for (const Vec3i& tri : initialMesh.triangles())
    {
        std::array<Vec3d, 3> vertices;
        for (int localVertexIndex : {0, 1, 2})
            vertices[localVertexIndex] = worldToIndex(initialMesh.vertex(tri[localVertexIndex]));
            
        AlignedBox3d triBbox(vertices[0]);
        triBbox.extend(vertices[1]);
        triBbox.extend(vertices[2]);

        // Expand outward by 2-voxels in each direction
        triBbox.extend(triBbox.min() - Vec3d::Constant(2));
        triBbox.extend(triBbox.max() + Vec3d::Constant(2));

        AlignedBox3d clampBbox;
        clampBbox.extend(Vec3d::Zero());
        clampBbox.extend(size().cast<double>() - Vec3d::Ones());

        triBbox.clamp(clampBbox);

        for (int axis : {0, 1, 2})
            assert(triBbox.min()[axis] >= 0 && triBbox.max()[axis] < size()[axis]);

        forEachVoxelRange(triBbox.min().cast<int>(), triBbox.max().cast<int>() + Vec3i::Ones(), [&](const Vec3i& cell)
        {
            if (reinitializedCells(cell) != VisitedCellLabels::UNVISITED_CELL)
            {
                Vec3d cellPoint = cell.cast<double>();

                Vec3d triProjectionPoint = pointToTriangleProjection(cellPoint, vertices[0], vertices[1], vertices[2]);

                double surfaceDistance = (cellPoint - triProjectionPoint).norm() * dx();

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

void LevelSet::reinitFastMarching(UniformGrid<VisitedCellLabels>& reinitializedCells)
{
    assert(reinitializedCells.size() == size());

    auto solveEikonal2D = [&](double Ux, double Uy) -> double
    {
        if (std::fabs(Ux - Uy) >= dx())
            return std::min(Ux, Uy) + dx();
        else
        {
            // Quadratic equation from the Eikonal
            double rootEntry = std::pow(Ux + Uy, 2) - 2. * (std::pow(Ux, 2) + std::pow(Uy, 2) - std::pow(dx(), 2));
            assert(rootEntry >= 0);
            return .5 * (Ux + Uy + std::sqrt(rootEntry));
        }
    };

    auto solveEikonal = [&](const Vec3i& cell) -> double
    {
        double max = std::numeric_limits<double>::max();

        Vec3d Uaxis = Vec3d::Constant(max);
		for (int axis : {0, 1, 2})
			for (int direction : {0, 1})
			{
				Vec3i adjacentCell = cellToCell(cell, axis, direction);

				if (adjacentCell[axis] < 0 || adjacentCell[axis] >= size()[axis])
				{
					assert(myPhiGrid(cell) > 0 && !myIsBackgroundNegative || myPhiGrid(cell) < 0 && myIsBackgroundNegative);
					Uaxis[axis] = std::min(max, Uaxis[axis]);
				}
				else
				{
					Uaxis[axis] = std::min(std::fabs(myPhiGrid(adjacentCell)), Uaxis[axis]);
				}
			}

        double discrim = std::pow(Uaxis.sum(), 2) - 3. * (Uaxis.squaredNorm() - std::pow(dx(), 2));
        if (discrim < 0.)
        {
            double dist = std::min(std::min(solveEikonal2D(Uaxis[0], Uaxis[1]), solveEikonal2D(Uaxis[1], Uaxis[2])), solveEikonal2D(Uaxis[0], Uaxis[2]));

            assert(std::isfinite(dist));

            return dist;
        }
        else
        {
            double dist = (Uaxis.sum() + std::sqrt(discrim)) / 3.;
            assert(std::isfinite(dist));
            return dist;
        }
    };

    // Load up the BFS queue with the unvisited cells next to the finished ones
    using Node = std::pair<Vec3i, double>;
    auto cmp = [](const Node& a, const Node& b) -> bool { return std::fabs(a.second) > std::fabs(b.second); };
    std::priority_queue<Node, std::vector<Node>, decltype(cmp)> marchingQ(cmp);

    forEachVoxelRange(Vec3i::Zero(), reinitializedCells.size(), [&](const Vec3i& cell)
    {
        if (reinitializedCells(cell) == VisitedCellLabels::FINISHED_CELL)
        {
            for (int axis : {0, 1, 2})
                for (int direction : {0, 1})
                {
                    Vec3i adjacentCell = cellToCell(cell, axis, direction);

                    if (adjacentCell[axis] < 0 || adjacentCell[axis] >= reinitializedCells.size()[axis]) continue;

                    if (reinitializedCells(adjacentCell) == VisitedCellLabels::UNVISITED_CELL)
                    {
                        double dist = solveEikonal(adjacentCell);

                        if (!std::isfinite(dist))
                            int a = 0;
                        assert(dist >= 0);

                        myPhiGrid(adjacentCell) = myPhiGrid(adjacentCell) < 0 ? -dist : dist;

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
        // and an older insert if doubleing around.
        if (reinitializedCells(localCell) == VisitedCellLabels::FINISHED_CELL)
        {
            // Make sure that the distance assigned to the cell is smaller than
            // what is doubleing around
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

                    if (adjacentCell[axis] < 0 || adjacentCell[axis] >= reinitializedCells.size()[axis]) continue;

                    if (reinitializedCells(adjacentCell) == VisitedCellLabels::FINISHED_CELL)
                        foundFinishedCell = true;
                    else
                    {
                        double dist = solveEikonal(adjacentCell);
                        assert(dist >= 0);

                        if (dist > myNarrowBand) dist = myNarrowBand;

                        if (reinitializedCells(adjacentCell) == VisitedCellLabels::VISITED_CELL &&
                            dist > std::fabs(myPhiGrid(adjacentCell)))
                            continue;

                        myPhiGrid(adjacentCell) = myPhiGrid(adjacentCell) < 0 ? -dist : dist;

                        Node node(adjacentCell, dist);

                        marchingQ.push(node);
                        reinitializedCells(adjacentCell) = VisitedCellLabels::VISITED_CELL;
                    }
                }
            assert(foundFinishedCell);
        }
        else
            myPhiGrid(localCell) = myPhiGrid(localCell) < 0 ? -myNarrowBand : myNarrowBand;

        reinitializedCells(localCell) = VisitedCellLabels::FINISHED_CELL;
    }
}

Vec3d LevelSet::findSurfaceIndex(const Vec3d& indexPoint, int iterationLimit, double tolerance) const
{
    Vec3d worldPoint = indexToWorld(indexPoint);
    worldPoint = findSurface(worldPoint, iterationLimit, tolerance);
    return worldToIndex(worldPoint);
}

}