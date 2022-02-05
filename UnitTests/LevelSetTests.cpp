#include "gtest/gtest.h"

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

#include "InitialGeometry.h"
#include "LevelSet.h"
#include "Transform.h"
#include "Utilities.h"

using namespace FluidSim3D;

TEST(LEVEL_SET_TESTS, INITIALIZE_TESTS)
{
    Vec3d origin = Vec3d::Random();
    double dx = .01;
    Transform xform(dx, origin);
    Vec3i gridSize = Vec3i::Constant(10);

    LevelSet surfaceGrid(xform, gridSize);

    EXPECT_EQ(surfaceGrid.dx(), dx);
    EXPECT_TRUE(surfaceGrid.offset() == origin);

    EXPECT_EQ(surfaceGrid.xform(), xform);
    EXPECT_TRUE(surfaceGrid.size() == gridSize);

    forEachVoxelRange(Vec3i::Zero(), gridSize, [&](const Vec3i& cell)
    {
        EXPECT_GT(surfaceGrid(cell), 0.);
    });

    LevelSet negativeBackgroundGrid(xform, gridSize, 10., true);

    EXPECT_TRUE(surfaceGrid.isGridMatched(negativeBackgroundGrid));

    forEachVoxelRange(Vec3i::Zero(), gridSize, [&](const Vec3i& cell)
    {
        EXPECT_LT(negativeBackgroundGrid(cell), 0.);
    });
}

template<typename IsoFunc>
static void testLevelSet(const IsoFunc& isoFunc, const TriMesh& mesh, double dx, bool rebuild)
{
    // Verify mesh lines up with the isosurface function
    for (const Vec3d& vertex : mesh.vertices())
    {
        EXPECT_TRUE(isNearlyEqual(isoFunc(vertex), 0., 1e-5, false));
    }

    // Get bounding box for mesh
    AlignedBox3d meshBBox = mesh.boundingBox();

    double bandwidth = 10;

    Vec3d origin = meshBBox.min() - 2. * bandwidth * dx * Vec3d::Ones();
    Vec3d topRight = meshBBox.max() + 2. * bandwidth * dx * Vec3d::Ones();

    Transform xform(dx, origin);

    Vec3i gridSize = ((topRight - origin) / dx).cast<int>();

    LevelSet surfaceGrid(xform, gridSize, bandwidth);

    // Set iso surface values to the grid
    tbb::parallel_for(tbb::blocked_range<int>(0, surfaceGrid.voxelCount(), tbbLightGrainSize),[&](const tbb::blocked_range<int>& range)
    {
        for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
        {
            Vec3i cell = surfaceGrid.unflatten(cellIndex);

            double phi = isoFunc(surfaceGrid.indexToWorld(cell.cast<double>()));

            surfaceGrid(cell) = phi;
        }
    });

    if (rebuild)
    {
        // Rebuild surface using fast marching
        surfaceGrid.reinit();

        // Verify narrow band enforced
        forEachVoxelRange(Vec3i::Zero(), surfaceGrid.size(), [&](const Vec3i& cell)
        {
            EXPECT_LE(std::fabs(surfaceGrid(cell)), bandwidth * dx);
        });
    }

    // Verify mesh points are on the zero isosurface
    for (const Vec3d& vertex : mesh.vertices())
    {
        EXPECT_TRUE(isNearlyEqual(surfaceGrid.triLerp(vertex), 0., 1e-3, false)) << surfaceGrid.triLerp(vertex);
    }

    // Verify the norm of the gradient is 1
    {
        double narrowBand = .5 * dx * bandwidth;
        forEachVoxelRange(Vec3i::Ones(), surfaceGrid.size() - Vec3i::Ones(), [&](const Vec3i& cell)
        {
            if (std::fabs(surfaceGrid(cell)) > narrowBand)
                return;

            Vec3d lerpNorm = surfaceGrid.normal(surfaceGrid.indexToWorld(cell.cast<double>()));

            EXPECT_TRUE(isNearlyEqual(lerpNorm.norm(), 1.)) << lerpNorm.norm();
        });
    }

    // Verify meshses generated from level set fall on isosurface
    {
        // Verify output mesh is close to the initial mesh
        TriMesh outputMesh = surfaceGrid.buildMesh();

        EXPECT_TRUE(outputMesh.unitTestMesh());

        // Verify mesh points are on the zero isosurface
        for (const Vec3d& vertex : outputMesh.vertices())
        {
            EXPECT_TRUE(isNearlyEqual(surfaceGrid.triLerp(vertex), 0., 1e-3, false)) << "Trilerp value: " << surfaceGrid.triLerp(vertex);
            EXPECT_TRUE(isNearlyEqual(isoFunc(vertex), 0., 1e-3, false));
        }
    }
}

TEST(LEVEL_SET_TESTS, SPHERE_ISO_SURFACE_TEST)
{
    double radius = 1;
    Vec3d center = Vec3d::Random();
    TriMesh mesh = makeSphereMesh(center, radius, 5);

    auto isoSurface = [&center, &radius](const Vec3d& point) -> double
    {
        return (point - center).norm() - radius;
    };

    double dx = .01;

    testLevelSet(isoSurface, mesh, dx, false);
}

TEST(LEVEL_SET_TESTS, CUBE_ISO_SURFACE_TEST)
{
    Vec3d scale = Vec3d::Ones();
    Vec3d center = Vec3d::Random();
    TriMesh mesh = makeCubeMesh(center, scale);

    auto isoSurface = [&center, &scale](const Vec3d& point) -> double
    {
        double dist = std::numeric_limits<double>::lowest();

        for (int axis : {0, 1, 2})
        {
            for (int direction : {0, 1})
            {
                double sign = direction == 0 ? -1 : 1;
                dist = std::max(dist, sign * ((point[axis] - center[axis]) - sign * scale[axis]));
            }
        }

        return dist;
    };

    double dx = .0025;

    testLevelSet(isoSurface, mesh, dx, false);
}

TEST(LEVEL_SET_TESTS, REBUILD_SPHERE_ISO_SURFACE_TEST)
{
    double radius = 1;
    Vec3d center = Vec3d::Random();
    TriMesh mesh = makeSphereMesh(center, radius, 5);

    auto isoSurface = [&center, &radius](const Vec3d& point) -> double
    {
        return (point - center).norm() - radius;
    };

    double dx = .01;

    testLevelSet(isoSurface, mesh, dx, true);
}

static void testUnionFunctions(const std::vector<std::function<double(const Vec3d&)>>& isoFuncs, const AlignedBox3d& bbox, double dx)
{
    // Build individual grids for each iso function
    Transform xform(dx, bbox.min());
    Vec3i gridSize = ceil(((bbox.max() - bbox.min()) / dx).eval()).cast<int>();
    double bandwidth = gridSize.maxCoeff();
    std::vector<LevelSet> surfaceGrids(isoFuncs.size(), LevelSet(xform, gridSize, bandwidth));

    for (int gridIndex = 0; gridIndex < surfaceGrids.size(); ++gridIndex)
    {
        LevelSet& grid = surfaceGrids[gridIndex];
        const auto& isoFunc = isoFuncs[gridIndex];

        tbb::parallel_for(tbb::blocked_range<int>(0, grid.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
        {
            for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
            {
                Vec3i cell = grid.unflatten(cellIndex);
                grid(cell) = isoFunc(grid.indexToWorld(cell.cast<double>()));
            }
        });
    }

    // Build union level set directly from lambdas
    LevelSet unionFuncGrid(xform, gridSize, bandwidth);

    for (int gridIndex = 0; gridIndex < surfaceGrids.size(); ++gridIndex)
    {
        ASSERT_TRUE(unionFuncGrid.isGridMatched(surfaceGrids[gridIndex]));
        EXPECT_TRUE(unionFuncGrid.narrowBand() == surfaceGrids[gridIndex].narrowBand());
    }

    tbb::parallel_for(tbb::blocked_range<int>(0, unionFuncGrid.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
    {
        for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
        {
            Vec3i cell = unionFuncGrid.unflatten(cellIndex);

            double minVal = std::numeric_limits<double>::max();
            for (const auto& isoFunc : isoFuncs)
            {
                double localPhi = isoFunc(unionFuncGrid.indexToWorld(cell.cast<double>()));
                minVal = std::min(localPhi, minVal);
            }

            unionFuncGrid(cell) = minVal;
        }
    });

    LevelSet unionGrid(xform, gridSize, bandwidth);

    ASSERT_TRUE(unionFuncGrid.isGridMatched(unionGrid));

    for (int gridIndex = 0; gridIndex < surfaceGrids.size(); ++gridIndex)
    {
        unionGrid.unionSurface(surfaceGrids[gridIndex]);
    }

    // Verify both unions match
    forEachVoxelRange(Vec3i::Zero(), unionGrid.size(), [&](const Vec3i& cell)
    {
        EXPECT_EQ(unionGrid(cell), unionFuncGrid(cell));
    });

    // Verify outer band of cells are "outside"
    for (int axis : {0, 1, 2})
        for (int direction : {0, 1})
        {
            Vec3i start = Vec3i::Zero();
            Vec3i end = unionGrid.size();

            if (direction == 0)
                end[axis] = 1;
            else
                start[axis] = unionGrid.size()[axis] - 1;

            forEachVoxelRange(start, end, [&](const Vec3i& cell)
            {
                ASSERT_GT(unionGrid(cell), 0.);
                ASSERT_GT(unionFuncGrid(cell), 0.);
            });
        }

    unionFuncGrid.reinit();
    unionGrid.reinit();

    // Verify both unions match after rebuilding
    forEachVoxelRange(Vec3i::Zero(), unionGrid.size(), [&](const Vec3i& cell)
    {
        EXPECT_EQ(unionGrid(cell), unionFuncGrid(cell));
    });

    // Verify inside / outside of union grid against iso functions
    forEachVoxelRange(Vec3i::Zero(), unionGrid.size(), [&](const Vec3i& cell)
    {
        Vec3d worldPoint = unionGrid.indexToWorld(cell.cast<double>());
        if (unionGrid(cell) > 0)
        {
            for (const auto& isoFunc : isoFuncs)
            {
                EXPECT_GT(isoFunc(worldPoint), 0.);
            }
        }
        else
        {
            int insideCount = 0;
            for (const auto& isoFunc : isoFuncs)
            {
                if (isoFunc(worldPoint) <= 0)
                    ++insideCount;
            }
            EXPECT_GT(insideCount, 0);
        }
    });

    // Verify the norm of the gradient is 1
    {
        double narrowBand = .5 * dx * bandwidth;
        forEachVoxelRange(Vec3i::Ones(), unionGrid.size() - Vec3i::Ones(), [&](const Vec3i& cell)
        {
            if (std::fabs(unionGrid(cell)) > narrowBand)
                return;

            Vec3d lerpNorm = unionGrid.normal(unionGrid.indexToWorld(cell.cast<double>()));

            EXPECT_TRUE(isNearlyEqual(lerpNorm.norm(), 1.)) << lerpNorm.norm();
        });
    }

    // Verify meshses generated from level set fall on isosurface
    {
        // Verify output mesh is close to the initial mesh
        TriMesh outputMesh = unionGrid.buildMesh();

        EXPECT_TRUE(outputMesh.unitTestMesh());

        // Verify mesh points are on the zero isosurface
        for (const Vec3d& vertex : outputMesh.vertices())
        {
            EXPECT_TRUE(isNearlyEqual(unionGrid.triLerp(vertex), 0., 1e-2, false)) << "Trilerp value: " << unionGrid.triLerp(vertex);

            double minVal = std::numeric_limits<double>::max();
            for (const auto& isoFunc : isoFuncs)
            {
                minVal = std::min(minVal, isoFunc(vertex));
            }

            EXPECT_TRUE(isNearlyEqual(minVal, 0., 1e-2, false)) << "Iso union value: " << minVal;
        }
    }
}

TEST(LEVEL_SET_TESTS, CIRCLE_UNION_TEST)
{
    auto circleIso = [](const Vec3d& center, const Vec3d& point, double radius)
    {
        return (point - center).norm() - radius;
    };

    auto iso0 = [&circleIso](const Vec3d& point) -> double
    {
        Vec3d center = Vec3d::Zero();
        double radius = 1;
        return circleIso(center, point, radius);
    };

    auto iso1 = [&circleIso](const Vec3d& point) -> double
    {
        Vec3d center = Vec3d::Constant(.5);
        double radius = 1;
        return circleIso(center, point, radius);
    };

    auto iso2 = [&circleIso](const Vec3d& point) -> double
    {
        Vec3d center(.75, 0, 0);
        double radius = 1;
        return circleIso(center, point, radius);
    };

    auto iso3 = [&circleIso](const Vec3d& point) -> double
    {
        Vec3d center(.5, -.5, -.5);
        double radius = 1;
        return circleIso(center, point, radius);
    };

    std::vector<std::function<double(const Vec3d&)>> isoFuncs;
    isoFuncs.push_back(iso0);
    isoFuncs.push_back(iso1);
    isoFuncs.push_back(iso2);
    isoFuncs.push_back(iso3);

    AlignedBox3d bbox(Vec3d::Constant(-2));
    bbox.extend(Vec3d::Constant(2));

    testUnionFunctions(isoFuncs, bbox, .025);
}

static void testInitFromMesh(const TriMesh& mesh, double dx)
{
    double bandwidth = 10;
    Transform xform(dx, Vec3d::Zero());
    LevelSet surfaceGrid(xform, Vec3i::Constant(100), bandwidth);

    surfaceGrid.initFromMesh(mesh, true);

    // Verify the mesh falls entirely inside the grid
    for (const Vec3d& vertex : mesh.vertices())
    {
        Vec3d indexPoint = surfaceGrid.worldToIndex(vertex);

        EXPECT_GT(indexPoint[0], 0.);
        EXPECT_GT(indexPoint[1], 0.);
        EXPECT_GT(indexPoint[2], 0.);
        EXPECT_LT(indexPoint[0], double(surfaceGrid.size()[0] - 1));
        EXPECT_LT(indexPoint[1], double(surfaceGrid.size()[1] - 1));
        EXPECT_LT(indexPoint[2], double(surfaceGrid.size()[2] - 1));
    }

    // Verify outer band of cells are "outside"
    for (int axis : {0, 1, 2})
        for (int direction : {0, 1})
        {
            Vec3i start = Vec3i::Zero();
            Vec3i end = surfaceGrid.size();

            if (direction == 0)
                end[axis] = 1;
            else
                start[axis] = surfaceGrid.size()[axis] - 1;

            forEachVoxelRange(start, end, [&](const Vec3i& cell)
            {
                ASSERT_GT(surfaceGrid(cell), 0.);
            });
        }

    // Verify mesh points are on the zero isosurface
    for (const Vec3d& vertex : mesh.vertices())
    {
        EXPECT_TRUE(isNearlyEqual(surfaceGrid.triLerp(vertex), 0., 2e-4, false)) << "Trilerp: " << surfaceGrid.triLerp(vertex);
    }

    // Verify the norm of the gradient is 1
    {
        double narrowBand = .5 * dx * bandwidth;
        forEachVoxelRange(Vec3i::Ones(), surfaceGrid.size() - Vec3i::Ones(), [&](const Vec3i& cell)
        {
            if (std::fabs(surfaceGrid(cell)) > narrowBand)
                return;

            Vec3d lerpNorm = surfaceGrid.normal(surfaceGrid.indexToWorld(cell.cast<double>()));

            EXPECT_TRUE(isNearlyEqual(lerpNorm.norm(), 1.)) << lerpNorm.norm();
        });
    }

    // Verify meshses generated from level set fall on isosurface
    {
        // Verify output mesh is close to the initial mesh
        TriMesh outputMesh = surfaceGrid.buildMesh();

        EXPECT_TRUE(outputMesh.unitTestMesh());

        // Verify mesh points are on the zero isosurface
        for (const Vec3d& vertex : outputMesh.vertices())
        {
            EXPECT_TRUE(isNearlyEqual(surfaceGrid.triLerp(vertex), 0., 2e-4, false)) << "Trilerp value: " << surfaceGrid.triLerp(vertex);
        }
    }
}

TEST(LEVEL_SET_TESTS, INIT_FROM_MESH_TEST)
{
    Vec3d origin = Vec3d::Random();
    double radius = 1.5;
    TriMesh mesh = makeSphereMesh(origin, radius, 10);

    double dx = .01;
    testInitFromMesh(mesh, dx);
}

static void testInitFromMeshUnion(const std::vector<TriMesh>& meshes, const std::vector<std::function<double(const Vec3d&)>>& isoFuncs, double dx)
{
    TriMesh unionMesh = meshes[0];
    unionMesh.insertMesh(meshes[1]);
    unionMesh.insertMesh(meshes[2]);
    unionMesh.insertMesh(meshes[3]);

    ASSERT_TRUE(unionMesh.unitTestMesh());

    double bandwidth = 100;
    Transform xform(dx, Vec3d::Zero());
    LevelSet surfaceGrid(xform, Vec3i::Constant(100), bandwidth);

    surfaceGrid.initFromMesh(unionMesh, true);

    // Verify the mesh falls entirely inside the grid
    for (const Vec3d& vertex : unionMesh.vertices())
    {
        Vec3d indexPoint = surfaceGrid.worldToIndex(vertex);

        EXPECT_GT(indexPoint[0], 0.);
        EXPECT_GT(indexPoint[1], 0.);
        EXPECT_GT(indexPoint[2], 0.);
        EXPECT_LT(indexPoint[0], double(surfaceGrid.size()[0] - 1));
        EXPECT_LT(indexPoint[1], double(surfaceGrid.size()[1] - 1));
        EXPECT_LT(indexPoint[2], double(surfaceGrid.size()[2] - 1));
    }

    // Verify outer band of cells are "outside"
    for (int axis : {0, 1, 2})
        for (int direction : {0, 1})
        {
            Vec3i start = Vec3i::Zero();
            Vec3i end = surfaceGrid.size();

            if (direction == 0)
                end[axis] = 1;
            else
                start[axis] = surfaceGrid.size()[axis] - 1;

            forEachVoxelRange(start, end, [&](const Vec3i& cell)
            {
                ASSERT_GT(surfaceGrid(cell), 0.);
            });
        }

    auto unionIsoFunc = [&isoFuncs](const Vec3d& point)
    {
        double minVal = std::numeric_limits<double>::max();

        for (auto& isoFunc : isoFuncs)
            minVal = std::min(minVal, isoFunc(point));

        return minVal;
    };

    // Verify mesh points fall on their own iso surface
    for (int meshIndex = 0; meshIndex < meshes.size(); ++meshIndex)
    {
        for (const Vec3d& vertex : meshes[meshIndex].vertices())
        {
            EXPECT_TRUE(isNearlyEqual(isoFuncs[meshIndex](vertex), 0., 1e-5, false)) << "Iso func val" << isoFuncs[meshIndex](vertex);
        }
    }

    // Verify the norm of the gradient is 1
    {
        double narrowBand = .5 * dx * bandwidth;
        forEachVoxelRange(Vec3i::Ones(), surfaceGrid.size() - Vec3i::Ones(), [&](const Vec3i& cell)
        {
            if (std::fabs(surfaceGrid(cell)) > narrowBand)
                return;

            Vec3d lerpNorm = surfaceGrid.normal(surfaceGrid.indexToWorld(cell.cast<double>()));

            EXPECT_TRUE(isNearlyEqual(lerpNorm.norm(), 1.)) << lerpNorm.norm();
        });
    }

    // Verify meshses generated from level set fall on isosurface
    {
        // Verify output mesh is close to the initial mesh
        TriMesh outputMesh = surfaceGrid.buildMesh();

        EXPECT_TRUE(outputMesh.unitTestMesh());

        // Verify mesh points are on the zero isosurface
        for (const Vec3d& vertex : outputMesh.vertices())
        {
            EXPECT_TRUE(isNearlyEqual(unionIsoFunc(vertex), 0., .1 * dx, false)) << "Iso value: " << unionIsoFunc(vertex);
            EXPECT_TRUE(isNearlyEqual(surfaceGrid.triLerp(vertex), 0., .1 * dx, false)) << "Trilerp value: " << surfaceGrid.triLerp(vertex);
        }
    }
}

TEST(LEVEL_SET_TESTS, UNION_INIT_FROM_MESH)
{
    double radius = 1;
    
    std::vector<Vec3d> centers{ Vec3d::Zero(), Vec3d::Constant(.5), Vec3d(.75, 0., 0.), Vec3d(.5, -.5, .5) };
    
    std::vector<TriMesh> meshes;

    TriMesh tempMesh = makeSphereMesh(Vec3d::Zero(), radius, 10);
    for (int meshIndex = 0; meshIndex < centers.size(); ++meshIndex)
    {
        meshes.push_back(tempMesh);
        meshes[meshIndex].translate(centers[meshIndex]);
    }

    auto circleIso = [](const Vec3d& center, const Vec3d& point, double radius)
    {
        return (point - center).norm() - radius;
    };
    
    auto iso0 = [&circleIso, &centers, radius](const Vec3d& point) -> double
    {
        return circleIso(centers[0], point, radius);
    };
    
    auto iso1 = [&circleIso, &centers, radius](const Vec3d& point) -> double
    {
        return circleIso(centers[1], point, radius);
    };
    
    auto iso2 = [&circleIso, &centers, radius](const Vec3d& point) -> double
    {
        return circleIso(centers[2], point, radius);
    };
    
    auto iso3 = [&circleIso, &centers, radius](const Vec3d& point) -> double
    {
        return circleIso(centers[3], point, radius);
    };
    
    std::vector<std::function<double(const Vec3d&)>> isoFuncs;
    isoFuncs.push_back(iso0);
    isoFuncs.push_back(iso1);
    isoFuncs.push_back(iso2);
    isoFuncs.push_back(iso3);

    double dx = .02;
    testInitFromMeshUnion(meshes, isoFuncs, dx);
}

// Test jittering mesh and iterating to the surface using gradients

static void testJitterMesh(const TriMesh& mesh, double dx)
{
    double bandwidth = 10;
    Transform xform(dx, Vec3d::Zero());
    LevelSet surfaceGrid(xform, Vec3i::Constant(100), bandwidth);

    surfaceGrid.initFromMesh(mesh, true);

    // Jitter vertex and then iterate it back to the zero isosurface
    for (Vec3d vertex : mesh.vertices())
    {
        vertex += 2. * dx * Vec3d::Random();

        vertex = surfaceGrid.findSurface(vertex, 100, 1e-5);

        EXPECT_TRUE(isNearlyEqual(surfaceGrid.triLerp(vertex), 0., 1e-5, false)) << "Trilerp: " << surfaceGrid.triLerp(vertex);
    }
}

TEST(LEVEL_SET_TESTS, FIND_SURFACE_TEST)
{
    Vec3d center = Vec3d::Random();
    double radius = 1.5;
    TriMesh mesh = makeSphereMesh(center, radius, 10);

    double dx = .01;
    testJitterMesh(mesh, dx);
}

// Test re-initializing with a mesh that extends outside of the grid

static void testOutOfBoundsMesh(const TriMesh& mesh, double dx)
{
    double bandwidth = 10;

    AlignedBox3d bbox = mesh.boundingBox();

    Vec3d origin = bbox.min() + 10. * dx * Vec3d::Ones();
    Vec3d topRight = bbox.max() - 10. * dx * Vec3d::Ones();
    Transform xform(dx, origin);

    Vec3i gridSize = ceil(((topRight - origin) / dx).eval()).cast<int>();
    LevelSet surfaceGrid(xform, gridSize, bandwidth);

    surfaceGrid.initFromMesh(mesh, false);

    // Verify the mesh falls entirely inside the grid
    bool outOfBounds = false;
    for (const Vec3d& vertex : mesh.vertices())
    {
        Vec3d indexPoint = surfaceGrid.worldToIndex(vertex);

        if (indexPoint[0] <= 0 || indexPoint[1] <= 0 || indexPoint[0] >= surfaceGrid.size()[0] - 1 || indexPoint[1] >= surfaceGrid.size()[1] - 1)
            outOfBounds = true;
    }

    EXPECT_TRUE(outOfBounds);

    // Verify outer band of cells are "outside" even though
    // the mesh falls outside of the grid
    for (int axis : {0, 1, 2})
        for (int direction : {0, 1})
        {
            Vec3i start = Vec3i::Zero();
            Vec3i end = surfaceGrid.size();

            if (direction == 0)
                end[axis] = 1;
            else
                start[axis] = surfaceGrid.size()[axis] - 1;

            forEachVoxelRange(start, end, [&](const Vec3i& cell) { ASSERT_GT(surfaceGrid(cell), 0.); });
        }
}

TEST(LEVEL_SET_TESTS, OUT_OF_BOUNDS_TEST)
{
    Vec3d center = Vec3d::Random();
    double radius = 1.5;
    TriMesh mesh = makeSphereMesh(center, radius, 5);

    double dx = .01;
    testOutOfBoundsMesh(mesh, dx);
}

TEST(LEVEL_SET_TESTS, ADVECT_TEST)
{
    Vec3d center = Vec3d::Random();
    double radius = 1.5;
    TriMesh mesh = makeSphereMesh(center, radius, 8);

    double dx = .01;
    LevelSet surfaceGrid(Transform(dx, Vec3d::Zero()), Vec3i::Constant(50), 20);
    surfaceGrid.initFromMesh(mesh, true);

    auto velFunc = [](double, const Vec3d&) { return Vec3d(-1., 1., 1.); };

    surfaceGrid.advectSurface(4. * dx, velFunc, IntegrationOrder::FORWARDEULER);
    surfaceGrid.reinit();

    mesh.advectMesh(4. * dx, velFunc, IntegrationOrder::FORWARDEULER);

    // Verify mesh points are on the zero isosurface or inside the unioned surface
    for (const Vec3d& vertex : mesh.vertices())
    {
        EXPECT_TRUE(isNearlyEqual(surfaceGrid.triLerp(vertex), 0., 1e-4, false)) << "Trilerp: " << surfaceGrid.triLerp(vertex);
    }
}
