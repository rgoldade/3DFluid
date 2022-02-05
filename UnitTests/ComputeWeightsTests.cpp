#include "gtest/gtest.h"

#include "ComputeWeights.h"
#include "InitialGeometry.h"
#include "LevelSet.h"
#include "Utilities.h"
#include "VectorGrid.h"

using namespace FluidSim3D;

//
// Ghost fluid weights tests
//

void testGhostFluidWeights(const LevelSet& surface)
{
	// Build ghost fluid weights
	VectorGrid<double> weights = computeGhostFluidWeights(surface);

	for (int axis : {0, 1, 2})
	{
		forEachVoxelRange(Vec3i::Zero(), weights.size(axis), [&](const Vec3i& face)
		{
			Vec3i backwardCell = faceToCell(face, axis, 0);
			Vec3i forwardCell = faceToCell(face, axis, 1);

			if (backwardCell[axis] < 0 || forwardCell[axis] == surface.size()[axis])
				EXPECT_EQ(weights(face, axis), 0);
			else if (surface(backwardCell) <= 0 && surface(forwardCell) <= 0)

				EXPECT_EQ(weights(face, axis), 1);
			else if (surface(backwardCell) > 0 && surface(forwardCell) > 0)
				EXPECT_EQ(weights(face, axis), 0);
			else
			{
				EXPECT_GE(weights(face, axis), 0);
				EXPECT_LE(weights(face, axis), 1);
			}
		});
	}
}

TEST(COMPUTE_WEIGHTS_TESTS, SPHERE_GHOST_FLUID_WEIGHT_TEST)
{
	double radius = 1;
	Vec3d center = Vec3d::Random();
	TriMesh mesh = makeSphereMesh(center, radius, 6);

	double dx = .025;
	Vec3d bottomLeft = center - 1.5 * Vec3d::Constant(radius);
	Vec3d topRight = center + 1.5 * Vec3d::Constant(radius);
	Vec3i gridSize = ((topRight - bottomLeft) / dx).cast<int>();

	Transform xform(dx, bottomLeft);
	LevelSet surface(xform, gridSize, 10);

	surface.initFromMesh(mesh);

	testGhostFluidWeights(surface);
}

TEST(COMPUTE_WEIGHTS_TESTS, CUBE_GHOST_FLUID_WEIGHT_TEST)
{
	Vec3d scale = Vec3d::Ones();
	Vec3d center = Vec3d::Random();
	TriMesh mesh = makeCubeMesh(center, scale);

	double dx = .025;
	Vec3d bottomLeft = center - 1.5 * scale;
	Vec3d topRight = center + 1.5 * scale;
	Vec3i gridSize = ((topRight - bottomLeft) / dx).cast<int>();

	Transform xform(dx, bottomLeft);
	LevelSet surface(xform, gridSize, 10);

	surface.initFromMesh(mesh);

	testGhostFluidWeights(surface);
}

//
// Cut-cell weights tests
//

void testCutCellWeights(const LevelSet& surface, bool invert)
{
    // Build ghost fluid weights
    VectorGrid<double> weights = computeCutCellWeights(surface, invert);

    for (int axis : {0, 1, 2})
    {
        forEachVoxelRange(Vec3i::Zero(), weights.size(axis), [&](const Vec3i& face)
        {
            std::array<Vec2d, 4> nodeOffset{ Vec2d(-.5, -.5), Vec2d(.5, -.5), Vec2d(-.5, .5), Vec2d(.5, .5) };
            std::array<double, 4> nodePhi;

            for (int nodeIndex : {0, 1, 2, 3})
            {
                Vec3d offset = Vec3d::Zero();
                offset[(axis + 1) % 3] = nodeOffset[nodeIndex][0];
                offset[(axis + 2) % 3] = nodeOffset[nodeIndex][1];

                Vec3d node = weights.indexToWorld(face.cast<double>() + offset, axis);
                    
                nodePhi[nodeIndex] = surface.triLerp(node);
            }

            if (nodePhi[0] <= 0 && nodePhi[1] <= 0 && nodePhi[2] <= 0 && nodePhi[3] <= 0)
            {
                if (invert)
                    EXPECT_EQ(weights(face, axis), 0);
                else
                    EXPECT_EQ(weights(face, axis), 1);
            }
            else if (nodePhi[0] > 0 && nodePhi[1] > 0 && nodePhi[2] > 0 && nodePhi[3] > 0)
            {
                if (invert)
                    EXPECT_EQ(weights(face, axis), 1);
                else
                    EXPECT_EQ(weights(face, axis), 0);
            }
            else
            {
                EXPECT_GT(weights(face, axis), 0);
                EXPECT_LT(weights(face, axis), 1);
            }
        });
    }
}

TEST(COMPUTE_WEIGHTS_TESTS, SPHERE_CUTCELL_WEIGHT_TEST)
{
	double radius = 1;
	Vec3d center = Vec3d::Random();
	TriMesh mesh = makeSphereMesh(center, radius, 6);

	double dx = .025;
	Vec3d bottomLeft = center - 1.5 * Vec3d::Constant(radius);
	Vec3d topRight = center + 1.5 * Vec3d::Constant(radius);
	Vec3i gridSize = ((topRight - bottomLeft) / dx).cast<int>();

	Transform xform(dx, bottomLeft);
	LevelSet surface(xform, gridSize, 10);

	surface.initFromMesh(mesh);

    testCutCellWeights(surface, false);
    testCutCellWeights(surface, true);
}

TEST(COMPUTE_WEIGHTS_TESTS, CUBE_CUTCELL_WEIGHT_TEST)
{
	Vec3d scale = Vec3d::Ones();
	Vec3d center = Vec3d::Random();
	TriMesh mesh = makeCubeMesh(center, scale);

	double dx = .025;
	Vec3d bottomLeft = center - 1.5 * scale;
	Vec3d topRight = center + 1.5 * scale;
	Vec3i gridSize = ((topRight - bottomLeft) / dx).cast<int>();

	Transform xform(dx, bottomLeft);
	LevelSet surface(xform, gridSize, 10);

	surface.initFromMesh(mesh);

    testCutCellWeights(surface, false);
    testCutCellWeights(surface, true);
}

//
// Super-sample volumes test
//

void testSupersampledVolumes(const LevelSet& surface, ScalarGridSettings::SampleType sampleType)
{
    // Build ghost fluid weights
    int samples = 3;
    ScalarGrid<double> weights(surface.xform(), surface.size(), sampleType);
    
    computeSupersampleVolumes(weights, surface, samples);

    double sampleDx = double(1) / double(samples);
    forEachVoxelRange(Vec3i::Zero(), weights.size(), [&](const Vec3i& coord)
    {
        int accum = 0;
        int sampleCount = 0;
        Vec3d startPoint = coord.cast<double>() - .5 * Vec3d::Ones() + .5 * Vec3d::Constant(sampleDx);
        Vec3d endPoint = coord.cast<double>() + .5 * Vec3d::Ones();

        for (Vec3d point = startPoint; point[0] <= endPoint[0]; point[0] += sampleDx)
            for (point[1] = startPoint[1]; point[1] <= endPoint[1]; point[1] += sampleDx)
                for (point[2] = startPoint[2]; point[2] <= endPoint[2]; point[2] += sampleDx)
                {
                    Vec3d samplePoint = weights.indexToWorld(point);

                    if (surface.triLerp(samplePoint) <= 0)
                        ++accum;

                    ++sampleCount;
                }

        double weight = double(accum) / double(sampleCount);

        EXPECT_TRUE(isNearlyEqual(weights(coord), weight));
    });
}

TEST(COMPUTE_WEIGHTS_TESTS, SPHERE_SUPERSAMPLE_WEIGHT_TEST)
{
    double radius = 1;
    Vec3d center = Vec3d::Random();
    TriMesh mesh = makeSphereMesh(center, radius, 6);

    double dx = .025;
    Vec3d bottomLeft = center - 1.5 * Vec3d::Constant(radius);
    Vec3d topRight = center + 1.5 * Vec3d::Constant(radius);
    Vec3i gridSize = ((topRight - bottomLeft) / dx).cast<int>();

    Transform xform(dx, bottomLeft);
    LevelSet surface(xform, gridSize, 10);

    surface.initFromMesh(mesh);

    testSupersampledVolumes(surface, ScalarGridSettings::SampleType::CENTER);
    testSupersampledVolumes(surface, ScalarGridSettings::SampleType::NODE);
}

TEST(COMPUTE_WEIGHTS_TESTS, CUBE_SUPERSAMPLE_WEIGHT_TEST)
{
    Vec3d scale = Vec3d::Ones();
    Vec3d center = Vec3d::Random();
    TriMesh mesh = makeCubeMesh(center, scale);

    double dx = .025;
    Vec3d bottomLeft = center - 1.5 * scale;
    Vec3d topRight = center + 1.5 * scale;
    Vec3i gridSize = ((topRight - bottomLeft) / dx).cast<int>();

    Transform xform(dx, bottomLeft);
    LevelSet surface(xform, gridSize, 10);

    surface.initFromMesh(mesh);

    testSupersampledVolumes(surface, ScalarGridSettings::SampleType::CENTER);
    testSupersampledVolumes(surface, ScalarGridSettings::SampleType::NODE);
}

//
// Super-sample face areas test
//

void testSupersampledFacesVolumes(const LevelSet& surface)
{
    // Build ghost fluid weights
    int samples = 3;
    VectorGrid<double> weights = computeSupersampledFaceVolumes(surface, samples);

    double sampleDx = double(1) / double(samples);
    for (int axis : {0, 1, 2})
    {
        forEachVoxelRange(Vec3i::Zero(), weights.size(axis), [&](const Vec3i& face)
        {
            int accum = 0;
            int sampleCount = 0;

            Vec3d start = face.cast<double>() - .5 * Vec3d::Ones() + .5 * Vec3d::Constant(sampleDx);
            Vec3d end = face.cast<double>() + .5 * Vec3d::Ones();

            for (Vec3d point = start; point[0] <= end[0]; point[0] += sampleDx)
                for (point[1] = start[1]; point[1] <= end[1]; point[1] += sampleDx)
                    for (point[2] = start[2]; point[2] <= end[2]; point[2] += sampleDx)
                    {
                        Vec3d samplePoint = weights.indexToWorld(point, axis);

                        if (surface.triLerp(samplePoint) <= 0)
                            ++accum;

                        ++sampleCount;
                    }

            double weight = double(accum) / double(sampleCount);
            
            EXPECT_TRUE(isNearlyEqual(weights(face, axis), weight)) << "Grid weight " << weights(face, axis) << ". Sampled weight " << weight;
        });
    }
}

TEST(COMPUTE_WEIGHTS_TESTS, SPHERE_SUPERSAMPLE_FACE_WEIGHT_TEST)
{
    double radius = 1;
    Vec3d center = Vec3d::Random();
    TriMesh mesh = makeSphereMesh(center, radius, 6);

    double dx = .025;
    Vec3d bottomLeft = center - 1.5 * Vec3d::Constant(radius);
    Vec3d topRight = center + 1.5 * Vec3d::Constant(radius);
    Vec3i gridSize = ((topRight - bottomLeft) / dx).cast<int>();

    Transform xform(dx, bottomLeft);
    LevelSet surface(xform, gridSize, 10);

    surface.initFromMesh(mesh);

    testSupersampledFacesVolumes(surface);
}

TEST(COMPUTE_WEIGHTS_TESTS, CUBE_SUPERSAMPLE_FACE_WEIGHT_TEST)
{
    Vec3d scale = Vec3d::Ones();
    Vec3d center = Vec3d::Random();
    TriMesh mesh = makeCubeMesh(center, scale);

    double dx = .025;
    Vec3d bottomLeft = center - 1.5 * scale;
    Vec3d topRight = center + 1.5 * scale;
    Vec3i gridSize = ((topRight - bottomLeft) / dx).cast<int>();

    Transform xform(dx, bottomLeft);
    LevelSet surface(xform, gridSize, 10);

    surface.initFromMesh(mesh);

    testSupersampledFacesVolumes(surface);
}