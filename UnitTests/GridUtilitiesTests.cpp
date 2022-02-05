#include <array>
#include <numeric>

#include <gtest/gtest.h>

#include "GridUtilities.h"
#include "Utilities.h"

using namespace FluidSim3D;

TEST(GRID_UTILITIES_TESTS, CELL_TO_CELL_TEST)
{
	int testSize = 1000;
	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		Vec3i cell = (10000. * Vec3d::Random()).cast<int>();

		for (int axis : {0, 1, 2})
			for (int direction : {0, 1})
			{
				Vec3i adjacentCell = cellToCell(cell, axis, direction);
				Vec3i returnCell = cellToCell(adjacentCell, axis, (direction + 1) % 2);

				EXPECT_TRUE(cell == returnCell);
			}
	}
}

TEST(GRID_UTILITIES_TESTS, CELL_TO_FACE_TEST)
{
	int testSize = 1000;
	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		Vec3i cell = (10000. * Vec3d::Random()).cast<int>();

		for (int axis : {0, 1, 2})
			for (int direction : {0, 1})
			{
				Vec3i adjacentFace = cellToFace(cell, axis, direction);
				Vec3i returnCell = faceToCell(adjacentFace, axis, (direction + 1) % 2);

				EXPECT_TRUE(cell == returnCell);
			}
	}
}

TEST(GRID_UTILITIES_TESTS, CELL_TO_EDGE_TEST)
{
    int testSize = 1000;
	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		Vec3i cell = (10000. * Vec3d::Random()).cast<int>();

		for (int edgeAxis : {0, 1, 2})
			for (int edgeIndex : {0, 1, 2, 3})
			{
				Vec3i adjacentEdge = cellToEdge(cell, edgeAxis, edgeIndex);

                int cellIndex = 3 - edgeIndex;
				Vec3i returnCell = edgeToCell(adjacentEdge, edgeAxis, cellIndex);

				EXPECT_TRUE(cell == returnCell);
			}
	}
}


TEST(GRID_UTILITIES_TESTS, CELL_TO_NODE_TEST)
{
    int testSize = 1000;
	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		Vec3i cell = (10000. * Vec3d::Random()).cast<int>();

        for (int nodeIndex = 0; nodeIndex < 8; ++nodeIndex)
        {
            Vec3i adjacentNode = cellToNode(cell, nodeIndex);
            int cellIndex = 7 - nodeIndex;
            Vec3i returnCell = nodeToCell(adjacentNode, cellIndex);

            EXPECT_TRUE(cell == returnCell);
        }
    }
}

TEST(GRID_UTILITIES_TESTS, FACE_TO_EDGE)
{
	int testSize = 1000;
	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		Vec3i face = (10000. * Vec3d::Random()).cast<int>();

		for (int faceAxis : {0, 1, 2})
            for (int edgeAxisOffset : {1, 2})
            {
                int edgeAxis = (faceAxis + edgeAxisOffset) % 3;

                for (int direction : {0, 1})
                {
                    Vec3i adjacentEdge = faceToEdge(face, faceAxis, edgeAxis, direction);

                    Vec3i returnFace = edgeToFace(adjacentEdge, edgeAxis, faceAxis, (direction + 1) % 2);

                    EXPECT_TRUE(face == returnFace);
                }
            }
	}
}

TEST(GRID_UTILITIES_TESTS, FACE_TO_NODE)
{
	int testSize = 1000;
	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		Vec3i face = (10000. * Vec3d::Random()).cast<int>();

		for (int faceAxis : {0, 1, 2})
			for (int nodeIndex : {0, 1, 2, 3})
			{
                Vec3i adjacentNode = faceToNode(face, faceAxis, nodeIndex);

				int faceIndex = 3 - nodeIndex;
                Vec3i returnFace = nodeToFace(adjacentNode, faceAxis, faceIndex);

				EXPECT_TRUE(face == returnFace);
			}
	}
}

TEST(GRID_UTILITIES_TESTS, FACE_TO_NODE_CCW)
{
	int testSize = 1000;
	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		Vec3i face = (10000. * Vec3d::Random()).cast<int>();

		for (int faceAxis : {0, 1, 2})
			for (int nodeIndex : {0, 1, 2, 3})
			{
                Vec3i adjacentNode = faceToNode(face, faceAxis, nodeIndex);

                int returnFaceCount = 0;
				for (int faceIndex : {0, 1, 2, 3})
                {
                    Vec3i returnFace = nodeToFace(adjacentNode, faceAxis, faceIndex);
                    returnFaceCount += returnFace == face;
                }

				EXPECT_EQ(returnFaceCount, 1);
			}
	}
}

TEST(GRID_UTILITIES_TESTS, EDGE_TO_CELL_CCW)
{
	int testSize = 1000;
	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		Vec3i edge = (10000. * Vec3d::Random()).cast<int>();

		for (int edgeAxis : {0, 1, 2})
			for (int cellIndex : {0, 1, 2, 3})
			{
                Vec3i adjacentCell = edgeToCellCCW(edge, edgeAxis, cellIndex);

                int returnEdgeCount = 0;
				for (int edgeIndex : {0, 1, 2, 3})
                {
                    Vec3i returnEdge = cellToEdge(adjacentCell, edgeAxis, edgeIndex);
                    returnEdgeCount += returnEdge == edge;
                }

				EXPECT_EQ(returnEdgeCount, 1);
			}
	}
}

TEST(GRID_UTILITIES_TESTS, EDGE_TO_NODE)
{
	int testSize = 1000;
	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		Vec3i edge = (10000. * Vec3d::Random()).cast<int>();

		for (int edgeAxis : {0, 1, 2})
			for (int direction : {0, 1})
			{
                Vec3i adjacentNode = edgeToNode(edge, edgeAxis, direction);

                if (direction == 0)
                    EXPECT_TRUE(edge == adjacentNode);
                else
                {
                    Vec3i tempNode = edge;
                    ++tempNode[edgeAxis];

                    EXPECT_TRUE(adjacentNode == tempNode);
                }
			}
	}
}

TEST(GRID_UTILITIES_TESTS, LENGTH_FRACTION_HALF_TEST)
{
	{
		double theta = lengthFraction<double>(5., -5.);
		EXPECT_TRUE(isNearlyEqual(theta, .5));
	}

	{
		double theta = lengthFraction<double>(-5., 5.);
		EXPECT_TRUE(isNearlyEqual(theta, .5));
	}
	
	{
		double theta = lengthFraction<double>(1., -1.);
		EXPECT_TRUE(isNearlyEqual(theta, .5));
	}

	{
		double theta = lengthFraction<double>(-1., 1.);
		EXPECT_TRUE(isNearlyEqual(theta, .5));
	}
}

TEST(GRID_UTILITIES_TESTS, LENGTH_FRACTION_ZERO_TEST)
{
	{
		double theta = lengthFraction<double>(0., 1.);
		EXPECT_TRUE(isNearlyEqual(theta, 0.));
	}

	{
		double theta = lengthFraction<double>(1., 0.);
		EXPECT_TRUE(isNearlyEqual(theta, 0.));
	}

	{
		double theta = lengthFraction<double>(5., 1.);
		EXPECT_TRUE(isNearlyEqual(theta, 0.));
	}
}

TEST(GRID_UTILITIES_TESTS, LENGTH_FRACTION_ONE_TEST)
{
	{
		double theta = lengthFraction<double>(-1., 0.);
		EXPECT_TRUE(isNearlyEqual(theta, 1.));
	}

	{
		double theta = lengthFraction<double>(0., -1.);
		EXPECT_TRUE(isNearlyEqual(theta, 1.));
	}

	{
		double theta = lengthFraction<double>(-1., -5.);
		EXPECT_TRUE(isNearlyEqual(theta, 1.));
	}
}

TEST(GRID_UTILITIES_TESTS, FOR_EACH_VOXEL_RANGE_TEST)
{
    Vec3i size(10, 20, 30);
    std::vector<std::vector<std::vector<int>>> tempGrid(size[0]);

    for (auto& middleLayer : tempGrid)
    {
        middleLayer.resize(size[1]);

        for (auto& innerLayer : middleLayer)
            innerLayer.resize(size[2]);
    }

    int index = 0;
    forEachVoxelRange(Vec3i::Zero(), size, [&](const Vec3i& cell)
    {
        tempGrid[cell[0]][cell[1]][cell[2]] = index++;
    });

    EXPECT_EQ(tempGrid[0][0][0], 0);
    EXPECT_EQ(tempGrid[size[0] - 1][size[1] - 1][size[2] - 1], size.prod() - 1);
}

TEST(GRID_UTILITIES_TESTS, FOR_EACH_VOXEL_RANGE_REVERSE_TEST)
{
    Vec3i size(10, 20, 30);
    std::vector<std::vector<std::vector<int>>> tempGrid(size[0]);

    for (auto& middleLayer : tempGrid)
    {
        middleLayer.resize(size[1]);

        for (auto& innerLayer : middleLayer)
            innerLayer.resize(size[2]);
    }

    int index = 0;
    forEachVoxelRangeReverse(Vec3i::Zero(), size, [&](const Vec3i& cell)
    {
        tempGrid[cell[0]][cell[1]][cell[2]] = index++;
    });

    EXPECT_EQ(tempGrid[size[0] - 1][size[1] - 1][size[2] - 1], 0);
    EXPECT_EQ(tempGrid[0][0][0], size.prod() - 1);
}