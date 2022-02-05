#include "gtest/gtest.h"

#include <Eigen/Geometry>

#include "Predicates.h"
#include "Utilities.h"

using namespace FluidSim3D;

//
// Orient2d tests
//

TEST(PREDICATES_TEST, ORIENT_2D_ZERO_DET_TEST)
{
	exactinit();

	int testCases = 1000;
	for (int testIndex = 0; testIndex < testCases; ++testIndex)
	{
		Vec2d startPoint = 1e5 * Vec2d::Random();
		Vec2d endPoint = 1e5 * Vec2d::Random();

		Vec2d testPoint = startPoint;
		
		EXPECT_EQ(orient2d(&startPoint[0], &endPoint[0], &testPoint[0]), 0.);

		testPoint = endPoint;
		EXPECT_EQ(orient2d(&startPoint[0], &endPoint[0], &testPoint[0]), 0.);
	}
}

TEST(PREDICATES_TEST, ORIENT_2D_POSITIVE_DET_TEST)
{
	exactinit();

	int testCases = 1000;
	for (int testIndex = 0; testIndex < testCases; ++testIndex)
	{
		Vec2d startPoint = 1e5 * Vec2d::Random();
		Vec2d endPoint = 1e5 * Vec2d::Random();

		if (startPoint == endPoint)
			continue;

		Vec2d vec = endPoint - startPoint;
		Vec2d norm(-vec[1], vec[0]);

		double offsetScalar = 1e-12;
		Vec2d testPoint = startPoint + offsetScalar * norm;

		EXPECT_GT(orient2d(&startPoint[0], &endPoint[0], &testPoint[0]), 0.);

		testPoint = endPoint + offsetScalar * norm;
		EXPECT_GT(orient2d(&startPoint[0], &endPoint[0], &testPoint[0]), 0.);

		testPoint = .5 * (startPoint + endPoint) + offsetScalar * norm;
		EXPECT_GT(orient2d(&startPoint[0], &endPoint[0], &testPoint[0]), 0.);
	}
}

TEST(PREDICATES_TEST, ORIENT_2D_NEGATIVE_DET_TEST)
{
	exactinit();

	int testCases = 1000;
	for (int testIndex = 0; testIndex < testCases; ++testIndex)
	{
		Vec2d startPoint = 1e5 * Vec2d::Random();
		Vec2d endPoint = 1e5 * Vec2d::Random();

		if (startPoint == endPoint)
			continue;

		Vec2d vec = endPoint - startPoint;
		Vec2d norm(-vec[1], vec[0]);

		double offsetScalar = -1e-12;
		Vec2d testPoint = startPoint + offsetScalar * norm;

		EXPECT_LT(orient2d(&startPoint[0], &endPoint[0], &testPoint[0]), 0.);

		testPoint = endPoint + offsetScalar * norm;
		EXPECT_LT(orient2d(&startPoint[0], &endPoint[0], &testPoint[0]), 0.);

		testPoint = .5 * (startPoint + endPoint) + offsetScalar * norm;
		EXPECT_LT(orient2d(&startPoint[0], &endPoint[0], &testPoint[0]), 0.);
	}
}

//
// Orient3d tests
//

TEST(PREDICATES_TEST, ORIENT_3D_ZERO_DET_TEST)
{
	exactinit();

	int testCases = 1000;
	for (int testIndex = 0; testIndex < testCases; ++testIndex)
	{
		Vec3d v0 = 1e5 * Vec3d::Random();
		Vec3d v1 = 1e5 * Vec3d::Random();
        Vec3d v2 = 1e5 * Vec3d::Random();

		if (v0 == v1 || v1 == v2 || v2 == v0)
			continue;

		Vec3d testPoint = v0;
		EXPECT_EQ(orient3d(v0.data(), v1.data(), v2.data(), testPoint.data()), 0.);

        testPoint = v1;
        EXPECT_EQ(orient3d(v0.data(), v1.data(), v2.data(), testPoint.data()), 0.);

        testPoint = v2;
        EXPECT_EQ(orient3d(v0.data(), v1.data(), v2.data(), testPoint.data()), 0.);
	}
}

TEST(PREDICATES_TEST, ORIENT_3D_POSITIVE_DET_TEST)
{
	exactinit();

	int testCases = 1000;
	for (int testIndex = 0; testIndex < testCases; ++testIndex)
	{
		Vec3d v0 = 1e5 * Vec3d::Random();
		Vec3d v1 = 1e5 * Vec3d::Random();
        Vec3d v2 = 1e5 * Vec3d::Random();

		if (v0 == v1 || v1 == v2 || v2 == v0)
			continue;

        Vec3d vec0 = v1 - v0;
        Vec3d vec1 = v2 - v0;
        Vec3d norm = vec0.cross(vec1).normalized();

		double offsetScalar = 1e-10;

		Vec3d testPoint = v0 + offsetScalar * norm;
		EXPECT_LT(orient3d(v0.data(), v1.data(), v2.data(), testPoint.data()), 0.);

		testPoint = v1 + offsetScalar * norm;
		EXPECT_LT(orient3d(v0.data(), v1.data(), v2.data(), testPoint.data()), 0.);

        testPoint = v2 + offsetScalar * norm;
		EXPECT_LT(orient3d(v0.data(), v1.data(), v2.data(), testPoint.data()), 0.);

		testPoint = (1. / 3.) * (v0 + v1 + v2) + offsetScalar * norm;
		EXPECT_LT(orient3d(v0.data(), v1.data(), v2.data(), testPoint.data()), 0.);
	}
}

TEST(PREDICATES_TEST, ORIENT_3D_NEGATIVE_DET_TEST)
{
	exactinit();

	int testCases = 1000;
	for (int testIndex = 0; testIndex < testCases; ++testIndex)
	{
		Vec3d v0 = 1e5 * Vec3d::Random();
		Vec3d v1 = 1e5 * Vec3d::Random();
        Vec3d v2 = 1e5 * Vec3d::Random();

		if (v0 == v1 || v1 == v2 || v2 == v0)
			continue;

        Vec3d vec0 = v1 - v0;
        Vec3d vec1 = v2 - v0;
        Vec3d norm = vec0.cross(vec1).normalized();

		double offsetScalar = 1e-10;

		Vec3d testPoint = v0 - offsetScalar * norm;
		EXPECT_GT(orient3d(v0.data(), v1.data(), v2.data(), testPoint.data()), 0.);

		testPoint = v1 - offsetScalar * norm;
		EXPECT_GT(orient3d(v0.data(), v1.data(), v2.data(), testPoint.data()), 0.);

        testPoint = v2 - offsetScalar * norm;
		EXPECT_GT(orient3d(v0.data(), v1.data(), v2.data(), testPoint.data()), 0.);

		testPoint = (1. / 3.) * (v0 + v1 + v2) - offsetScalar * norm;
		EXPECT_GT(orient3d(v0.data(), v1.data(), v2.data(), testPoint.data()), 0.);
	}
}

TEST(PREDICATES_TEST, EXACT_TRIANGLE_INTERSECTION_CROSSING_TRIANGLE_TEST)
{
	exactinit();

	int testCases = 100;
	int offsetCases = 100;
	for (int testIndex = 0; testIndex < testCases; ++testIndex)
	{
		Vec3d rayStart = 1e5 * Vec3d::Random();

		for (auto axis : { Axis::XAXIS, Axis::YAXIS, Axis::ZAXIS })
		{
            int axisIndex = axis == Axis::XAXIS ? 0 : axis == Axis::YAXIS ? 1 : 2;

			Vec3d offset = 1e-10 * rayStart.cwiseAbs();

			for (int offsetIndex = 0; offsetIndex < offsetCases; ++offsetIndex)
			{
				Vec3d v0 = rayStart;
                v0[axisIndex] += offset[axisIndex] + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));

				Vec3d v1 = v0;
                v1[(axisIndex + 1) % 3] += offset[(axisIndex + 1) % 3] + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));

				Vec3d v2 = v0;
                v2[(axisIndex + 1) % 3] -= offset[(axisIndex + 1) % 3] + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));

                v0[(axisIndex + 2) % 3] += offset[(axisIndex + 2) % 3] + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));
                v1[(axisIndex + 2) % 3] -= offset[(axisIndex + 2) % 3] + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));
                v2[(axisIndex + 2) % 3] -= offset[(axisIndex + 2) % 3] + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));

				EXPECT_EQ(exactTriIntersect(rayStart, v0, v1, v2, axis), IntersectionLabels::YES);
	
				std::swap(v0, v1);

				EXPECT_EQ(exactTriIntersect(rayStart, v0, v1, v2, axis), IntersectionLabels::YES);
			}
		}
	}
}
//
//TEST(PREDICATES_TEST, EXACT_TRIANGLE_INTERSECTION_CROSSING_POINT_TEST)
//{
//	exactinit();
//
//	int testCases = 1000;
//	int offsetCases = 1000;
//	for (int testIndex = 0; testIndex < testCases; ++testIndex)
//	{
//		Vec3d rayStart = 1e5 * Vec2d::Random();
//
//		for (auto axis : { Axis::XAXIS, Axis::YAXIS, Axis::ZAXIS })
//		{
//			double offset = 1e-12 * rayStart.cwiseAbs().maxCoeff();
//
//            int axisIndex = axis == Axis::XAXIS ? 0 : axis == Axis::YAXIS ? 1 : 2;
//            
//            Vec3d v0 = rayStart;
//            v0[axisIndex] += offset + std::fabs(1e5 * double(std::rand()) / double(RAND_MAX));
//
//            
//
//			// Start down-left, end up-right
//			for (int offsetIndex = 0; offsetIndex < offsetCases; ++offsetIndex)
//			{
//				Vec2d startPoint = rayStart;
//				Vec2d endPoint = rayStart;
//
//				startPoint += Vec2d(-offset, -offset);
//				endPoint += Vec2d(offset, offset);
//
//				EXPECT_EQ(exactTriIntersect(startPoint, endPoint, rayStart, axis), IntersectionLabels::ON);
//
//				std::swap(startPoint, endPoint);
//
//				EXPECT_EQ(exactTriIntersect(startPoint, endPoint, rayStart, axis), IntersectionLabels::ON);
//			}
//
//			// Start up-left, end down-right
//			for (int offsetIndex = 0; offsetIndex < offsetCases; ++offsetIndex)
//			{
//				Vec2d startPoint = rayStart;
//				Vec2d endPoint = rayStart;
//
//				startPoint += Vec2d(-offset, offset);
//				endPoint += Vec2d(offset, -offset);
//
//				EXPECT_EQ(exactEdgeIntersect(startPoint, endPoint, rayStart, axis), IntersectionLabels::ON);
//
//				std::swap(startPoint, endPoint);
//
//				EXPECT_EQ(exactEdgeIntersect(startPoint, endPoint, rayStart, axis), IntersectionLabels::ON);
//			}
//
//			// Start up, end down
//			for (int offsetIndex = 0; offsetIndex < offsetCases; ++offsetIndex)
//			{
//				Vec2d startPoint = rayStart;
//				Vec2d endPoint = rayStart;
//
//				if (axis == Axis::XAXIS)
//				{
//					startPoint[1] += offset;
//					endPoint[1] -= offset;
//				}
//				else
//				{
//					startPoint[0] += offset;
//					endPoint[0] -= offset;
//				}
//
//				EXPECT_EQ(exactEdgeIntersect(startPoint, endPoint, rayStart, axis), IntersectionLabels::ON);
//
//				std::swap(startPoint, endPoint);
//
//				EXPECT_EQ(exactEdgeIntersect(startPoint, endPoint, rayStart, axis), IntersectionLabels::ON);
//			}
//		}
//	}
//}