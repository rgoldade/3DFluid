#include "gtest/gtest.h"

#include "InitialGeometry.h"
#include "TriMesh.h"
#include "Utilities.h"

using namespace FluidSim3D;

static void testTriMesh(const TriMesh& mesh)
{
   const VecVec3d& vertices = mesh.vertices();
   const VecVec3i& triangles = mesh.triangles();

   const std::vector<std::vector<int>>& adjacentTriangles = mesh.adjacentTriangles();

   // Verify that each vertex has two or more adjacent triangles.
   for (int vertIndex = 0; vertIndex < vertices.size(); ++vertIndex)
   {
       EXPECT_GE(adjacentTriangles[vertIndex].size(), 2);
   }

   // Verify that each vertices' adjacent triangles reciprocates
   for (int vertIndex = 0; vertIndex < vertices.size(); ++vertIndex)
   {
       for (int triIndex : adjacentTriangles[vertIndex])
       {
           const Vec3i& tri = triangles[triIndex];
           EXPECT_TRUE(tri[0] == vertIndex ||
                       tri[1] == vertIndex ||
                       tri[2] == vertIndex);
       }
   }

   // Verify triangles's adjacent vertex reciprocates
   for (int triIndex = 0; triIndex < triangles.size(); ++triIndex)
   {
       const Vec3i& tri = triangles[triIndex];
       for (int localTriIndex : {0, 1, 2})
       {
           int vertIndex = tri[localTriIndex];
           EXPECT_TRUE(std::find(adjacentTriangles[vertIndex].begin(), adjacentTriangles[vertIndex].end(), triIndex) != adjacentTriangles[vertIndex].end());
       }
   }

   // Bounding box test
   AlignedBox3d meshBBox = mesh.boundingBox();

   for (const Vec3d& vertex : mesh.vertices())
   {
       EXPECT_GE(vertex[0], meshBBox.min()[0]);
       EXPECT_GE(vertex[1], meshBBox.min()[1]);
       EXPECT_GE(vertex[2], meshBBox.min()[2]);
       
       EXPECT_LE(vertex[0], meshBBox.max()[0]);
       EXPECT_LE(vertex[1], meshBBox.max()[1]);
       EXPECT_LE(vertex[2], meshBBox.max()[2]);
   }
}

TEST(TRI_MESH_TESTS, DIAMOND_MESH_TEST)
{
   Vec3d center = Vec3d::Random();
   TriMesh mesh = makeDiamondMesh(center, 1.25);
   testTriMesh(mesh);
}

TEST(TRI_MESH_TESTS, CUBE_MESH_TEST)
{
   Vec3d scale = Vec3d::Random() + 1.5 * Vec3d::Ones();
   Vec3d center = Vec3d::Random();
   TriMesh mesh = makeCubeMesh(center, scale);
   testTriMesh(mesh);

   // Verify normals
   for (int triIndex = 0; triIndex < mesh.triangleCount(); ++triIndex)
   {
       const Vec3i& tri = mesh.triangle(triIndex);
       Vec3d midPoint = (mesh.vertex(tri[0]) + mesh.vertex(tri[1]) + mesh.vertex(tri[2])) / 3.;

       Vec3d offset = midPoint - center;
       Vec3d normal = mesh.normal(triIndex);

       EXPECT_GT(offset.dot(normal), 0);
   }
}

TEST(TRI_MESH_TESTS, ICOSAHEDRON_MESH_TEST)
{
   TriMesh mesh = makeIcosahedronMesh();
   testTriMesh(mesh);
}

TEST(TRI_MESH_TESTS, SPHERE_MESH_TEST)
{
   double radius = 1.25;
   Vec3d center = Vec3d::Random();
   int subdivs = 3;
   TriMesh mesh = makeSphereMesh(center, radius, subdivs);
   testTriMesh(mesh);

   for (int triIndex = 0; triIndex < mesh.triangleCount(); ++triIndex)
   {
       const Vec3i& tri = mesh.triangle(triIndex);
       Vec3d midPoint = (mesh.vertex(tri[0]) + mesh.vertex(tri[1]) + mesh.vertex(tri[2])) / 3.;

       Vec3d offsetNormal = (midPoint - center).normalized();
       Vec3d normal = mesh.normal(triIndex);

       EXPECT_GT(offsetNormal.dot(normal), .9);
   }
}

TEST(TRI_MESH_TESTS, COPY_MESH_TEST)
{
   double radius = 1.25;
   Vec3d center = Vec3d::Random();
   int subdivs = 3;
   TriMesh mesh = makeSphereMesh(center, radius, subdivs);
   TriMesh copyMesh = mesh;

   EXPECT_EQ(mesh.triangleCount(), copyMesh.triangleCount());
   EXPECT_EQ(mesh.vertexCount(), copyMesh.vertexCount());

   for (int triIndex = 0; triIndex < mesh.triangleCount(); ++triIndex)
   {
       EXPECT_TRUE(mesh.triangle(triIndex) == copyMesh.triangle(triIndex));
   }

   for (int vertIndex = 0; vertIndex < mesh.vertexCount(); ++vertIndex)
   {
       EXPECT_TRUE(mesh.vertex(vertIndex) == copyMesh.vertex(vertIndex));
   }
}

TEST(TRI_MESH_TESTS, REINITIALIZE_MESH_TEST)
{
   double radius = 1.25;
   Vec3d center = Vec3d::Random();
   int subdivs = 3;
   TriMesh mesh = makeSphereMesh(center, radius, subdivs);

   TriMesh reinitMesh = makeSphereMesh(center + Vec3d::Ones(), 2. * radius, subdivs);

   mesh.reinitialize(reinitMesh.triangles(), reinitMesh.vertices());

   EXPECT_EQ(mesh.triangleCount(), reinitMesh.triangleCount());
   EXPECT_EQ(mesh.vertexCount(), reinitMesh.vertexCount());

   for (int triIndex = 0; triIndex < mesh.triangleCount(); ++triIndex)
   {
       EXPECT_TRUE(mesh.triangle(triIndex) == reinitMesh.triangle(triIndex));
   }

   for (int vertIndex = 0; vertIndex < mesh.vertexCount(); ++vertIndex)
   {
       EXPECT_TRUE(mesh.vertex(vertIndex) == reinitMesh.vertex(vertIndex));
   }
}

TEST(TRI_MESH_TESTS, INSERT_MESH_TEST)
{
   double radius = 1.25;
   Vec3d center = Vec3d::Random();
   int subdivs = 3;
   TriMesh mesh = makeSphereMesh(center, radius, subdivs);

   TriMesh copyMesh = mesh;

   Vec3d scale = Vec3d::Random() + 1.5 * Vec3d::Ones();
   center = Vec3d::Random();
   TriMesh tempMesh = makeCubeMesh(center, scale);

   mesh.insertMesh(tempMesh);

   EXPECT_TRUE(mesh.unitTestMesh());

   EXPECT_EQ(mesh.triangleCount(), copyMesh.triangleCount() + tempMesh.triangleCount());
   EXPECT_EQ(mesh.vertexCount(), copyMesh.vertexCount() + tempMesh.vertexCount());

   for (int triIndex = 0; triIndex < mesh.triangleCount(); ++triIndex)
   {
       if (triIndex < copyMesh.triangleCount())
       {
           EXPECT_TRUE(mesh.triangle(triIndex) == copyMesh.triangle(triIndex));
       }
       else
       {
           Vec3i tri = mesh.triangle(triIndex) - Vec3i::Constant(copyMesh.vertexCount());
           EXPECT_TRUE(tri == tempMesh.triangle(triIndex - copyMesh.triangleCount()));
       }
   }

   for (int vertIndex = 0; vertIndex < mesh.vertexCount(); ++vertIndex)
   {
       if (vertIndex < copyMesh.vertexCount())
       {
           EXPECT_TRUE(mesh.vertex(vertIndex) == copyMesh.vertex(vertIndex));
       }
       else
       {
           EXPECT_TRUE(mesh.vertex(vertIndex) == tempMesh.vertex(vertIndex - copyMesh.vertexCount()));
       }
   }
}

TEST(TRI_MESH_TESTS, SET_VERTEX_TEST)
{
   double radius = 1.25;
   Vec3d center = Vec3d::Random();
   int subdivs = 3;
   TriMesh mesh = makeSphereMesh(center, radius, subdivs);

   TriMesh biggerMesh = makeSphereMesh(center, 1.2 * radius, subdivs);

   EXPECT_EQ(mesh.triangleCount(), biggerMesh.triangleCount());
   EXPECT_EQ(mesh.vertexCount(), biggerMesh.vertexCount());

   for (int vertIndex = 0; vertIndex < mesh.vertexCount(); ++vertIndex)
   {
       mesh.setVertex(vertIndex, biggerMesh.vertex(vertIndex));
   }

   for (int vertIndex = 0; vertIndex < mesh.vertexCount(); ++vertIndex)
   {
       EXPECT_TRUE(mesh.vertex(vertIndex) == biggerMesh.vertex(vertIndex));
   }
}

TEST(TRI_MESH_TESTS, NORMAL_TEST)
{
   double radius = 1.25;
   Vec3d center = Vec3d::Random();

   std::vector<double> maxErrors;
   for (int subdiv = 3; subdiv < 7; ++subdiv)
   {
       TriMesh lowResMesh = makeSphereMesh(center, radius, subdiv);

       double maxLowResNormError = 0;

       for (int triIndex = 0; triIndex < lowResMesh.triangleCount(); ++triIndex)
       {
           const Vec3i& tri = lowResMesh.triangle(triIndex);

           Vec3d meshNorm = lowResMesh.normal(triIndex);

           Vec3d midPoint = (lowResMesh.vertex(tri[0]) + lowResMesh.vertex(tri[1]) + lowResMesh.vertex(tri[2])) / 3.;
           Vec3d midPointNormal = (midPoint - center).normalized();

           maxLowResNormError = std::max(maxLowResNormError, (meshNorm - midPointNormal).norm());
       }

       maxErrors.push_back(maxLowResNormError);
   }

   for (int errorIndex = 1; errorIndex < maxErrors.size(); ++errorIndex)
   {
       EXPECT_GT(maxErrors[errorIndex - 1] / maxErrors[errorIndex], 1.8);
   }
}

TEST(TRI_MESH_TESTS, REVERSE_MESH_TEST)
{
    double radius = 1;
    Vec3d center = Vec3d::Random();
    
    TriMesh mesh = makeSphereMesh(center, radius, 3);

    TriMesh reverseMesh = mesh;
    reverseMesh.reverse();

    EXPECT_EQ(mesh.triangleCount(), reverseMesh.triangleCount());
    EXPECT_EQ(mesh.vertexCount(), reverseMesh.vertexCount());

    for (int triIndex = 0; triIndex < mesh.triangleCount(); ++triIndex)
    {
        double normDot = mesh.normal(triIndex).dot(reverseMesh.normal(triIndex));
        EXPECT_TRUE(isNearlyEqual(normDot, -1.));
    }
}

TEST(TRI_MESH_TESTS, DEGENERATE_TEST)
{
    double radius = 1;
    Vec3d center = Vec3d::Random();

    TriMesh mesh = makeSphereMesh(center, radius, 3);

    for (int triIndex = 0; triIndex < mesh.triangleCount(); ++triIndex)
    {
        EXPECT_FALSE(mesh.isTriangleDegenerate(triIndex));
    }

    for (int vertIndex = 0; vertIndex < mesh.vertexCount(); ++vertIndex)
    {
        mesh.setVertex(vertIndex, center);
    }

    for (int triIndex = 0; triIndex < mesh.triangleCount(); ++triIndex)
    {
        EXPECT_TRUE(mesh.isTriangleDegenerate(triIndex));
    }
}

TEST(TRI_MESH_TESTS, ADVECT_TEST)
{
    double radius = 1;
    Vec3d center = Vec3d::Random();

    TriMesh mesh = makeSphereMesh(center, radius, 3);

    TriMesh copyMesh = mesh;

    auto velocity = [&](const double, const Vec3d& point) -> Vec3d { return point - center; };

    mesh.advectMesh(1., velocity, IntegrationOrder::FORWARDEULER);

    for (int vertIndex = 0; vertIndex < copyMesh.vertexCount(); ++vertIndex)
    {
        Vec3d copyVertex = copyMesh.vertex(vertIndex) + velocity(0., copyMesh.vertex(vertIndex));
        Vec3d vertex = mesh.vertex(vertIndex);
        EXPECT_TRUE(isNearlyEqual(vertex[0], copyVertex[0]));
        EXPECT_TRUE(isNearlyEqual(vertex[1], copyVertex[1]));
        EXPECT_TRUE(isNearlyEqual(vertex[2], copyVertex[2]));
    }
}

TEST(TRI_MESH_TESTS, BARYCENTER_WEIGHT_TEST)
{
    double radius = 1.25;
    Vec3d center = Vec3d::Random();
    int subdivs = 3;
    TriMesh mesh = makeSphereMesh(center, radius, subdivs);
    testTriMesh(mesh);

    for (const Vec3i& tri : mesh.triangles())
    {
        {
            // Test midpoint
            Vec3d midpoint = (mesh.vertex(tri[0]) + mesh.vertex(tri[1]) + mesh.vertex(tri[2])) / 3.;

            Vec3d barycenterWeights = computeBarycenters(midpoint, mesh.vertex(tri[0]), mesh.vertex(tri[1]), mesh.vertex(tri[2]));

            EXPECT_TRUE(isNearlyEqual(barycenterWeights[0], 1. / 3., 1e-5, false));
            EXPECT_TRUE(isNearlyEqual(barycenterWeights[1], 1. / 3., 1e-5, false));
            EXPECT_TRUE(isNearlyEqual(barycenterWeights[2], 1. / 3., 1e-5, false));
        }

        for (int vertIndex : {0, 1, 2})
        {
            // Test vertex
            Vec3d point = mesh.vertex(tri[vertIndex]);

            Vec3d barycenterWeights = computeBarycenters(point, mesh.vertex(tri[0]), mesh.vertex(tri[1]), mesh.vertex(tri[2]));

            EXPECT_TRUE(isNearlyEqual(barycenterWeights[vertIndex], 1., 1e-5, false));
            EXPECT_TRUE(isNearlyEqual(barycenterWeights[(vertIndex + 1) % 3], 0., 1e-5, false));
            EXPECT_TRUE(isNearlyEqual(barycenterWeights[(vertIndex + 2) % 3], 0., 1e-5, false));
        }
    }
}