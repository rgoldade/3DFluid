#include "InitialGeometry.h"

#include <iostream>
#include <set>
#include <unordered_map>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

#include "LevelSet.h"

namespace FluidSim3D
{
TriMesh makeDiamondMesh(const Vec3d& center, double scale)
{
    VecVec3d vertices(6);

    vertices[0] = Vec3d(0., 1., 0.);
    vertices[1] = Vec3d(0., -1., 0.);
    vertices[2] = Vec3d(1., 0., 0.);

    vertices[3] = Vec3d(-1., 0., 0.);
    vertices[4] = Vec3d(0., 0., 1.);
    vertices[5] = Vec3d(0., 0., -1.);

    for (auto& vertex : vertices)
    {
        vertex *= scale;
    }
    for (auto& vertex : vertices)
    {
        vertex += center;
    }

    VecVec3i triFaces(8);

    triFaces[0] = Vec3i(2, 0, 4);
    triFaces[1] = Vec3i(4, 0, 3);
    triFaces[2] = Vec3i(3, 0, 5);
    triFaces[3] = Vec3i(5, 0, 2);
    triFaces[4] = Vec3i(2, 1, 5);
    triFaces[5] = Vec3i(5, 1, 3);
    triFaces[6] = Vec3i(3, 1, 4);
    triFaces[7] = Vec3i(4, 1, 2);

    return TriMesh(triFaces, vertices);
}

TriMesh makeCubeMesh(const Vec3d& center, const Vec3d& scale)
{
    VecVec3d vertices(8);

    vertices[0] = Vec3d(-1., -1., 1.);
    vertices[1] = Vec3d(1., -1., 1.);
    vertices[2] = Vec3d(-1., 1., 1.);
    vertices[3] = Vec3d(1., 1., 1.);

    vertices[4] = Vec3d(-1., 1., -1.);
    vertices[5] = Vec3d(1., 1., -1.);
    vertices[6] = Vec3d(-1., -1., -1.);
    vertices[7] = Vec3d(1., -1., -1.);

    for (auto& vertex : vertices)
    {
        vertex = vertex.cwiseProduct(scale);
    }
    for (auto& vertex : vertices)
    {
        vertex += center;
    }

    VecVec3i triFaces(12);

    triFaces[0] = Vec3i(0, 1, 2);
    triFaces[1] = Vec3i(2, 1, 3);
    triFaces[2] = Vec3i(2, 3, 4);
    triFaces[3] = Vec3i(4, 3, 5);

    triFaces[4] = Vec3i(4, 5, 6);
    triFaces[5] = Vec3i(6, 5, 7);
    triFaces[6] = Vec3i(6, 7, 0);
    triFaces[7] = Vec3i(0, 7, 1);

    triFaces[8] = Vec3i(1, 7, 3);
    triFaces[9] = Vec3i(3, 7, 5);
    triFaces[10] = Vec3i(6, 0, 4);
    triFaces[11] = Vec3i(4, 0, 2);

    return TriMesh(triFaces, vertices);
}

static void projectToUnitSphere(VecVec3d& points)
{
    tbb::parallel_for(tbb::blocked_range<int>(0, int(points.size())), [&](const tbb::blocked_range<int>& range)
    {
        for (int pointIndex = range.begin(); pointIndex != range.end(); ++pointIndex)
        {
            Vec3d& point = points[pointIndex];

            for (int iter = 0; iter < 10; ++iter)
            {
                if (std::fabs(1. - point.norm()) < 1e-12)
                    break;

                point -= (1. - 1. / point.norm()) * point;
            }
        }
    });
}

TriMesh makeIcosahedronMesh()
{
    double phi = (1. + std::sqrt(5.)) * .5;
    double a = 1.;
    double b = 1. / phi;

    VecVec3d vertices;

    vertices.emplace_back(0, b, -a);
    vertices.emplace_back(b, a, 0);
    vertices.emplace_back(-b, a, 0);
    vertices.emplace_back(0, b, a);
    vertices.emplace_back(0, -b, a);
    vertices.emplace_back(-a, 0, b);
    vertices.emplace_back(0, -b, -a);
    vertices.emplace_back(a, 0, -b);
    vertices.emplace_back(a, 0, b);
    vertices.emplace_back(-a, 0, -b);
    vertices.emplace_back(b, -a, 0);
    vertices.emplace_back(-b, -a, 0);

    projectToUnitSphere(vertices);

    VecVec3i triangles;
    // add triangles
    triangles.emplace_back(2, 1, 0);
    triangles.emplace_back(1, 2, 3);
    triangles.emplace_back(5, 4, 3);
    triangles.emplace_back(4, 8, 3);
    triangles.emplace_back(7, 6, 0);
    triangles.emplace_back(6, 9, 0);
    triangles.emplace_back(11, 10, 4);
    triangles.emplace_back(10, 11, 6);
    triangles.emplace_back(9, 5, 2);
    triangles.emplace_back(5, 9, 11);
    triangles.emplace_back(8, 7, 1);
    triangles.emplace_back(7, 8, 10);
    triangles.emplace_back(2, 5, 3);
    triangles.emplace_back(8, 1, 3);
    triangles.emplace_back(9, 2, 0);
    triangles.emplace_back(1, 7, 0);
    triangles.emplace_back(11, 9, 6);
    triangles.emplace_back(7, 10, 6);
    triangles.emplace_back(5, 11, 4);
    triangles.emplace_back(10, 8, 4);

    return TriMesh(triangles, vertices);
}

static TriMesh makeUnitSphereMesh(int subdivisions)
{
    TriMesh ico = makeIcosahedronMesh();

    VecVec3d vertices = ico.vertices();
    VecVec3i triangles = ico.triangles();

    struct VecHash
    {
        size_t operator()(const Vec2i& hashVec) const
        {
            size_t h = 0;
            for (int index : {0, 1})
            {
                h ^= std::hash<int>{}(hashVec[index])+0x9e3779b9 + (h << 6) + (h >> 2);
            }
            return h;
        }
    };

    std::unordered_map<Vec2i, int, VecHash> edgeSplitIndex;

    for (int iter = 0; iter < subdivisions; ++iter)
    {
        VecVec3i newTriangles;
        newTriangles.reserve(4 * triangles.size());

        edgeSplitIndex.clear();
        // Split edges and register in hash
        for (const Vec3i& tri : triangles)
        {
            for (int localEdgeIndex : {0, 1, 2})
            {
                Vec2i edge{ tri[localEdgeIndex], tri[(localEdgeIndex + 1) % 3] };

                if (edge[0] > edge[1])
                {
                    std::swap(edge[0], edge[1]);
                }
                
                // If the edge has not already been registered, add to hash table
                if (edgeSplitIndex.find(edge) == edgeSplitIndex.end())
                {
                    Vec3d edgeMidpoint = (vertices[edge[0]] + vertices[edge[1]]) / 2.;
                    edgeSplitIndex[edge] = int(vertices.size());
                    vertices.push_back(edgeMidpoint);                    
                }
            }
        }

        // Subdivide triangles
        for (const Vec3i& tri : triangles)
        {
            std::array<int, 3> midpointVertIndices;

            for (int localEdgeIndex : {0, 1, 2})
            {
                Vec2i edge(tri[localEdgeIndex], tri[(localEdgeIndex + 1) % 3]);

                if (edge[0] > edge[1])
                {
                    std::swap(edge[0], edge[1]);
                }

                assert(edgeSplitIndex.find(edge) != edgeSplitIndex.end());

                midpointVertIndices[localEdgeIndex] = int(edgeSplitIndex[edge]);
            }

            newTriangles.emplace_back(tri[0], midpointVertIndices[0], midpointVertIndices[2]);
            newTriangles.emplace_back(tri[1], midpointVertIndices[1], midpointVertIndices[0]);
            newTriangles.emplace_back(tri[2], midpointVertIndices[2], midpointVertIndices[1]);
            newTriangles.emplace_back(midpointVertIndices[0], midpointVertIndices[1], midpointVertIndices[2]);
        }

        std::swap(triangles, newTriangles);
    }

    projectToUnitSphere(vertices);

    return TriMesh(triangles, vertices);
}

TriMesh makeSphereMesh(const Vec3d& center, double radius, int subdivisions)
{
    TriMesh mesh = makeUnitSphereMesh(subdivisions);

    mesh.scale(radius);
    mesh.translate(center);

    return mesh;
}

}