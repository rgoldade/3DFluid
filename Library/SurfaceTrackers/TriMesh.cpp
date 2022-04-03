#include "TriMesh.h"

#include <fstream>
#include <iostream>
#include <set>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"
#include "tbb/parallel_sort.h"

namespace FluidSim3D
{

TriMesh::TriMesh(const VecVec3i& triangles, const VecVec3d& vertices)
{
    initialize(triangles, vertices);
}

void TriMesh::reinitialize(const VecVec3i& triangles, const VecVec3d& vertices)
{
    initialize(triangles, vertices);
}

void TriMesh::insertMesh(const TriMesh& mesh)
{
    size_t triCount = myTriangles.size();
    size_t vertexCount = myVertices.size();

    myVertices.insert(myVertices.end(), mesh.myVertices.begin(), mesh.myVertices.end());
    myTriangles.insert(myTriangles.end(), mesh.myTriangles.begin(), mesh.myTriangles.end());

    tbb::parallel_for(tbb::blocked_range<size_t>(triCount, myTriangles.size(), tbbLightGrainSize), [&](const tbb::blocked_range<size_t>& range)
    {
        for (size_t triIndex = range.begin(); triIndex != range.end(); ++triIndex)
            for (int localVertexIndex : {0, 1, 2})
            {
                int vertexIndex = myTriangles[triIndex][localVertexIndex];
                myTriangles[triIndex][localVertexIndex] = vertexIndex + int(vertexCount);

                assert(vertexIndex >= 0 && vertexIndex < mesh.vertexCount());
                assert(myTriangles[triIndex][localVertexIndex] < myVertices.size());
            }
    });

    buildAdjacentTriangles();
}

const VecVec3i& TriMesh::triangles() const
{
    return myTriangles;
}

const VecVec3d& TriMesh::vertices() const
{
    return myVertices;
}

const std::vector<std::vector<int>>& TriMesh::adjacentTriangles() const
{
    return myAdjacentTriangles;
}

void TriMesh::clear()
{
    myVertices.clear();
    myTriangles.clear();
}

void TriMesh::reverse()
{
    tbb::parallel_for(tbb::blocked_range<size_t>(0, myTriangles.size(), tbbLightGrainSize), [&](const tbb::blocked_range<size_t>& range)
    {
        for (size_t triIndex = range.begin(); triIndex != range.end(); ++triIndex)
            std::swap(myTriangles[triIndex][0], myTriangles[triIndex][1]);
    });
}

void TriMesh::scale(double s)
{
    tbb::parallel_for(tbb::blocked_range<size_t>(0, myVertices.size(), tbbLightGrainSize), [&](const tbb::blocked_range<size_t>& range)
    {
        for (size_t vertexIndex = range.begin(); vertexIndex != range.end(); ++vertexIndex)
            myVertices[vertexIndex] *= s;
    });
}

void TriMesh::translate(const Vec3d& t)
{
    tbb::parallel_for(tbb::blocked_range<size_t>(0, myVertices.size(), tbbLightGrainSize), [&](const tbb::blocked_range<size_t>& range)
    {
        for (size_t vertexIndex = range.begin(); vertexIndex != range.end(); ++vertexIndex)
            myVertices[vertexIndex] += t;
    });
}

bool TriMesh::isTriangleDegenerate(int triIndex) const
{
    const Vec3i& tri = myTriangles[triIndex];

    return (myVertices[tri[0]] == myVertices[tri[1]] ||
            myVertices[tri[1]] == myVertices[tri[2]] ||
            myVertices[tri[2]] == myVertices[tri[0]]);
}

AlignedBox3d TriMesh::boundingBox() const
{
    AlignedBox3d bbox = tbb::parallel_reduce(tbb::blocked_range<int>(0, int(myVertices.size()), tbbLightGrainSize), AlignedBox3d(myVertices[0]), [&](const tbb::blocked_range<int>& range, AlignedBox3d bbox)
    {
        for (int vertIndex = range.begin(); vertIndex != range.end(); ++vertIndex)
        {
            bbox.extend(myVertices[vertIndex]);
        }

        return bbox;
    },
    [](AlignedBox3d bbox0, const AlignedBox3d& bbox1)
    {
        bbox0.extend(bbox1);
        return bbox0;
    });

    return bbox;
}

void TriMesh::drawMesh(Renderer& renderer, bool doRenderTriFaces, Vec3d triFaceColour, bool doRenderTriNormals, Vec3d normalColour, bool doRenderVertices, Vec3d vertexColour,
                       bool doRenderTriEdges, Vec3d edgeColour, bool doRenderVertexNormals, Vec3d vertexNormalColours)
{    
    // Pre-compute area-weighted triangle normals
    // and set triangle faces
    VecVec3d weightedTriNormals(myTriangles.size(), Vec3d::Zero());

    // Render triangles
    VecVec3d vertexNormals(myVertices.size(), Vec3d::Zero());

    tbb::parallel_for(tbb::blocked_range<size_t>(0, myTriangles.size(), tbbLightGrainSize), [&](const tbb::blocked_range<size_t>& range)
    {
        for (size_t triIndex = range.begin(); triIndex != range.end(); ++triIndex)
        {
            weightedTriNormals[triIndex] = scaledNormal(int(triIndex));
        }
    });

    // Accumulate area-weight triangle normals to each vertex and normalize.
    // Set vertex points.
    tbb::parallel_for(tbb::blocked_range<size_t>(0, myVertices.size(), tbbLightGrainSize), [&](const tbb::blocked_range<size_t>& range)
    {
        for (size_t vertexIndex = range.begin(); vertexIndex != range.end(); ++vertexIndex)
        {
            for (int triIndex : myAdjacentTriangles[vertexIndex])
            {
                vertexNormals[vertexIndex] += weightedTriNormals[triIndex];
            }

            vertexNormals[vertexIndex].normalize();
        }
    });

    if (doRenderTriFaces) renderer.addTriFaces(myVertices, vertexNormals, myTriangles, triFaceColour);

    // Render triangle normals
    if (doRenderTriNormals)
    {
        VecVec3d startPoints(myTriangles.size());
        VecVec3d endPoints(myTriangles.size());

        tbb::parallel_for(tbb::blocked_range<size_t>(0, myTriangles.size(), tbbLightGrainSize), [&](const tbb::blocked_range<size_t>& range)
        {
            for (size_t triIndex = range.begin(); triIndex != range.end(); ++triIndex)
            {
                const Vec3i& tri = myTriangles[triIndex];

                Vec3d localNormalStart = Vec3d::Zero();

                for (int localVertexIndex : {0, 1, 2})
                    localNormalStart += myVertices[tri[localVertexIndex]];

                localNormalStart *= 1. / 3.;

                startPoints[triIndex] = localNormalStart;

                // Get triangle normal end point
                endPoints[triIndex] = startPoints[triIndex] + .1 * weightedTriNormals[triIndex].normalized();
            }
        });

        renderer.addLines(startPoints, endPoints, normalColour);
    }

    // Render vertices
    if (doRenderVertices) renderer.addPoints(myVertices, vertexColour, 2.);

    if (doRenderTriEdges)
    {
        VecVec3d startPoints(3 * myTriangles.size());
        VecVec3d endPoints(3 * myTriangles.size());

        tbb::parallel_for(tbb::blocked_range<size_t>(0, myTriangles.size(), tbbLightGrainSize), [&](const tbb::blocked_range<size_t>& range)
        {
            for (size_t triIndex = range.begin(); triIndex != range.end(); ++triIndex)
            {
                const Vec3i& tri = myTriangles[triIndex];
                for (int localStartVertexIndex : {0, 1, 2})
                {
                    int localEndVertexIndex = (localStartVertexIndex + 1) % 3;
                    startPoints[3 * triIndex + localStartVertexIndex] = myVertices[tri[localStartVertexIndex]];
                    endPoints[3 * triIndex + localStartVertexIndex] = myVertices[tri[localEndVertexIndex]];
                }
            }
        });

        renderer.addLines(startPoints, endPoints, edgeColour);
    }

    if (doRenderVertexNormals)
    {
        VecVec3d vertexNormals = computeVertexNormals();

        VecVec3d startPoints = myVertices;
        VecVec3d endPoints = myVertices;

        tbb::parallel_for(tbb::blocked_range<int>(0, int(myVertices.size())), [&](const tbb::blocked_range<int>& range)
        {
            for (int vertexIndex = range.begin(); vertexIndex != range.end(); ++vertexIndex)
            {
                endPoints[vertexIndex] += .1 * vertexNormals[vertexIndex];
            }
        });

        renderer.addLines(startPoints, endPoints, vertexNormalColours);
    }
}

void TriMesh::writeAsOBJ(const std::string &filename) const
{
    std::ofstream objFile;
    objFile.open(filename);

    for (const Vec3d& vertex : myVertices)
    {
	    objFile << "v " << vertex[0] << " " << vertex[1] << " " << vertex[2] << "\n";
    }

    for (const Vec3i& tri : myTriangles)
    {
        objFile << "f " << tri[0] + 1 << " " << tri[1] + 1 << " " << tri[2] + 1 << "\n";
    }
    
    objFile << "\n";
    objFile.close();
}

bool TriMesh::unitTestMesh() const
{
    // Verify vertex has two or more adjacent triangles
    for (int vertexIndex = 0; vertexIndex != myVertices.size(); ++vertexIndex)
    {
        if (myAdjacentTriangles[vertexIndex].size() < 2)
        {
			return false;
        }
    }

    // Verify triangle has three valid vertices
    for (const Vec3i& tri : myTriangles)
    {
        for (int localVertexIndex : {0, 1, 2})
        {
            if (tri[localVertexIndex] < 0 || tri[localVertexIndex] >= myVertices.size())
            {
				return false;
            }
        }
    }

    // Verify vertex's adjacent edge reciprocates
    for (int vertexIndex = 0; vertexIndex != myVertices.size(); ++vertexIndex)
    {
        for (int triIndex : myAdjacentTriangles[vertexIndex])
        {
            const Vec3i& tri = myTriangles[triIndex];
            if (tri[0] != vertexIndex && tri[1] != vertexIndex && tri[2] != vertexIndex)
            {
                return false;
            }
        }
    }

    // Verify triangle's vertices reciprocates
    for (int triIndex = 0; triIndex != myTriangles.size(); ++triIndex)
    {
        for (int localVertexIndex : {0, 1, 2})
        {
            int vertexIndex = myTriangles[triIndex][localVertexIndex];
            if (std::find(myAdjacentTriangles[vertexIndex].begin(), myAdjacentTriangles[vertexIndex].end(), triIndex) == myAdjacentTriangles[vertexIndex].end())
            {
                return false;
            }
        }
    }

    return true;
}

VecVec3d TriMesh::computeVertexNormals() const
{
    std::vector<std::pair<int, Vec3d>> localVertexNormals(3 * myTriangles.size());

    tbb::parallel_for(tbb::blocked_range<int>(0, int(myTriangles.size())), [&](const tbb::blocked_range<int>& range)
    {
        for (int triIndex = range.begin(); triIndex != range.end(); ++triIndex)
        {
            const Vec3i& tri = myTriangles[triIndex];

            for (int localVertexIndex : {0, 1, 2})
            {
                Vec3d edge0 = myVertices[tri[(localVertexIndex + 1) % 3]] - myVertices[tri[localVertexIndex]];
                Vec3d edge1 = myVertices[tri[(localVertexIndex + 2) % 3]] - myVertices[tri[localVertexIndex]];

                double squareArea = edge0.squaredNorm() * edge1.squaredNorm();

                Vec3d localNormal;
                if (squareArea > 0)
                {
                    localNormal = edge0.cross(edge1) / squareArea;
                }
                else
                {
                    localNormal = Vec3d::Zero();
                }

                localVertexNormals[3 * triIndex + localVertexIndex] = std::make_pair(tri[localVertexIndex], localNormal);
            }
        }
    });

    //tbb::parallel_for(tbb::blocked_range<int>(0, int(myTriangles.size())), [&](const tbb::blocked_range<int>& range)
    //{
    //    for (int triIndex = range.begin(); triIndex != range.end(); ++triIndex)
    //    {
    //        const Vec3i& tri = myTriangles[triIndex];

    //        Vec3d triNormal = normal(triIndex);

    //        for (int localVertexIndex : {0, 1, 2})
    //        {
    //            double a = (myVertices[tri[(localVertexIndex + 1) % 3]] - myVertices[tri[localVertexIndex]]).norm();
    //            double b = (myVertices[tri[(localVertexIndex + 2) % 3]] - myVertices[tri[localVertexIndex]]).norm();
    //            double c = (myVertices[tri[(localVertexIndex + 2) % 3]] - myVertices[tri[(localVertexIndex + 1) % 3]]).norm();

    //            double angle = std::acos((a * a + b * b - c * c) / (2. * a * b));
    //            localVertexNormals[3 * triIndex + localVertexIndex] = std::make_pair(tri[localVertexIndex], angle * triNormal);
    //        }
    //    }
    //});

    tbb::parallel_sort(localVertexNormals.begin(), localVertexNormals.end(), [&](const std::pair<int, Vec3d>& pair0, const std::pair<int, Vec3d>& pair1)
    {
        return pair0.first < pair1.first;
    });

    VecVec3d vertexNormals(myVertices.size(), Vec3d::Zero());

    //tbb::parallel_for(tbb::blocked_range<int>(0, int(localVertexNormals.size())), [&](const tbb::blocked_range<int>& range)
    //{
    //    int startIndex = range.begin();

    //    if (startIndex > 0 && localVertexNormals[startIndex - 1].first == localVertexNormals[startIndex].first)
    //    {
    //        while (startIndex < range.end() && localVertexNormals[startIndex - 1].first == localVertexNormals[startIndex].first)
    //        {
    //            ++startIndex;
    //        }
    //    }

    //    int normIndex = startIndex;
    //    while (normIndex < range.end())
    //    {
    //        int oldVertexIndex = localVertexNormals[normIndex].first;
    //        int triCount = 0;
    //        while (normIndex != range.end() && oldVertexIndex == localVertexNormals[normIndex].first)
    //        {
    //            vertexNormals[oldVertexIndex] += localVertexNormals[normIndex].second;
    //            ++normIndex;
    //            triCount++;
    //        }


    //        vertexNormals[oldVertexIndex] /= double(triCount);
    //        //vertexNormals[oldVertexIndex].normalize();
    //    }
    //});


    for (int index = 0; index < localVertexNormals.size(); ++index)
    {
        vertexNormals[localVertexNormals[index].first] += localVertexNormals[index].second;
    }

    for (Vec3d& vertNorm : vertexNormals)
    {
        vertNorm.normalize();
    }

    return vertexNormals;
}

//
// Private methods
//

void TriMesh::initialize(const VecVec3i& triangles, const VecVec3d& vertices)
{
    myTriangles.clear();
    myVertices.clear();

    myTriangles.insert(myTriangles.end(), triangles.begin(), triangles.end());
	myVertices.insert(myVertices.end(), vertices.begin(), vertices.end());

    buildAdjacentTriangles();
}

void TriMesh::buildAdjacentTriangles()
{
    myAdjacentTriangles.clear();
    myAdjacentTriangles.resize(myVertices.size());

    for (size_t triIndex = 0; triIndex != myTriangles.size(); ++triIndex)
    {
        const Vec3i& tri = myTriangles[triIndex];

        for (int localVertIndex : {0, 1, 2})
            myAdjacentTriangles[tri[localVertIndex]].push_back(int(triIndex));
    }
}

}