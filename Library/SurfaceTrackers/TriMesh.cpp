#include "TriMesh.h"

#include <iostream>

namespace FluidSim3D::SurfaceTrackers
{
void TriMesh::initialize(const std::vector<Vec3i>& triFaces, const std::vector<Vec3f>& vertices)
{
    std::vector<std::pair<int, int>> vertexFacePairs(3 * triFaces.size());

    myTriFaces.resize(triFaces.size());
    tbb::parallel_for(tbb::blocked_range<int>(0, triFaces.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range) {
        for (int triFaceIndex = range.begin(); triFaceIndex != range.end(); ++triFaceIndex)
        {
            myTriFaces[triFaceIndex] = TriFace(triFaces[triFaceIndex]);

            for (int localVertexIndex : {0, 1, 2})
                vertexFacePairs[3 * triFaceIndex + localVertexIndex] = std::pair<int, int>(triFaces[triFaceIndex][localVertexIndex], triFaceIndex);
        }
    });

    myVertices.resize(vertices.size());

    tbb::parallel_for(tbb::blocked_range<int>(0, vertices.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range) {
        for (int vertexIndex = range.begin(); vertexIndex != range.end(); ++vertexIndex) myVertices[vertexIndex] = Vertex(vertices[vertexIndex]);
    });

    tbb::parallel_sort(vertexFacePairs.begin(), vertexFacePairs.end(),
                        [&](const std::pair<int, int>& pair0, const std::pair<int, int>& pair1) { return pair0.first < pair1.first; });

    tbb::parallel_for(tbb::blocked_range<int>(0, vertexFacePairs.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range) {
        // Advance to next new vertex
        int vertexPairIndex = range.begin();
        if (vertexPairIndex > 0)
        {
            while (vertexPairIndex < vertexFacePairs.size() && vertexFacePairs[vertexPairIndex].first == vertexFacePairs[vertexPairIndex - 1].first)
                ++vertexPairIndex;
        }

        while (vertexPairIndex < range.end())
        {
            int vertexIndex = vertexFacePairs[vertexPairIndex].first;
            while (vertexPairIndex < vertexFacePairs.size() && vertexIndex == vertexFacePairs[vertexPairIndex].first)
            {
                int triIndex = vertexFacePairs[vertexPairIndex].second;
                myVertices[vertexIndex].addTriFace(triIndex);

                ++vertexPairIndex;
            }
        }
    });
}

void TriMesh::insertMesh(const TriMesh& mesh)
{
    int triFaceCount = myTriFaces.size();
    int vertexCount = myVertices.size();

    myVertices.insert(myVertices.end(), mesh.myVertices.begin(), mesh.myVertices.end());
    myTriFaces.insert(myTriFaces.end(), mesh.myTriFaces.begin(), mesh.myTriFaces.end());

    // Update vertices to new tris
    tbb::parallel_for(tbb::blocked_range<int>(vertexCount, myVertices.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range) {
        for (int vertexIndex = range.begin(); vertexIndex != range.end(); ++vertexIndex)
            for (int neighbourTriFaceIndex = 0; neighbourTriFaceIndex < myVertices[vertexIndex].valence(); ++neighbourTriFaceIndex)
            {
                int triFaceIndex = myVertices[vertexIndex].triFace(neighbourTriFaceIndex);
                assert(triFaceIndex >= 0 && triFaceIndex < mesh.triFaceCount());

                myVertices[vertexIndex].replaceTriFace(triFaceIndex, triFaceIndex + triFaceCount);
            }
    });

    tbb::parallel_for(tbb::blocked_range<int>(triFaceCount, myTriFaces.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range) {
        for (int triFaceIndex = range.begin(); triFaceIndex != range.end(); ++triFaceIndex)
            for (int localVertexIndex : {0, 1, 2})
            {
                int meshVertexIndex = myTriFaces[triFaceIndex].vertex(localVertexIndex);
                assert(meshVertexIndex >= 0 && meshVertexIndex < mesh.vertexCount());
                myTriFaces[triFaceIndex].replaceVertex(meshVertexIndex, meshVertexIndex + vertexCount);
            }
    });
}

std::vector<Vec3f> TriMesh::vertexNormals() const
{
    std::vector<Vec3f> triFaceWeightedNormals(myTriFaces.size());

    tbb::parallel_for(tbb::blocked_range<int>(0, myTriFaces.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range) {
        for (int triFaceIndex = range.begin(); triFaceIndex != range.end(); ++triFaceIndex) triFaceWeightedNormals[triFaceIndex] = scaledNormal(triFaceIndex);
    });

    std::vector<Vec3f> vertexNormals(myVertices.size());

    tbb::parallel_for(tbb::blocked_range<int>(0, myVertices.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range) {
        for (int vertexIndex = range.begin(); vertexIndex != range.end(); ++vertexIndex)
        {
            Vec3f localVertexNormal(0);

            for (int neighbourTriFaceIndex = 0; neighbourTriFaceIndex < myVertices[vertexIndex].valence(); ++neighbourTriFaceIndex)
            {
                int triFaceIndex = myVertices[vertexIndex].triFace(neighbourTriFaceIndex);
                assert(triFaceIndex >= 0 && triFaceIndex < myTriFaces.size());

                localVertexNormal += triFaceWeightedNormals[triFaceIndex];
            }

            vertexNormals[vertexIndex] = normalize(localVertexNormal);
        }
    });

    return vertexNormals;
}

void TriMesh::drawMesh(Renderer& renderer, bool doRenderTriFaces, Vec3f triFaceColour, bool doRenderTriNormals, Vec3f normalColour, bool doRenderVertices, Vec3f vertexColour,
                       bool doRenderTriEdges, Vec3f edgeColour)
{
    // Pre-compute area-weighted triangle normals
    // and set triangle faces
    std::vector<Vec3f> weightedTriNormals(myTriFaces.size(), Vec3f(0));

    // Render triangles
    std::vector<Vec3f> vertexPoints(myVertices.size());
    std::vector<Vec3f> vertexNormals(myVertices.size(), Vec3f(0));
    std::vector<Vec3i> triFaces(myTriFaces.size());

    tbb::parallel_for(tbb::blocked_range<int>(0, myTriFaces.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range) {
        for (int triFaceIndex = range.begin(); triFaceIndex != range.end(); ++triFaceIndex)
        {
            weightedTriNormals[triFaceIndex] = scaledNormal(triFaceIndex);

            triFaces[triFaceIndex] = myTriFaces[triFaceIndex].vertices();
        }
    });

    // Accumulate area-weight triangle normals to each vertex and normalize.
    // Set vertex points.
    tbb::parallel_for(tbb::blocked_range<int>(0, myVertices.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range) {
        for (int vertexIndex = range.begin(); vertexIndex != range.end(); ++vertexIndex)
        {
            for (int neighbourTriFaceIndex = 0; neighbourTriFaceIndex < myVertices[vertexIndex].valence(); ++neighbourTriFaceIndex)
            {
                int triFaceIndex = myVertices[vertexIndex].triFace(neighbourTriFaceIndex);
                vertexNormals[vertexIndex] += weightedTriNormals[triFaceIndex];
            }

            normalizeInPlace(vertexNormals[vertexIndex]);

            vertexPoints[vertexIndex] = myVertices[vertexIndex].point();
        }
    });

    if (doRenderTriFaces) renderer.addTriFaces(vertexPoints, vertexNormals, triFaces, triFaceColour);

    // Render triangle normals
    if (doRenderTriNormals)
    {
        std::vector<Vec3f> startPoints(myTriFaces.size());
        std::vector<Vec3f> endPoints(myTriFaces.size());

        tbb::parallel_for(tbb::blocked_range<int>(0, myTriFaces.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range) {
            for (int triFaceIndex = range.begin(); triFaceIndex != range.end(); ++triFaceIndex)
            {
                Vec3f localNormalStart(0);
                for (int localVertexIndex : {0, 1, 2}) localNormalStart += myVertices[myTriFaces[triFaceIndex].vertex(localVertexIndex)].point();

                localNormalStart *= 1. / 3.;

                startPoints[triFaceIndex] = localNormalStart;

                // Get triangle normal end point
                endPoints[triFaceIndex] = startPoints[triFaceIndex] + .1 * normalize(weightedTriNormals[triFaceIndex]);
            }
        });

        renderer.addLines(startPoints, endPoints, normalColour);
    }

    // Render vertices
    if (doRenderVertices) renderer.addPoints(vertexPoints, vertexColour, 2.);

    if (doRenderTriEdges)
    {
        std::vector<Vec3f> startPoints(3 * myTriFaces.size());
        std::vector<Vec3f> endPoints(3 * myTriFaces.size());

        tbb::parallel_for(tbb::blocked_range<int>(0, myTriFaces.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range) {
            for (int triFaceIndex = range.begin(); triFaceIndex != range.end(); ++triFaceIndex)
                for (int localStartVertexIndex : {0, 1, 2})
                {
                    int localEndVertexIndex = (localStartVertexIndex + 1) % 3;
                    startPoints[3 * triFaceIndex + localStartVertexIndex] = myVertices[myTriFaces[triFaceIndex].vertex(localStartVertexIndex)].point();
                    endPoints[3 * triFaceIndex + localStartVertexIndex] = myVertices[myTriFaces[triFaceIndex].vertex(localEndVertexIndex)].point();
                }
        });

        renderer.addLines(startPoints, endPoints, edgeColour);
    }
}

bool TriMesh::unitTestMesh() const
{
    tbb::enumerable_thread_specific<std::vector<Vec2i>> parallelFailedMeshPairs;

    tbb::parallel_for(tbb::blocked_range<int>(0, myVertices.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range) {
        auto& localFailedMeshPairs = parallelFailedMeshPairs.local();

        for (int vertexIndex = range.begin(); vertexIndex != range.end(); ++vertexIndex)
            for (int adjacentTriFaceIndex = 0; adjacentTriFaceIndex < myVertices[vertexIndex].valence(); ++adjacentTriFaceIndex)
            {
                int triFaceIndex = myVertices[vertexIndex].triFace(adjacentTriFaceIndex);

                if (!myTriFaces[triFaceIndex].findVertex(vertexIndex)) localFailedMeshPairs.emplace_back(vertexIndex, triFaceIndex);
            }
    });

    std::vector<Vec2i> failedMeshPairs;
    mergeLocalThreadVectors(failedMeshPairs, parallelFailedMeshPairs);

    if (failedMeshPairs.size() > 0)
    {
        for (const auto& failedPair : failedMeshPairs)
            std::cout << "Unit test failed in adjacent tri-face test. Vertex: " << failedPair[0] << ". Tri: " << failedPair[1] << std::endl;

        return false;
    }

    parallelFailedMeshPairs.clear();
    failedMeshPairs.clear();

    tbb::parallel_for(tbb::blocked_range<int>(0, myTriFaces.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range) {
        auto& localFailedMeshPairs = parallelFailedMeshPairs.local();

        for (int triFaceIndex = range.begin(); triFaceIndex != range.end(); ++triFaceIndex)
            for (int localVertexIndex : {0, 1, 2})
            {
                int meshVertexIndex = myTriFaces[triFaceIndex].vertex(localVertexIndex);

                if (!myVertices[meshVertexIndex].findTriFace(triFaceIndex)) localFailedMeshPairs.emplace_back(meshVertexIndex, triFaceIndex);
            }
    });

    mergeLocalThreadVectors(failedMeshPairs, parallelFailedMeshPairs);

    if (failedMeshPairs.size() > 0)
    {
        for (const auto& failedPair : failedMeshPairs)
            std::cout << "Unit test failed in adjacent vertex test. Vertex: " << failedPair[0] << ". Tri: " << failedPair[1] << std::endl;

        return false;
    }

    tbb::enumerable_thread_specific<std::vector<int>> parallelNanVertices;

    tbb::parallel_for(tbb::blocked_range<int>(0, myVertices.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range) {
        auto& localNanVertices = parallelNanVertices.local();

        for (int vertexIndex = range.begin(); vertexIndex != range.end(); ++vertexIndex)
        {
            bool isVertexNan = false;
            for (int axis : {0, 1, 2})
            {
                if (!std::isfinite(myVertices[vertexIndex].point()[axis])) isVertexNan = true;
            }

            if (isVertexNan) localNanVertices.push_back(vertexIndex);
        }
    });

    std::vector<int> nanVertices;
    mergeLocalThreadVectors(nanVertices, parallelNanVertices);

    if (nanVertices.size() > 0)
    {
        for (const auto& nanVertex : nanVertices) std::cout << "Unit test failed in NaN check. Vertex: " << nanVertex << std::endl;

        return false;
    }

    return true;
}

}  // namespace FluidSim3D::SurfaceTrackers