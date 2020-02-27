#ifndef LIBRARY_TRI_MESH_H
#define LIBRARY_TRI_MESH_H

#include <algorithm>
#include <vector>

#include "Integrator.h"
#include "Renderer.h"
#include "Utilities.h"
#include "Vec.h"
#include "tbb/tbb.h"

namespace FluidSim3D::SurfaceTrackers
{
using namespace Utilities;
using namespace RenderTools;

class Vertex
{
public:
    Vertex() : myPoint(Vec3f(0)) {}

    Vertex(const Vec3f& point) : myPoint(point) {}

    const Vec3f& point() const { return myPoint; }

    void setPoint(const Vec3f& point) { myPoint = point; }

    void operator=(const Vec3f& point) { myPoint = point; }

    // Get triangle face indicent to vertex
    int triFace(int index) const { return myTriFaces[index]; }

    void addTriFace(int index)
    {
        assert(index >= 0);
        myTriFaces.push_back(index);
    }

    // Search through the triangle list for a matching old
    // triangle index. If found, replace that edge index
    // with new edge index.
    bool replaceTriFace(int oldIndex, int newIndex)
    {
        assert(oldIndex >= 0 && newIndex >= 0);
        auto result = std::find(myTriFaces.begin(), myTriFaces.end(), oldIndex);

        if (result == myTriFaces.end())
            return false;
        else
            *result = newIndex;

        return true;
    }

    // Search through the tri list and return true if
    // there is a tri index that matches search index
    bool findTriFace(int index) const
    {
        auto result = std::find(myTriFaces.begin(), myTriFaces.end(), index);
        return result != myTriFaces.end();
    }

    int valence() const { return myTriFaces.size(); }

    template <typename T>
    void operator*=(const T& s)
    {
        myPoint *= s;
    }

    template <typename T>
    void operator+=(const T& s)
    {
        myPoint += s;
    }

private:
    Vec3f myPoint;
    std::vector<int> myTriFaces;
};

class TriFace
{
public:
    TriFace(const Vec3i& vertices) : myVertices(vertices) {}

    const Vec3i& vertices() const { return myVertices; }

    int vertex(int index) const { return myVertices[index]; }

    // Given a vertex index, return the next
    // vertex in the winding order.
    int adjacentVertex(int index) const
    {
        if (myVertices[0] == index)
            return myVertices[1];
        else if (myVertices[1] == index)
            return myVertices[2];
        else if (myVertices[2] == index)
            return myVertices[0];
        else
            assert(false);

        return -1;
    }

    bool replaceVertex(int oldIndex, int newIndex)
    {
        if (myVertices[0] == oldIndex)
        {
            myVertices[0] = newIndex;
            return true;
        }
        else if (myVertices[1] == oldIndex)
        {
            myVertices[1] = newIndex;
            return true;
        }
        else if (myVertices[2] == oldIndex)
        {
            myVertices[2] = newIndex;
            return true;
        }

        return false;
    }

    bool findVertex(int index) const
    {
        if (myVertices[0] == index || myVertices[1] == index || myVertices[2] == index) return true;

        return false;
    }

    void reverse() { std::swap(myVertices[0], myVertices[2]); }

private:
    // Each triangle can be viewed as having a winding
    // order 0-1-2, using the right-hand rule to
    // determine the normal orientation.

    Vec3i myVertices;
};

class TriMesh
{
public:
    // Vanilla constructor leaves initialization up to the caller
    TriMesh() {}

    // Initialize mesh container with tris and the associated vertices
    TriMesh(const std::vector<Vec3i>& triFaces, const std::vector<Vec3f>& vertices) { initialize(triFaces, vertices); }

    void reinitialize(const std::vector<Vec3i>& triFaces, const std::vector<Vec3f>& vertices)
    {
        myTriFaces.clear();
        myVertices.clear();

        initialize(triFaces, vertices);
    }

    // Add more mesh pieces to an already existing mesh (although the existing mesh could empty).
    // The incoming mesh triangles point to vertices (and vice versa) from 0 to ne-1 locally. They need
    // to be offset by the triangle/vertex size in the existing mesh.
    void insertMesh(const TriMesh& mesh);

    const std::vector<TriFace>& triFaces() const { return myTriFaces; }

    const TriFace& triFace(int index) const { return myTriFaces[index]; }

    const std::vector<Vertex>& vertices() const { return myVertices; }

    const Vertex& vertex(int index) const { return myVertices[index]; }

    void setVertex(int index, const Vec3f& vertex) { myVertices[index].setPoint(vertex); }

    void clear()
    {
        myVertices.clear();
        myTriFaces.clear();
    }

    int triFaceCount() const { return myTriFaces.size(); }

    int vertexCount() const { return myVertices.size(); }

    Vec3f scaledNormal(int triFaceIndex) const
    {
        Vec3i vertices = myTriFaces[triFaceIndex].vertices();

        Vec3f tangent0 = myVertices[vertices[1]].point() - myVertices[vertices[0]].point();
        Vec3f tangent1 = myVertices[vertices[2]].point() - myVertices[vertices[1]].point();

        return cross(tangent0, tangent1);
    }

    Vec3f normal(int triFaceIndex) const { return normalize(scaledNormal(triFaceIndex)); }

    std::vector<Vec3f> vertexNormals() const;

    // Reverse winding order
    void reverse()
    {
        tbb::parallel_for(tbb::blocked_range<int>(0, myTriFaces.size(), tbbLightGrainSize),
                          [&](const tbb::blocked_range<int>& range) {
                              for (int triFaceIndex = range.begin(); triFaceIndex != range.end(); ++triFaceIndex)
                                  myTriFaces[triFaceIndex].reverse();
                          });
    }

    void scale(float s)
    {
        tbb::parallel_for(tbb::blocked_range<int>(0, myVertices.size(), tbbLightGrainSize),
                          [&](const tbb::blocked_range<int>& range) {
                              for (int vertexIndex = range.begin(); vertexIndex != range.end(); ++vertexIndex)
                                  myVertices[vertexIndex] *= s;
                          });
    }

    void translate(const Vec3f& t)
    {
        tbb::parallel_for(tbb::blocked_range<int>(0, myVertices.size(), tbbLightGrainSize),
                          [&](const tbb::blocked_range<int>& range) {
                              for (int vertexIndex = range.begin(); vertexIndex != range.end(); ++vertexIndex)
                                  myVertices[vertexIndex] += t;
                          });
    }

    // Test for a degenerate tri
    bool isTriFaceDegenerate(int triFaceIndex) const
    {
        const TriFace& triFace = myTriFaces[triFaceIndex];

        return (myVertices[triFace.vertex(0)].point() == myVertices[triFace.vertex(1)].point() ||
                myVertices[triFace.vertex(1)].point() == myVertices[triFace.vertex(2)].point() ||
                myVertices[triFace.vertex(0)].point() == myVertices[triFace.vertex(2)].point());
    }

    void drawMesh(Renderer& renderer, bool doRenderTriFaces = false, Vec3f triFaceColour = Vec3f(0),
                  bool doRenderTriNormals = false, Vec3f normalColour = Vec3f(0), bool doRenderVertices = false,
                  Vec3f vertexColour = Vec3f(0), bool doRenderTriEdges = false, Vec3f edgeColour = Vec3f(0));

    template <typename VelocityField>
    void advectMesh(float dt, const VelocityField& vel, IntegrationOrder order);

    bool unitTestMesh() const;

private:
    void initialize(const std::vector<Vec3i>& triFaces, const std::vector<Vec3f>& vertices);

    std::vector<TriFace> myTriFaces;
    std::vector<Vertex> myVertices;
};

template <typename VelocityField>
void TriMesh::advectMesh(float dt, const VelocityField& velocity, IntegrationOrder order)
{
    tbb::parallel_for(
        tbb::blocked_range<int>(0, myVertices.size(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range) {
            for (int vertexIndex = range.begin(); vertexIndex != range.end(); ++vertexIndex)
                myVertices[vertexIndex].setPoint(Integrator(dt, myVertices[vertexIndex].point(), velocity, order));
        });
}

}  // namespace FluidSim3D::SurfaceTrackers

#endif