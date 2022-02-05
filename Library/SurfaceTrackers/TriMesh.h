#ifndef FLUIDSIM3D_TRI_MESH_H
#define FLUIDSIM3D_TRI_MESH_H

#include <algorithm>
#include <vector>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

#include "Integrator.h"
#include "Renderer.h"
#include "Utilities.h"

namespace FluidSim3D
{

class TriMesh
{
public:
    // Vanilla constructor leaves initialization up to the caller
    TriMesh() {}

    // Initialize mesh container with tris and the associated vertices
    TriMesh(const VecVec3i& triangles, const VecVec3d& vertices);

    void reinitialize(const VecVec3i& triangles, const VecVec3d& vertices);

    // Add more mesh pieces to an already existing mesh (although the existing mesh could empty).
    // The incoming mesh triangles point to vertices (and vice versa) from 0 to ne-1 locally. They need
    // to be offset by the triangle/vertex size in the existing mesh.
    void insertMesh(const TriMesh& mesh);

    const VecVec3i& triangles() const;

    const VecVec3d& vertices() const;

    const std::vector<std::vector<int>>& adjacentTriangles() const;

	FORCE_INLINE const Vec3i& triangle(int index) const
	{
		return myTriangles[index];
	}

	FORCE_INLINE const Vec3d& vertex(int index) const
	{
		return myVertices[index];
	}

	FORCE_INLINE void setVertex(int index, const Vec3d& vertex)
	{
		myVertices[index] = vertex;
	}

    void clear();

    FORCE_INLINE int triangleCount() const
    {
        return int(myTriangles.size());
    }

    FORCE_INLINE int vertexCount() const
    {
        return int(myVertices.size());
    }

    FORCE_INLINE Vec3d scaledNormal(int triIndex) const
    {
        const Vec3i& tri = myTriangles[triIndex];

        Vec3d tangent0 = myVertices[tri[1]] - myVertices[tri[0]];
        Vec3d tangent1 = myVertices[tri[2]] - myVertices[tri[0]];

        return tangent0.cross(tangent1);
    }

    FORCE_INLINE Vec3d normal(int triIndex) const
    {
		Vec3d normal = scaledNormal(triIndex);
		double norm = normal.norm();

		if (norm > 0)
		{
			return normal / norm;
		}

		return Vec3d::Zero();
    }

	AlignedBox3d boundingBox() const;

    // Reverse winding order
    void reverse();

    void scale(double s);

    void translate(const Vec3d& t);

    bool isTriangleDegenerate(int triFaceIndex) const;

    void drawMesh(Renderer& renderer, bool doRenderTriFaces = false, Vec3d triFaceColour = Vec3d::Zero(), bool doRenderTriNormals = false, Vec3d normalColour = Vec3d::Zero(),
                  bool doRenderVertices = false, Vec3d vertexColour = Vec3d::Zero(), bool doRenderTriEdges = false, Vec3d edgeColour = Vec3d::Zero());

    template <typename VelocityField>
    void advectMesh(double dt, const VelocityField& vel, IntegrationOrder order);

    bool unitTestMesh() const;

private:
    void initialize(const VecVec3i& faces, const VecVec3d& vertices);

    void buildAdjacentTriangles();

    VecVec3i myTriangles;
    VecVec3d myVertices;

    std::vector<std::vector<int>> myAdjacentTriangles;
};

template <typename VelocityField>
void TriMesh::advectMesh(double dt, const VelocityField& velocity, IntegrationOrder order)
{
    tbb::parallel_for(tbb::blocked_range<size_t>(0, myVertices.size(), tbbLightGrainSize), [&](const tbb::blocked_range<size_t>& range)
    {
        for (size_t vertexIndex = range.begin(); vertexIndex != range.end(); ++vertexIndex)
            myVertices[vertexIndex] = Integrator(dt, myVertices[vertexIndex], velocity, order);
    });
}

}

#endif