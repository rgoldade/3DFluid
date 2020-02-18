#ifndef LIBRARY_RENDERER_H
#define LIBRARY_RENDERER_H

#include <functional>

#include <GL/glut.h>

#include "Camera3D.h"
#include "Utilities.h"
#include "Vec.h"

///////////////////////////////////
//
// Renderer.h/cpp
// Ryan Goldade 2017
//
// Render machine to handle adding
// primitives from various different
// sources and having a central place
// to run the render loop.
//
////////////////////////////////////

namespace FluidSim3D::RenderTools
{

using namespace Utilities;

class Renderer
{
public:
	Renderer(const char* title,
				const Vec2i& windowSize,
				const Vec2f& screenOrigin,
				float screenHeight,
				int* argc, char** argv);

	void display();
	void mouse(int button, int state, int x, int y);
	void drag(int x, int y);
	void keyboard(unsigned char key, int x, int y);
	void reshape(int width, int height);

	void setCamera(Camera3D* camera) { myCamera = camera; }
	void setUserMouseClick(const std::function<void(int, int, int, int)>& mouseClickFunction);
	void setUserKeyboard(const std::function<void(unsigned char, int, int)>& keyboardFunction);
	void setUserMouseDrag(const std::function<void(int, int)>& mouseDragFunction);
	void setUserDisplay(const std::function<void()>& displayFunction);

	void addPoint(const Vec3f& point,
					const Vec3f& colour = Vec3f(0),
					float size = 1);

	void addPoints(const std::vector<Vec3f>& points,
					const Vec3f& colour = Vec3f(0),
					float size = 1);

	void addLine(const Vec3f& startingPoint,
					const Vec3f& endingPoint,
					const Vec3f& colour = Vec3f(0),
					float lineWidth = 1);

	void addLines(const std::vector<Vec3f>& startingPoints,
					const std::vector<Vec3f>& endingPoints,
					const Vec3f& colour,
					float lineWidth = 1);

	void addTriFaces(const std::vector<Vec3f>& vertices,
						const std::vector<Vec3f>& normals,
						const std::vector<Vec3i>& faces,
						const Vec3f& faceColour);

	void drawPrimitives() const;

	void printImage(const std::string& filename) const;

	void clear();
	void run();

private:
	std::vector<std::vector<Vec3f>> myPoints;
	std::vector<Vec3f> myPointColours;
	std::vector<float> myPointSizes;

	std::vector<std::vector<Vec3f>> myLineStartingPoints;
	std::vector<std::vector<Vec3f>> myLineEndingPoints;
	std::vector<Vec3f> myLineColours;
	std::vector<float> myLineWidths;

	std::vector<std::vector<Vec3f>> myTriVertices;
	std::vector<std::vector<Vec3f>> myTriNormals;
	std::vector<std::vector<Vec3i>> myTriFaces;
	std::vector<Vec3f> myTriFaceColours;

	// width, height
	Vec2i myWindowSize;

	Vec2f myCurrentScreenOrigin;
	float myCurrentScreenHeight;

	Vec2f myDefaultScreenOrigin;
	float myDefaultScreenHeight;

	// Mouse specific state
	Vec2i myMousePosition;
	bool myMouseHasMoved;

	enum class MouseAction { INACTIVE, PAN, ZOOM_IN, ZOOM_OUT };
	MouseAction myMouseAction;

	// User specific extensions for each glut callback
	std::function<void(unsigned char, int, int)> myUserKeyboardFunction;
	std::function<void(int, int, int, int)> myUserMouseClickFunction;
	std::function<void(int, int)> myUserMouseDragFunction;
	std::function<void()> myUserDisplayFunction;

	// Simple material and lights
	void setGenericLights(void) const;
	void setGenericMaterial(float r, float g, float b, GLenum face) const;

	// 3-D Camera object
	Camera3D* myCamera;
};

}
#endif