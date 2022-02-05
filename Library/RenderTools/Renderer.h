#ifndef FLUIDSIM3D_RENDERER_H
#define FLUIDSIM3D_RENDERER_H

#include <functional>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include "Camera3D.h"
#include "Utilities.h"

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

namespace FluidSim3D
{

class Renderer
{
public:
    Renderer(const char* title, const Vec2i& windowSize, const Vec2d& screenOrigin, double screenHeight, int* argc,
             char** argv);

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

    void addPoint(const Vec3d& point, const Vec3d& colour = Vec3d::Zero(), double size = 1);

    void addPoints(const VecVec3d& points, const Vec3d& colour = Vec3d::Zero(), double size = 1);

    void addLine(const Vec3d& startingPoint, const Vec3d& endingPoint, const Vec3d& colour = Vec3d::Zero(),
                 double lineWidth = 1);

    void addLines(const VecVec3d& startingPoints, const VecVec3d& endingPoints, const Vec3d& colour,
                  double lineWidth = 1);

    void addTriFaces(const VecVec3d& vertices, const VecVec3d& normals,
                     const VecVec3i& faces, const Vec3d& faceColour);

    void drawPrimitives() const;

    void printImage(const std::string& filename) const;

    void clear();
    void run();

private:
    std::vector<VecVec3d> myPoints;
    VecVec3d myPointColours;
    std::vector<double> myPointSizes;

    std::vector<VecVec3d> myLineStartingPoints;
    std::vector<VecVec3d> myLineEndingPoints;
    VecVec3d myLineColours;
    std::vector<double> myLineWidths;

    std::vector<VecVec3d> myTriVertices;
    std::vector<VecVec3d> myTriNormals;
    std::vector<VecVec3i> myTriFaces;
    VecVec3d myTriFaceColours;

    // width, height
    Vec2i myWindowSize;

    Vec2d myCurrentScreenOrigin;
    double myCurrentScreenHeight;

    Vec2d myDefaultScreenOrigin;
    double myDefaultScreenHeight;

    // Mouse specific state
    Vec2i myMousePosition;
    bool myMouseHasMoved;

    enum class MouseAction
    {
        INACTIVE,
        PAN,
        ZOOM_IN,
        ZOOM_OUT
    };
    MouseAction myMouseAction;

    // User specific extensions for each glut callback
    std::function<void(unsigned char, int, int)> myUserKeyboardFunction;
    std::function<void(int, int, int, int)> myUserMouseClickFunction;
    std::function<void(int, int)> myUserMouseDragFunction;
    std::function<void()> myUserDisplayFunction;

    // Simple material and lights
    void setGenericLights(void) const;
    void setGenericMaterial(double r, double g, double b, GLenum face) const;

    // 3-D Camera object
    Camera3D* myCamera;
};

}
#endif