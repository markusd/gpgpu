/**
 * @author Markus Doellinger
 * @date Jun 5, 2011
 * @file opengl/oglutil.hpp
 */

#ifndef OGL_UTIL_HPP_
#define OGL_UTIL_HPP_

#include <m3d/m3d.hpp>
#include <opengl/camera.hpp>
#include <opengl/framebuffer.hpp>
#include <opengl/texture.hpp>

namespace ogl {

using namespace m3d;

/**
 * Creates a frame buffer with disabled color buffer and a 2D texture of the given
 * size as the depth attachment. Deleting the frame buffer and the texture is the
 * responsibility of the caller.

 * @param width  The width of the depth texture
 * @param height The height of the depth texture
 * @return A pair of frame buffer and the attached depth texture
 */
std::pair<FrameBuffer, Texture> createShadowFBO(unsigned width, unsigned height);

/**
 * Draws the given axis-aligned bounding box
 *
 * @param min The minimum
 * @param max The maximum
 */
void drawAABB(const Vec3f& min, const Vec3f& max);

/**
 * Returns the world coordinates of the given position on the
 * viewport.
 *
 * @param screen The screen coordinates
 * @return       The world coordinates
 */
Vec3f screenToWorld(const Vec3d& screen, const Camera& cam);

/**
 * Returns the world coordinates of the given position on the
 * viewport. The z coordinate will be retrieved by glReadPixels.
 *
 * @param screen The screen coordinates
 * @return       The world coordinates
 */
Vec3f screenToWorld(const Vec2d& screen, const Camera& cam);

/**
 * Shoots a ray at from the screen position to the near and far plane.
 * Returns the normalized direction vector of the two intersection points.
 * The intersection points are stored in near and far.
 *
 * @param screen The screen coordinates
 * @param near   The intersection on the near plane
 * @param far    The intersection on the far plane
 * @return       The direction vector between near and far
 */
Vec3f getScreenRay(const Vec2d& screen, Vec3f& near, Vec3f& far, const Camera& cam);

}

#endif /* OGL_UTIL_HPP_ */
