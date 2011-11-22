/**
 * @author Markus Doellinger
 * @date Jun 5, 2011
 * @file opengl/oglutil.cpp
 */

#include <opengl/oglutil.hpp>
#include <GL/glew.h>
#include <iostream>

namespace ogl {

std::pair<FrameBuffer, Texture> createShadowFBO(unsigned width, unsigned height)
{
	Texture depthTexture = __Texture::create();
	depthTexture->bind();

	depthTexture->setFilter(GL_LINEAR, GL_LINEAR);
	depthTexture->setWrap(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
	depthTexture->setTexParameteri(GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
	depthTexture->setTexParameteri(GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
	depthTexture->setTexParameteri(GL_DEPTH_TEXTURE_MODE, GL_INTENSITY);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
	depthTexture->unbind();

	FrameBuffer frameBuffer = __FrameBuffer::create();
	frameBuffer->disableColorBuffer();
	frameBuffer->attachTexture2D(GL_DEPTH_ATTACHMENT_EXT, depthTexture->m_textureID);

	if (!frameBuffer->check())
		std::cerr << "Could not create FBO" << std::endl;

	__FrameBuffer::unbind();
	return std::make_pair(frameBuffer, depthTexture);
}

void drawAABB(const Vec3f& min, const Vec3f& max)
{
	glBegin(GL_LINES);

	glVertex3f(min.x, min.y, min.z);
	glVertex3f(min.x, min.y, max.z);

	glVertex3f(max.x, min.y, min.z);
	glVertex3f(max.x, min.y, max.z);

	glVertex3f(min.x, max.y, min.z);
	glVertex3f(min.x, max.y, max.z);

	glVertex3f(max.x, max.y, min.z);
	glVertex3f(max.x, max.y, max.z);

	glVertex3f(min.x, min.y, min.z);
	glVertex3f(max.x, min.y, min.z);

	glVertex3f(min.x, min.y, min.z);
	glVertex3f(min.x, max.y, min.z);

	glVertex3f(max.x, min.y, min.z);
	glVertex3f(max.x, max.y, min.z);

	glVertex3f(min.x, max.y, min.z);
	glVertex3f(max.x, max.y, min.z);

	glVertex3f(min.x, min.y, max.z);
	glVertex3f(max.x, min.y, max.z);

	glVertex3f(min.x, min.y, max.z);
	glVertex3f(min.x, max.y, max.z);

	glVertex3f(max.x, min.y, max.z);
	glVertex3f(max.x, max.y, max.z);

	glVertex3f(min.x, max.y, max.z);
	glVertex3f(max.x, max.y, max.z);
	glEnd();
}

Vec3f screenToWorld(const Vec3d& screen, const Camera& cam)
{
	Mat4d mv(cam.m_modelview);
	Mat4d proj(cam.m_projection);

	double x, y, z;
	y = cam.m_viewport.w - screen.y;

	gluUnProject(screen.x, y, screen.z, mv[0], proj[0], &cam.m_viewport[0], &x, &y, &z);
	return Vec3f(x, y, z);
}

Vec3f screenToWorld(const Vec2d& screen, const Camera& cam)
{
	Mat4d mv(cam.m_modelview);
	Mat4d proj(cam.m_projection);

	double x, y, z;
	float screenZ;
	y = cam.m_viewport.w - screen.y;

	glReadPixels(screen.x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &screenZ);

	gluUnProject(screen.x, y, screenZ, mv[0], proj[0], &cam.m_viewport[0], &x, &y, &z);
	return Vec3f(x, y, z);
}

Vec3f getScreenRay(const Vec2d& screen, Vec3f& near, Vec3f& far, const Camera& cam)
{
	Mat4d mv(cam.m_modelview);
	Mat4d proj(cam.m_projection);

	float _y = cam.m_viewport.w - screen.y;

	// shoot a ray to the near and far plane
	double dX, dY, dZ;
	gluUnProject(screen.x, _y, 0.0, mv[0], proj[0], &cam.m_viewport[0], &dX, &dY, &dZ);
	near = Vec3f(dX, dY, dZ);
	gluUnProject(screen.x, _y, 1.0, mv[0], proj[0], &cam.m_viewport[0], &dX, &dY, &dZ);
	far = Vec3f(dX, dY, dZ);

	return (near - far).normalized();
}

}
