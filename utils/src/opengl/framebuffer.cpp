/*
 * framebuffer.cpp
 *
 *  Created on: Jul 3, 2011
 *      Author: Markus Doellinger
 */

#include <opengl/framebuffer.hpp>

namespace ogl {


FrameBuffer __FrameBuffer::create()
{
	FrameBuffer result(new __FrameBuffer());
	glGenFramebuffersEXT(1, &result->m_id);
	result->bind();
	result->m_status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
	return result;
}

__FrameBuffer::__FrameBuffer()
{
}

__FrameBuffer::__FrameBuffer(const __FrameBuffer& other)
{
}

__FrameBuffer::~__FrameBuffer()
{
	glDeleteFramebuffersEXT(1, &m_id);
}

void __FrameBuffer::attachTexture2D(GLenum attachment, GLenum textarget, GLuint texture, GLint level)
{
	bind();
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, attachment, textarget, texture, level);
	m_status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
}

}
