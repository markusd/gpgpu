/*
 * framebuffer.hpp
 *
 *  Created on: Jul 3, 2011
 *      Author: Markus Doellinger
 */

#ifndef FRAMEBUFFER_HPP_
#define FRAMEBUFFER_HPP_

#include <boost/tr1/memory.hpp>
#include "GL/glew.h"

namespace ogl {

class __FrameBuffer;
typedef std::tr1::shared_ptr<__FrameBuffer> FrameBuffer;

/**
 * Wrapper for an OpenGL frame buffer.
 */
class __FrameBuffer {
protected:
	GLuint m_id;
	GLenum m_status;

	__FrameBuffer();
	__FrameBuffer(const __FrameBuffer& other);
public:
	virtual ~__FrameBuffer();

	void attachTexture2D(GLenum attachment, GLenum textarget, GLuint texture, GLint level);
	void attachTexture2D(GLenum attachment, GLuint texture);

	void disableColorBuffer();

	GLenum status();
	bool check();

	void bind();
	static void unbind();

	/**
	 * Creates a new empty frame buffer.
	 *
	 * @return The create frame buffer.
	 */
	static FrameBuffer create();
};



inline
void __FrameBuffer::attachTexture2D(GLenum attachment, GLuint texture)
{
	attachTexture2D(attachment, GL_TEXTURE_2D, texture, 0);
}

inline
void __FrameBuffer::disableColorBuffer()
{
	glReadBuffer(GL_NONE);
	glDrawBuffer(GL_NONE);
}

inline
GLenum __FrameBuffer::status()
{
	return m_status;
}

inline
bool __FrameBuffer::check()
{
	return m_status == GL_FRAMEBUFFER_COMPLETE_EXT;
}

inline
void __FrameBuffer::bind()
{
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_id);
}

inline
void __FrameBuffer::unbind()
{
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}

}

#endif /* FRAMEBUFFER_HPP_ */
