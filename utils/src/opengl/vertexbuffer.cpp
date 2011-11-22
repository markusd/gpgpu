/**
 * @author Markus Doellinger, Robert Waury
 * @date May 28, 2011
 * @file opengl/vertexbuffer.cpp
 */

#include <opengl/vertexbuffer.hpp>

namespace ogl {

VertexBuffer::VertexBuffer()
	: m_format(GL_T2F_N3F_V3F),
	  m_ibo(0), m_vbo(0),
	  m_vboSize(0), m_vboUsedSize(0),
	  m_iboSize(0), m_iboUsedSize(0)
{
}

VertexBuffer::~VertexBuffer()
{
	flush();
}

void VertexBuffer::bind(bool setup)
{
	if (m_vbo) {
		if (setup) {
			glEnableClientState(GL_VERTEX_ARRAY);
			glEnableClientState(GL_TEXTURE_COORD_ARRAY);
			glEnableClientState(GL_NORMAL_ARRAY);
			glDisableClientState(GL_COLOR_ARRAY);
		}
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
		if (setup) {
			glVertexPointer(3, GL_FLOAT, byteSize(), (void*)((2+3)*4));
			glTexCoordPointer(3, GL_FLOAT, byteSize(), (void*)0);
			glNormalPointer(GL_FLOAT, byteSize(), (void*)((2)*4));
		}
	}

	if (m_ibo) {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ibo);
		if (setup) {
			glIndexPointer(GL_UNSIGNED_INT, 0, 0);
		}
	}

}

void VertexBuffer::upload()
{
	// calculate the actual size of the required buffer
	unsigned sizeInBytes = m_data.size() * sizeof(float);

	// create the vertex buffer object, if necessary
	if (sizeInBytes > 0 && m_vbo == 0) {
		glGenBuffers(1, &m_vbo);
		m_vboSize = m_vboUsedSize = 0;
	}

	// delete the buffer, if there is no data
	if (sizeInBytes == 0) {
		glDeleteBuffers(1, &m_vbo);
		m_vbo = 0;
		m_vboSize = m_vboUsedSize = 0;

	// resize the buffer, if the reserved size is too small, or twice as
	// large as required
	} else if (m_vboSize < sizeInBytes || m_vboSize > 2 * sizeInBytes) {
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeInBytes, &m_data[0], GL_DYNAMIC_DRAW);
		m_vboSize = m_vboUsedSize = sizeInBytes;

	// re-use the old buffer because there is low or no overhead
	} else {
		 // use old buffer
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeInBytes, &m_data[0]);
		m_vboUsedSize = sizeInBytes;
	}

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// check if index buffer is required
	if (m_indices.size() > 0) {

		// create buffer, if necessary
		if (m_ibo == 0) {
			glGenBuffers(1, &m_ibo);
			m_iboSize = m_iboUsedSize = 0;
		}

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ibo);
		sizeInBytes = m_indices.size() * sizeof(GLuint);

		// resize the buffer
		if (m_iboSize < sizeInBytes || m_iboSize > 2 * sizeInBytes) {
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeInBytes, &m_indices[0], GL_DYNAMIC_DRAW);
			m_iboSize = m_iboUsedSize = sizeInBytes;

		// use old buffer
		} else {
			glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, sizeInBytes, &m_indices[0]);
			m_iboUsedSize = sizeInBytes;
		}

	// delete unused buffer
	} else if (m_ibo != 0) {
		glDeleteBuffers(1, &m_ibo);
		m_ibo = 0;
	}
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void VertexBuffer::flush() {
	// clear data in memory
	m_indices.clear();
	m_data.clear();

	for (SubBuffers::iterator i = m_buffers.begin(); i != m_buffers.end(); ++i) {
		delete (*i);
	}
	m_buffers.clear();

	// destroy buffers
	if (m_vbo != 0) {
		glDeleteBuffers(1, &m_vbo);
		m_vbo = m_vboSize = m_vboUsedSize = 0;
	}

	if (m_ibo != 0) {
		glDeleteBuffers(1, &m_ibo);
		m_ibo = m_iboSize = m_iboUsedSize = 0;
	}
}

}
