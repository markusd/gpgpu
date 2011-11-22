/**
 * @author Markus Doellinger, Robert Waury
 * @date May 28, 2011
 * @file opengl/vertexbuffer.hpp
 */

#ifndef VERTEXBUFFER_HPP_
#define VERTEXBUFFER_HPP_

#include <vector>
#include <list>
#include <string>
#include <GL/glew.h>
#ifdef _WIN32
#include <pstdint.h>
#else
#include <stdint.h>
#endif
#include <iostream>

namespace ogl {

typedef std::vector<float> Floats;
typedef std::vector<uint32_t> UInts;

/**
 * A logical sub-buffer within a vertex buffer. It is used
 * to control different positions and materials, i.e. different
 * objects within a single VBO.
 */
struct SubBuffer {
	// material of the sub mesh
	std::string material;

	// the object that generated this sub mesh
	void* userData;

	// the offset and size in the global index buffer
	uint32_t indexOffset;
	uint32_t indexCount;

	// the offset and size in the global vertex buffer
	uint32_t dataOffset;
	uint32_t dataCount;

	SubBuffer() {
		material = "";
		indexOffset = indexCount = 0;
		dataOffset = dataCount = 0;
		userData = NULL;
	}

	static bool compare(const SubBuffer* const first, const SubBuffer* const second) {
		return first->material < second->material;
	}
};

/** A linked list of SubBuffers */
typedef std::list<SubBuffer*> SubBuffers;

/**
 * A vertex buffer with vertex and index data
 */
class VertexBuffer {
public:
	// the format of the buffer, using the GL_ constants
	GLuint m_format;

	// index and vertex buffers
	GLuint m_ibo, m_vbo;
	// actual size and used size of the buffers
	uint32_t m_vboSize, m_vboUsedSize;
	uint32_t m_iboSize, m_iboUsedSize;

	// the index data
	UInts m_indices;

	// the vertices, uvs and normals
	Floats m_data;

	SubBuffers m_buffers;

	VertexBuffer();
	~VertexBuffer();

	/** @return The size of a single vertex in floats */
	unsigned floatSize();

	/** @return The size of a single vertex in bytes */
	unsigned byteSize();

	/**
	 * Binds the vertex buffer object. If setup is specified, also enables
	 * the client states and sets the element pointers.
	 *
	 * @param setup If true, a setup is done
	 */
	void bind(bool setup = true);

	static void unbind();

	/**
	 * Uploads and creates the buffers.
	 */
	void upload();

	/**
	 * Clears the internal data and destroys the buffers.
	 */
	void flush();
};

inline
unsigned VertexBuffer::floatSize()
{
	switch (m_format) {
	case GL_T2F_N3F_V3F:
		return (2 + 3 + 3);
	}
	return 0;
}

inline
unsigned VertexBuffer::byteSize()
{
	return floatSize() * 4;
}

inline
void VertexBuffer::unbind()
{
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

}

#endif /* VERTEXBUFFER_HPP_ */
