/**
 * @author Markus Doellinger, Robert Waury
 * @date May 21, 2011
 * @file opengl/texture.hpp
 */


#ifndef TEXTURE_HPP_
#define TEXTURE_HPP_

#include <GL/glew.h>
#include <boost/tr1/memory.hpp>
#include <string>
#include <map>

namespace ogl {

class __Texture;

/**
 * @see __Texture
 */
typedef std::tr1::shared_ptr<__Texture> Texture;

/**
 * This class is a wrapper for OpenGL textures. It provides
 * methods to load them from many different file types and
 * allows to set texture parameters.
 */
class __Texture {
public:
	__Texture(GLuint textureID, GLuint stage);
	virtual ~__Texture();

	void setTexParameterf(GLenum pname, GLfloat param);
	void setTexParameteri(GLenum pname, GLint param);
	void setTexParameterfv(GLenum pname, const GLfloat* param);
	void setTexParameteriv(GLenum pname, const GLint* param);

	void setFilter(GLint min, GLint mag);
	void setWrap(GLint s, GLint t);

	void bind();
	void unbind();

	/**
	 * Activates the given (n-th) texture stage.
	 *
	 * @param stage The new stage, GL_TEXTURE0 + n.
	 */
	static void stage(GLuint stage);

	/**
	 * Creates and binds a new texture using the specified image file and
	 * the given target.
	 *
	 * @param file   The texture file in a common image format
	 * @param target The target of the texture
	 * @return       The newly create texture object
	 */
	static Texture load(std::string file, GLuint target = GL_TEXTURE_2D);

	/**
	 * Creates a new (empty) texture and binds it.
	 *
	 * @param target The target of the texture
	 * @return       The newly created texture object
	 */
	static Texture create(GLuint target = GL_TEXTURE_2D);

	GLuint m_textureID;
	GLuint m_target;
};

/**
 * A simple texture manager for named texture objects. Provides
 * methods to load and add new textures efficiently.
 */
class TextureMgr : public std::map<std::string, Texture> {
private:
	static TextureMgr* s_instance;
	TextureMgr();
	TextureMgr(const TextureMgr& other);
	virtual ~TextureMgr();
protected:
public:
	static TextureMgr& instance();
	static void destroy();

	/**
	 * Adds the given texture and associates it with the specified name.
	 *
	 * @param name    The name of the texture
	 * @param texture The texture object to add
	 * @return        The added texture object
	 */
	Texture add(const std::string& name, Texture texture);

	/**
	 * Loads all textures within the given folder. The texture names will
	 * be the base name of the files, i.e. the file name without its extension.
	 *
	 * @param folder The folder to load the textures from
	 */
	unsigned load(const std::string& folder);

	/**
	 * Returns the texture object with the given name, or an empty
	 * smart pointer if it does not exist.
	 *
	 * @param name The name of the texture
	 * @return     The associated texture object or an empty smart pointer
	 */
	Texture get(const std::string& name);
};


inline
void __Texture::setTexParameterf(GLenum pname, GLfloat param)
{
	glTexParameterf(m_target, pname, param);
}

inline
void __Texture::setTexParameteri(GLenum pname, GLint param)
{
	glTexParameteri(m_target, pname, param);
}

inline
void __Texture::setTexParameterfv(GLenum pname, const GLfloat* param)
{
	glTexParameterfv(m_target, pname, param);
}

inline
void __Texture::setTexParameteriv(GLenum pname, const GLint* param)
{
	glTexParameteriv(m_target, pname, param);
}

inline
void __Texture::setFilter(GLint min, GLint mag)
{
	setTexParameteri(GL_TEXTURE_MIN_FILTER, min);
	setTexParameteri(GL_TEXTURE_MAG_FILTER, mag);
}

inline
void __Texture::setWrap(GLint s, GLint t)
{
	setTexParameteri(GL_TEXTURE_WRAP_S, s);
	setTexParameteri(GL_TEXTURE_WRAP_T, t);
}

inline
void __Texture::bind()
{
	glBindTexture(m_target, m_textureID);
}

inline
void __Texture::unbind()
{
	glBindTexture(m_target, 0);
}


inline
void __Texture::stage(GLuint stage)
{
	glActiveTexture(stage);
}

inline
TextureMgr& TextureMgr::instance()
{
	if (!s_instance)
		s_instance = new TextureMgr();
	return *s_instance;
}

}

#endif /* TEXTURE_HPP_ */
