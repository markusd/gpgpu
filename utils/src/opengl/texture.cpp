/**
 * @author Markus Doellinger, Robert Waury
 * @date May 21, 2011
 * @file opengl/texture.cpp
 */

#include <opengl/texture.hpp>
#include <opengl/stb_image.hpp>
#define BOOST_FILESYSTEM_VERSION 2
#include <boost/filesystem.hpp>
#include <stdexcept>
#include <iostream>
#include <util/config.hpp>

namespace ogl {

TextureMgr* TextureMgr::s_instance = NULL;

__Texture::__Texture(GLuint textureID, GLuint target)
	: m_textureID(textureID), m_target(target)
{
}

__Texture::~__Texture()
{
#ifdef _DEBUG
	std::cout << "delete texture" << std::endl;
#endif
	if (glIsTexture(m_textureID))
		glDeleteTextures(0, &m_textureID);
}

Texture __Texture::load(std::string file, GLuint target)
{
	int w, h, c;
	unsigned char* data = stbi_load(file.c_str(), &w, &h, &c, STBI_rgb_alpha);

	if (!data)
		return Texture();

    GLuint textureID = 0;
    glGenTextures(1, &textureID);
    glBindTexture(target, textureID);

    //glTexImage2D(stage, 0, GL_RGBA, Image.GetWidth(), Image.GetHeight(), 0, GL_RGB, GL_UNSIGNED_BYTE, Image.GetPixelsPtr());
    //gluBuild2DMipmaps(target, GL_RGBA, Image.GetWidth(), Image.GetHeight(), GL_RGBA, GL_UNSIGNED_BYTE, Image.GetPixelsPtr());
    gluBuild2DMipmaps(target, GL_RGBA, w, h, GL_RGBA, GL_UNSIGNED_BYTE, data);

    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

    glTexParameteri(target, GL_TEXTURE_WRAP_R, GL_REPEAT);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_REPEAT);

    // enable anisotropic filtering
    if (util::Config::instance().get("useAF", false) && GLEW_EXT_texture_filter_anisotropic) {
		float maxAF;
		glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAF);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, maxAF);
    }

    Texture result(new __Texture(textureID, target));
    return result;
}

Texture __Texture::create(GLuint target)
{
    GLuint textureID = 0;
    glGenTextures(1, &textureID);
    glBindTexture(target, textureID);

    Texture result(new __Texture(textureID, target));
    return result;
}

TextureMgr::TextureMgr()
	: std::map<std::string, Texture>()
{
}

TextureMgr::TextureMgr(const TextureMgr& other)
{
}

TextureMgr::~TextureMgr()
{
}

void TextureMgr::destroy()
{
	if (s_instance)
		delete s_instance;
	s_instance = NULL;
}

Texture TextureMgr::add(const std::string& name, Texture texture)
{
	(*this)[name] = texture;
	return texture;
}

unsigned TextureMgr::load(const std::string& folder)
{
	int count = 0;
	using namespace boost::filesystem;

	path p (folder);
	
	if(is_directory(p)) {
		if(!is_empty(p)) {
			directory_iterator end_itr;
			for(directory_iterator itr(p); itr != end_itr; ++itr) {
				if(itr->leaf().size() > 3) {
					Texture texture = __Texture::load(itr->string(), GL_TEXTURE_2D);
					add(basename(*itr), texture);
				}
			}
		}

	}
	// open folder
	// iterate over the folder
	// find *.[a-zA-Z] files
	// load with Texture Manager
	// close folder
	return count;
}

Texture TextureMgr::get(const std::string& name)
{
	TextureMgr::iterator it = this->find(name);
	if (it == this->end()) {
		Texture result;
		return result;
	}
	return it->second;

}

}
