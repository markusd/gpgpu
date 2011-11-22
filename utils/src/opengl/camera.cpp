/**
 * @author Markus Doellinger
 * @date May 26, 2011
 * @file opengl/camera.cpp
 */

#include <opengl/camera.hpp>
#include <GL/glew.h>

namespace ogl {


Camera::Camera()
{
	m_position = Vec3f(0.0f, 0.0f,  0.0f);
	m_eye 	   = Vec3f(0.0f, 0.0f, -1.0f);
	m_up       = Vec3f(0.0f, 1.0f,  0.0f);
	m_strafe   = Vec3f(0.0f, 0.0f,  0.0f);
	update();
}

Camera::~Camera()
{
}

bool Camera::checkAABB(const Vec3f& min, const Vec3f& max) const
{
	Vec3f near;
	float distance = 0.0f;
	for (int i = 0; i < 6; ++i) {
    	near.x = (m_frustum[i][0] > 0.0f) ? max.x : min.x;
		near.y = (m_frustum[i][1] > 0.0f) ? max.y : min.y;
		near.z = (m_frustum[i][2] > 0.0f) ? max.z : min.z;

		distance = m_frustum[i][0] * near.x + m_frustum[i][1] * near.y + m_frustum[i][2] * near.z + m_frustum[i][3];
    	if (distance < 0.0f)
			return false;
	}
	return true;
}

Camera::Visibility Camera::testAABB(const Vec3f& min, const Vec3f& max) const
{
	Visibility result = INSIDE;
	float distance = 0.0f;

	Vec3f vert;

	for (int i = 0; i < 6; ++i) {

		vert.x = (m_frustum[i][0] >= 0.0f) ? max.x : min.x;
		vert.y = (m_frustum[i][1] >= 0.0f) ? max.y : min.y;
		vert.z = (m_frustum[i][2] >= 0.0f) ? max.z : min.z;

		distance = m_frustum[i][0] * vert.x + m_frustum[i][1] * vert.y + m_frustum[i][2] * vert.z + m_frustum[i][3];
		if (distance < 0)
			return OUTSIDE;

		vert.x = (m_frustum[i][0] >= 0.0f) ? min.x : max.x;
		vert.y = (m_frustum[i][1] >= 0.0f) ? min.y : max.y;
		vert.z = (m_frustum[i][2] >= 0.0f) ? min.z : max.z;

    	distance = m_frustum[i][0] * vert.x + m_frustum[i][1] * vert.y + m_frustum[i][2] * vert.z + m_frustum[i][3];
		if (distance < 0)
			result =  INTERSECT;
	}
	return result;
}

Camera::Visibility Camera::testSphere(const Vec3f& center, float radius) const
{
	float distance;
	Visibility result = INSIDE;

	for (int i = 0; i < 6; ++i) {
		distance = m_frustum[i][0] * center.x + m_frustum[i][1] * center.y + m_frustum[i][2] * center.z + m_frustum[i][3];
		if (distance < -radius)
			return OUTSIDE;
		else if (distance < radius)
			result =  INTERSECT;
	}
	return result;
}

void Camera::update()
{
	m_strafe = (m_eye - m_position) % m_up;
	m_strafe.normalize();

	m_modelview = Mat4f::lookAt(m_position, m_eye, m_up);
	m_inverse = m_modelview.inverse();

	Mat4f mvproj = m_modelview * m_projection;

	// left
	m_frustum[0][0] = mvproj._14 + mvproj._11;
    m_frustum[0][1] = mvproj._24 + mvproj._21;
    m_frustum[0][2] = mvproj._34 + mvproj._31;
    m_frustum[0][3] = mvproj._44 + mvproj._41;
    float len = 1.0f / sqrt(m_frustum[0][0] * m_frustum[0][0] + m_frustum[0][1] * m_frustum[0][1] + m_frustum[0][2] * m_frustum[0][2]);
    m_frustum[0][0] *= len;
    m_frustum[0][1] *= len;
    m_frustum[0][2] *= len;
    m_frustum[0][3] *= len;

    // right
    m_frustum[1][0] = mvproj._14 - mvproj._11;
    m_frustum[1][1] = mvproj._24 - mvproj._21;
    m_frustum[1][2] = mvproj._34 - mvproj._31;
    m_frustum[1][3] = mvproj._44 - mvproj._41;
    len = 1.0f / sqrt(m_frustum[1][0] * m_frustum[1][0] + m_frustum[1][1] * m_frustum[1][1] + m_frustum[1][2] * m_frustum[1][2]);
    m_frustum[1][0] *= len;
    m_frustum[1][1] *= len;
    m_frustum[1][2] *= len;
    m_frustum[1][3] *= len;

    // bottom
    m_frustum[2][0] = mvproj._14 + mvproj._12;
    m_frustum[2][1] = mvproj._24 + mvproj._22;
    m_frustum[2][2] = mvproj._34 + mvproj._32;
    m_frustum[2][3] = mvproj._44 + mvproj._42;
    len = 1.0f / sqrt(m_frustum[2][0] * m_frustum[2][0] + m_frustum[2][1] * m_frustum[2][1] + m_frustum[2][2] * m_frustum[2][2]);
    m_frustum[2][0] *= len;
    m_frustum[2][1] *= len;
    m_frustum[2][2] *= len;
    m_frustum[2][3] *= len;

    // top
    m_frustum[3][0] = mvproj._14 - mvproj._12;
    m_frustum[3][1] = mvproj._24 - mvproj._22;
    m_frustum[3][2] = mvproj._34 - mvproj._32;
    m_frustum[3][3] = mvproj._44 - mvproj._42;
    len = 1.0f / sqrt(m_frustum[3][0] * m_frustum[3][0] + m_frustum[3][1] * m_frustum[3][1] + m_frustum[3][2] * m_frustum[3][2]);
    m_frustum[3][0] *= len;
    m_frustum[3][1] *= len;
    m_frustum[3][2] *= len;
    m_frustum[3][3] *= len;

    // near
    m_frustum[4][0] = mvproj._14 + mvproj._13;
    m_frustum[4][1] = mvproj._24 + mvproj._23;
    m_frustum[4][2] = mvproj._34 + mvproj._33;
    m_frustum[4][3] = mvproj._44 + mvproj._43;
    len = 1.0f / sqrt(m_frustum[4][0] * m_frustum[4][0] + m_frustum[4][1] * m_frustum[4][1] + m_frustum[4][2] * m_frustum[4][2]);
    m_frustum[4][0] *= len;
    m_frustum[4][1] *= len;
    m_frustum[4][2] *= len;
    m_frustum[4][3] *= len;

    // far
    m_frustum[5][0] = mvproj._14 - mvproj._13;
    m_frustum[5][1] = mvproj._24 - mvproj._23;
    m_frustum[5][2] = mvproj._34 - mvproj._33;
    m_frustum[5][3] = mvproj._44 - mvproj._43;
    len = 1.0f / sqrt(m_frustum[5][0] * m_frustum[5][0] + m_frustum[5][1] * m_frustum[5][1] + m_frustum[5][2] * m_frustum[5][2]);
    m_frustum[5][0] *= len;
    m_frustum[5][1] *= len;
    m_frustum[5][2] *= len;
    m_frustum[5][3] *= len;
}

void Camera::positionCamera(Vec3f position, Vec3f view, Vec3f up)
{
	m_position = position;
	m_eye      = position + view;
	m_up       = up;
	update();
}

void Camera::move(float amount)
{
	Vec3f view = m_eye - m_position;
	view *= amount / view.len();

	//if (view.y < 0.0f && m_position.y <= 0.0f)
	//	view.y = 0.0f;

	m_position += view;
	m_eye += view;
	update();
}

void Camera::strafe(float amount)
{
	m_position.x += m_strafe.x * amount;
	m_position.z += m_strafe.z * amount;

	m_eye.x += m_strafe.x * amount;
	m_eye.z += m_strafe.z * amount;
	update();
}

void Camera::rotate(float angle, Vec3f axis)
{
	// rotation around the axis
	Quatf rot(axis, angle * 3.141f / 180.0f);

	// current rotation
	Quatf view(0.0f,
			   m_eye.x - m_position.x,
			   m_eye.y - m_position.y,
			   m_eye.z - m_position.z);

	// combine new and old rotation and reduce by old rotation
	Quatf updated = ((rot * view) * rot.conjugated());

	// add the change in rotation
	m_eye.x = m_position.x + updated.b;
	m_eye.y = m_position.y + updated.c;
	m_eye.z = m_position.z + updated.d;
	update();
}

void Camera::apply()
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	//gluLookAt(m_position.x, m_position.y, m_position.z,
	//			   m_eye.x,	     m_eye.y,      m_eye.z,
	//			    m_up.x,       m_up.y,       m_up.z);
	glLoadMatrixf(m_modelview[0]);
}

Vec3f Camera::viewVector() const
{
	return (m_eye - m_position).normalized();
}

Vec3f Camera::pointer() const
{
	return pointer((m_viewport.z - m_viewport.x) / 2,
				   (m_viewport.w - m_viewport.y) / 2);
}

Vec3f Camera::pointer(int x, int y) const
{
	Mat4d modelview(m_modelview);
	Mat4d projection(m_projection);
	GLfloat z;

	y = m_viewport.w - y;

	glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &z);

	GLdouble _x, _y, _z;
	gluUnProject(x, y, z,
			modelview[0], projection[0], &m_viewport[0],
			&_x, &_y, &_z);

	return Vec3f(_x, _y, _z);
}

}
