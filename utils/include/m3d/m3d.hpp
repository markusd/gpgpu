/*
 * m3d.hpp
 *
 * A 3D math package.
 *
 *  Created on: May 14, 2011
 *      Author: Markus Doellinger
 */

#ifndef M3D_HPP_
#define M3D_HPP_

#ifdef _WIN32
#include <pstdint.h>
#else
#include <stdint.h>
#endif
#include <string>
#include <sstream>
#include <math.h>
#include <algorithm>

//#include <iostream>

#define M3D_USE_OPENGL 1

#ifdef M3D_USE_OPENGL
#include <GL/glew.h>
#endif

const double PI = 3.14159265358979323846;
const double EPSILON = 0.00001;

namespace m3d {

template<typename T> class Vec2;
typedef Vec2<float> Vec2f;
typedef Vec2<double> Vec2d;
typedef Vec2<int32_t> Vec2i;

template<typename T> class Vec3;
typedef Vec3<float> Vec3f;
typedef Vec3<double> Vec3d;
typedef Vec3<int32_t> Vec3i;

template<typename T> class Vec4;
typedef Vec4<float> Vec4f;
typedef Vec4<double> Vec4d;
typedef Vec4<int32_t> Vec4i;

template<typename T> class Mat4;
typedef Mat4<float> Mat4f;
typedef Mat4<double> Mat4d;

template<typename T> class Quat;
typedef Quat<float> Quatf;
typedef Quat<double> Quatd;

#include "vec2.hpp"
#include "vec3.hpp"
#include "vec4.hpp"
#include "mat4.hpp"
#include "quat.hpp"
#include "operators.hpp"

/**
 * Returns the intersection point of the ray (origin, dest) and the plane
 * (p1, p2, p3). p1, p2, p3 are three points on the plane.
 *
 * @param origin The origin of the ray
 * @param dest   The destination of the ray
 * @param p1     A point on the plane
 * @param p2     A point on the plane
 * @param p3     A point on the plane
 * @return
 */
template<typename T>
Vec3<T> rayPlaneIntersect(Vec3<T> origin, Vec3<T> dest, Vec3<T> p1, Vec3<T> p2, Vec3<T> p3)
{
	// compute
	Vec3<T> v1 = p2 - p1;
	Vec3<T> v2 = p3 - p1;
	Vec3<T> v3 = v1 % v2;

	// project ray onto the plane
	Vec3<T> proj1 = Vec3f(v1 * (origin - p1), v2 * (origin - p1), v3 * (origin - p1));
	Vec3<T> proj2 = Vec3f(v1 * (dest - p1),   v2 * (dest - p1),   v3 * (dest - p1));

	// check if the plane normal is parallel to the ray direction
	if (proj1.z == proj2.z) return Vec3<T>();

	// calculate ray coefficient
	float t = proj1.z / (proj2.z - proj1.z);

	// calculate the world-coordinates of the intersection
	return origin + (origin - dest) * t;
}

}

#endif /* M3D_HPP_ */
