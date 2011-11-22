/*
 * mat4.hpp
 *
 *  Created on: May 14, 2011
 *      Author: Markus Doellinger
 */

#ifndef MAT4_HPP_
#define MAT4_HPP_

template<typename T>
class Mat4 {
public:
	/**
	 * Constructs an empty matrix, i.e all elements are 0.
	 */
	Mat4<T>();

	/**
	 * Copies the matrix m.
	 *
	 * @param m The matrix to copy from.
	 */
	Mat4<T>(const Mat4<float>& m);

	/**
	 * Copies the matrix m.
	 *
	 * @param m The matrix to copy from.
	 */
	Mat4<T>(const Mat4<double>& m);

	/**
	 * Assigns the values of the two-dimensional array m to the
	 * components of the matrix, where _xy = m[x][y].
	 *
	 * @param m A two-dimensional array that are the components.
	 */
	Mat4<T>(const T m[4][4]);

	/**
	 * Assigns the values of the array m to the components of the
	 * matrix, where _xy = m[x + 4*y]
	 *
	 * @param m
	 */
	Mat4<T>(const T* const m);

	/**
	 * Assigns the parameter values to the corresponding components
	 * of the matrix.
	 */
	Mat4<T>(const T& _11, const T& _12, const T& _13, const T& _14,
			const T& _21, const T& _22, const T& _23, const T& _24,
			const T& _31, const T& _32, const T& _33, const T& _34,
			const T& _41, const T& _42, const T& _43, const T& _44);

	/**
	 * Constructs a matrix using the specified row vectors, where x = right,
	 * y = up, z = front and w = pos.
	 *
	 * @param right The first row vector of the matrix, i.e the x axis
	 * @param up 	The second row vector of the matrix, i.e the y axis
	 * @param front The third row vector of the matrix, i.e the z axis
	 * @param pos 	The fourth row vector of the matrix, i.e the position
	 */
	Mat4<T>(const Vec4<T>& right, const Vec4<T>& up, const Vec4<T>& front, const Vec4<T>& pos);

	/**
	 * Constructs an orthonormal matrix using the up vector, the tangent
	 * front and the position. All vectors will be normalized and will be
	 * pairwise perpendicular.
	 *
	 * @param up 	The up vector of the matrix, i.e. the y axis
	 * @param front The front vector of the matrix, i.e. the z axis
	 * @param pos	The position vector of the matrix
	 */
	Mat4<T>(Vec3<T> up, Vec3<T> front, Vec3<T> pos);

#ifdef M3D_USE_OPENGL
	/**
	 * Returns the current modelview matrix of OpenGL.
	 */
	static Mat4<T> modelview();

	/**
	 * Returns the current projection matrix of OpenGL.
	 */
	static Mat4<T> projection();
#endif

	Mat4<T>& operator*=(const Mat4<T>& m);
	Mat4<T>& operator*=(const T& t);

	Mat4<T> operator*(const Mat4<T>& m) const;
	Mat4<T> operator*(const T& t) const;

	Mat4<T> operator-() const;

	/**
	 * Multiplies the matrix m with this matrix without regarding
	 * the position vector. m %= r is the same as m' = r * m, where
	 * the m'.w = m.w.
	 *
	 * It is a convenience operator that avoids having to restore
	 * the position vector of a matrix after a multiplication
	 * where only the rotation part is important, i.e. if one wants to
	 * rotate an object's matrix around its axes.
	 *
	 * @param m A matrix that hold the rotation around the local axes
	 * 			of this matrix
	 * @return  A reference to this matrix
	 */
	Mat4<T>& operator%=(const Mat4<T>& m);

	/**
	 * Multiplies this matrix with m, only regarding the rotation part,
	 * i.e. x, y, and z. The position w of this matrix will be replaced
	 * by the position vector of m.
	 *
	 * @param m Multiplies the first three rows of the matrix with m and
	 * 			replaces the position.
	 * @return A reference to this matrix
	 */
	Mat4<T>& rotMultiply(const Mat4<T>& m);

	Mat4<T> transposed() const;
	Mat4<T> inverse() const;
	Mat4<T> orthonormalInverse() const;

	/**
	 * Creates the euler angles for the following sequence of
	 * rotation matrices: rotY * rotX * rotZ
	 *
	 * Note that when multiplying these matrices with a vector,
	 * one has to multiply the matrices in reverse order:
	 *
	 * v' = ((rotZ * rotX) * rotY) * v
	 *   or
	 * m = rotZ * rotX * rotY
	 * v' = m * v
	 */
	Vec3<T> eulerAngles() const;

	static Mat4<T> identity();
	static Mat4<T> translate(const Vec3<T>& translation);
	static Mat4<T> translate(const T& x, const T& y, const T& z);
	static Mat4<T> scale(const Vec3<T>& scale);
	static Mat4<T> rotX(const T& angle);
	static Mat4<T> rotY(const T& angle);
	static Mat4<T> rotZ(const T& angle);
	static Mat4<T> rotAxis(const Vec3<T>& vAxis, const T& angle);
	static Mat4<T> gramSchmidt(const Vec3<T>& dir, const Vec3<T>& pos);

	static Mat4<T> perspective(const T& fovy, const T& aspect, const T& zNear, const T& zFar);
	static Mat4<T> lookAt(const Vec3<T>& eye, const Vec3<T>& center, const Vec3<T>& up);

	Mat4<T>& setX(const Vec3<T>& x);
	Mat4<T>& setY(const Vec3<T>& y);
	Mat4<T>& setZ(const Vec3<T>& z);
	Mat4<T>& setW(const Vec3<T>& pos);

	Vec3<T> getX() const;
	Vec3<T> getY() const;
	Vec3<T> getZ() const;
	Vec3<T> getW() const;

	std::string str();
	void assign(std::string str);

	T* operator[](const int i);
	const T* operator[](const int i) const;

	T _11, _12, _13, _14; // x
	T _21, _22, _23, _24; // y
	T _31, _32, _33, _34; // z
	T _41, _42, _43, _44; // pos
};

template<typename T>
inline
Mat4<T>::Mat4()
	: _11((T)0), _12((T)0), _13((T)0), _14((T)0),
	  _21((T)0), _22((T)0), _23((T)0), _24((T)0),
	  _31((T)0), _32((T)0), _33((T)0), _34((T)0),
	  _41((T)0), _42((T)0), _43((T)0), _44((T)0)
{
}


template<typename T>
inline
Mat4<T>::Mat4(const Mat4<float>& m)
	: _11((T)m._11), _12((T)m._12), _13((T)m._13), _14((T)m._14),
	  _21((T)m._21), _22((T)m._22), _23((T)m._23), _24((T)m._24),
	  _31((T)m._31), _32((T)m._32), _33((T)m._33), _34((T)m._34),
	  _41((T)m._41), _42((T)m._42), _43((T)m._43), _44((T)m._44)
{
}

template<typename T>
inline
Mat4<T>::Mat4(const Mat4<double>& m)
	: _11((T)m._11), _12((T)m._12), _13((T)m._13), _14((T)m._14),
	  _21((T)m._21), _22((T)m._22), _23((T)m._23), _24((T)m._24),
	  _31((T)m._31), _32((T)m._32), _33((T)m._33), _34((T)m._34),
	  _41((T)m._41), _42((T)m._42), _43((T)m._43), _44((T)m._44)
{
}

template<typename T>
inline
Mat4<T>::Mat4(const T m[4][4])
{
	for (unsigned i = 0; i < 4; ++i) {
		for (unsigned j = 0; j < 4; ++j) {
			(*this)[i][j] = m[i][j];
		}
	}
	/*
	_11 = m[0][0]; _12 = m[0][1]; _13 = m[0][2]; _14 = m[0][3];
	_12 = m[1][0]; _22 = m[1][1]; _23 = m[1][2]; _24 = m[1][3];
	_13 = m[2][0]; _32 = m[2][1]; _33 = m[2][2]; _34 = m[2][3];
	_14 = m[3][0]; _42 = m[3][1]; _43 = m[3][2]; _44 = m[3][3];
	*/
}

template<typename T>
inline
Mat4<T>::Mat4(const T* const m)
{
	T* dest = &_11;
	const T* const src = m;
	for (unsigned i = 0; i < 16; ++i) {
		dest[i] = src[i];
	}
}

template<typename T>
inline
Mat4<T>::Mat4(const T& _11, const T& _12, const T& _13, const T& _14,
			  const T& _21, const T& _22, const T& _23, const T& _24,
			  const T& _31, const T& _32, const T& _33, const T& _34,
			  const T& _41, const T& _42, const T& _43, const T& _44)
	: _11(_11), _12(_12), _13(_13), _14(_14),
	  _21(_21), _22(_22), _23(_23), _24(_24),
	  _31(_31), _32(_32), _33(_33), _34(_34),
	  _41(_41), _42(_42), _43(_43), _44(_44)
{
}


template<typename T>
inline
Mat4<T>::Mat4(const Vec4<T>& right, const Vec4<T>& up, const Vec4<T>& front, const Vec4<T>& pos)
	: _11(right.x), _12(right.y), _13(right.z), _14(right.w),
		 _21(up.x),    _22(up.y),    _23(up.z),    _24(up.w),
	  _31(front.x), _32(front.y), _33(front.z), _34(front.w),
		_41(pos.x),   _42(pos.y),   _43(pos.z),   _44(pos.w)
{
}


template<typename T>
inline
Mat4<T>::Mat4(Vec3<T> up, Vec3<T> front, Vec3<T> pos)
{
	Vec3<T> right = -(front % up);
	front = -(up % right);

	right.normalize();
	up.normalize();
	front.normalize();

	_11 = right.x; _12 = right.y; _13 = right.z; _14 = 0.0f;
	_21 = up.x;    _22 = up.y;    _23 = up.z;    _24 = 0.0f;
	_31 = front.x; _32 = front.y; _33 = front.z; _34 = 0.0f;
	_41 = pos.x;   _42 = pos.y;   _43 = pos.z;   _44 = 1.0f;
}

template<typename T>
inline
Mat4<T> Mat4<T>::gramSchmidt(const Vec3<T>& dir, const Vec3<T>& pos)
{
	Mat4<T> r;
	Vec3<T> up;
	Vec3<T> right;
	Vec3<T> front(dir);

	front.normalize();
	if (fabs(front.z) > 0.577f) {
		right = front % Vec3<T>(-front.y, front.z, 0.0f);
	} else {
		right = front % Vec3<T>(-front.y, front.x, 0.0f);
	}
	right.normalize();
	up = right % front;
	front *= -1.0f;

	return Mat4<T>(right, up, front, Vec4<T>(pos.x, pos.y, pos.z, 1.0f));
}

#ifdef M3D_USE_OPENGL
template<typename T>
inline
Mat4<T> Mat4<T>::modelview()
{
	Mat4d res;
	glGetDoublev(GL_MODELVIEW_MATRIX, res[0]);
	return Mat4<T>(res);
}

template<typename T>
inline
Mat4<T> Mat4<T>::projection()
{
	Mat4d res;
	glGetDoublev(GL_PROJECTION_MATRIX, res[0]);
	return Mat4<T>(res);
}
#endif

template<typename T>
inline
Mat4<T>& Mat4<T>::operator*=(const Mat4<T>& m)
{
	T t1, t2, t3, t4;
	t1 = _11*m._11 + _12*m._21 + _13*m._31 + _14*m._41;
	t2 = _11*m._12 + _12*m._22 + _13*m._32 + _14*m._42;
	t3 = _11*m._13 + _12*m._23 + _13*m._33 + _14*m._43;
	t4 = _11*m._14 + _12*m._24 + _13*m._34 + _14*m._44;
	_11 = t1;
	_12 = t2;
	_13 = t3;
	_14 = t4;

	t1 = _21*m._11 + _22*m._21 + _23*m._31 + _24*m._41;
	t2 = _21*m._12 + _22*m._22 + _23*m._32 + _24*m._42;
	t3 = _21*m._13 + _22*m._23 + _23*m._33 + _24*m._43;
	t4 = _21*m._14 + _22*m._24 + _23*m._34 + _24*m._44;
	_21 = t1;
	_22 = t2;
	_23 = t3;
	_24 = t4;

	t1 = _31*m._11 + _32*m._21 + _33*m._31 + _34*m._41;
	t2 = _31*m._12 + _32*m._22 + _33*m._32 + _34*m._42;
	t3 = _31*m._13 + _32*m._23 + _33*m._33 + _34*m._43;
	t4 = _31*m._14 + _32*m._24 + _33*m._34 + _34*m._44;
	_31 = t1;
	_32 = t2;
	_33 = t3;
	_34 = t4;

	t1 = _41*m._11 + _42*m._21 + _43*m._31 + _44*m._41;
	t2 = _41*m._12 + _42*m._22 + _43*m._32 + _44*m._42;
	t3 = _41*m._13 + _42*m._23 + _43*m._33 + _44*m._43;
	t4 = _41*m._14 + _42*m._24 + _43*m._34 + _44*m._44;
	_41 = t1;
	_42 = t2;
	_43 = t3;
	_44 = t4;

	return *this;
}

template<typename T>
inline
Mat4<T>& Mat4<T>::operator*=(const T& t)
{
	_11 *= t; _12 *= t; _13 *= t; _14 *= t;
	_21 *= t; _22 *= t; _23 *= t; _24 *= t;
	_31 *= t; _32 *= t; _33 *= t; _34 *= t;
	_41 *= t; _42 *= t; _43 *= t; _44 *= t;
	return *this;
}

template<typename T>
inline
Mat4<T> Mat4<T>::operator*(const Mat4<T>& m) const
{
	return Mat4<T>(
	        (_11*m._11 + _12*m._21 + _13*m._31 + _14*m._41),
	        (_11*m._12 + _12*m._22 + _13*m._32 + _14*m._42),
	        (_11*m._13 + _12*m._23 + _13*m._33 + _14*m._43),
	        (_11*m._14 + _12*m._24 + _13*m._34 + _14*m._44),

	        (_21*m._11 + _22*m._21 + _23*m._31 + _24*m._41),
	        (_21*m._12 + _22*m._22 + _23*m._32 + _24*m._42),
	        (_21*m._13 + _22*m._23 + _23*m._33 + _24*m._43),
	        (_21*m._14 + _22*m._24 + _23*m._34 + _24*m._44),

	        (_31*m._11 + _32*m._21 + _33*m._31 + _34*m._41),
	        (_31*m._12 + _32*m._22 + _33*m._32 + _34*m._42),
	        (_31*m._13 + _32*m._23 + _33*m._33 + _34*m._43),
	        (_31*m._14 + _32*m._24 + _33*m._34 + _34*m._44),

	        (_41*m._11 + _42*m._21 + _43*m._31 + _44*m._41),
	        (_41*m._12 + _42*m._22 + _43*m._32 + _44*m._42),
	        (_41*m._13 + _42*m._23 + _43*m._33 + _44*m._43),
	        (_41*m._14 + _42*m._24 + _43*m._34 + _44*m._44)
	);
}

template<typename T>
inline
Mat4<T> Mat4<T>::operator*(const T& t) const
{
	return Mat4<T>(_11*t, _12*t, _13*t, _14*t,
				   _21*t, _22*t, _23*t, _24*t,
				   _31*t, _32*t, _33*t, _34*t,
				   _41*t, _42*t, _43*t, _44*t);
}

template<typename T>
inline
Mat4<T> Mat4<T>::operator-() const
{
	return Mat4<T>(-_11, -_12, -_13, -_14,
				   -_21, -_22, -_23, -_24,
				   -_31, -_32, -_33, -_34,
				   -_41, -_42, -_43, -_44);
}

template<typename T>
inline
Mat4<T>& Mat4<T>::operator%=(const Mat4<T>& m)
{

	Mat4<T> _m = m * *this;
	_m.setW(this->getW());
	_m.setX(_m.getX().normalized());
	_m.setY(_m.getY().normalized());
	_m.setZ(_m.getZ().normalized());
	*this = _m;
	return *this;

/*
	T t1, t2, t3, t4;
	t1 = m._11*_11 + m._12*_21 + m._13*_31 + m._14*_41;
	t2 = m._11*_12 + m._12*_22 + m._13*_32 + m._14*_42;
	t3 = m._11*_13 + m._12*_23 + m._13*_33 + m._14*_43;
	t4 = m._11*_14 + m._12*_24 + m._13*_34 + m._14*_44;
	_11 = t1;
	_12 = t2;
	_13 = t3;
	_14 = t4;

	std::cout << t1 << ", " << t2 << ", " << t3 << ", " << t4 << ", " << std::endl;

	t1 = m._21*_11 + m._22*_21 + m._23*_31 + m._24*_41;
	t2 = m._21*_12 + m._22*_22 + m._23*_32 + m._24*_42;
	t3 = m._21*_13 + m._22*_23 + m._23*_33 + m._24*_43;
	t4 = m._21*_14 + m._22*_24 + m._23*_34 + m._24*_44;
	_21 = t1;
	_22 = t2;
	_23 = t3;
	_24 = t4;

	std::cout << t1 << ", " << t2 << ", " << t3 << ", " << t4 << ", " << std::endl;


	t1 = m._31*_11 + m._32*_21 + m._33*_31 + m._34*_41;
	t2 = m._31*_12 + m._32*_22 + m._33*_32 + m._34*_42;
	t3 = m._31*_13 + m._32*_23 + m._33*_33 + m._34*_43;
	t4 = m._31*_14 + m._32*_24 + m._33*_34 + m._34*_44;
	_31 = t1;
	_32 = t2;
	_33 = t3;
	_34 = t4;

	// _32 and _33 need to be exchanged, no idea why though

	std::cout << t1 << ", " << t2 << ", " << t3 << ", " << t4 << ", " << std::endl;

*/
	return *this;
}

template<typename T>
inline
Mat4<T>& Mat4<T>::rotMultiply(const Mat4<T>& m)
{
	T t1, t2, t3, t4;
	t1 = _11*m._11 + _12*m._21 + _13*m._31 + _14*m._41;
	t2 = _11*m._12 + _12*m._22 + _13*m._32 + _14*m._42;
	t3 = _11*m._13 + _12*m._23 + _13*m._33 + _14*m._43;
	t4 = _11*m._14 + _12*m._24 + _13*m._34 + _14*m._44;
	_11 = t1;
	_12 = t2;
	_13 = t3;
	_14 = t4;

	//std::cout << t1 << ", " << t2 << ", " << t3 << ", " << t4 << ", " << std::endl;

	t1 = _21*m._11 + _22*m._21 + _23*m._31 + _24*m._41;
	t2 = _21*m._12 + _22*m._22 + _23*m._32 + _24*m._42;
	t3 = _21*m._13 + _22*m._23 + _23*m._33 + _24*m._43;
	t4 = _21*m._14 + _22*m._24 + _23*m._34 + _24*m._44;
	_21 = t1;
	_22 = t2;
	_23 = t3;
	_24 = t4;


	//std::cout << t1 << ", " << t2 << ", " << t3 << ", " << t4 << ", " << std::endl;

	t1 = _31*m._11 + _32*m._21 + _33*m._31 + _34*m._41;
	t2 = _31*m._12 + _32*m._22 + _33*m._32 + _34*m._42;
	t3 = _31*m._13 + _32*m._23 + _33*m._33 + _34*m._43;
	t4 = _31*m._14 + _32*m._24 + _33*m._34 + _34*m._44;
	_31 = t1;
	_32 = t2;
	_33 = t3;
	_34 = t4;


	//std::cout << t1 << ", " << t2 << ", " << t3 << ", " << t4 << ", " << std::endl;

	setW(m.getW());

	return *this;
}

template<typename T>
inline
Mat4<T> Mat4<T>::transposed() const
{
	Mat4<T> res;
	const T* const mat = &_11;
	T* dst = &res._11;

	for (unsigned i = 0; i < 4; i++) {
		dst[i]      = mat[i*4];
		dst[i + 4]  = mat[i*4 + 1];
		dst[i + 8]  = mat[i*4 + 2];
		dst[i + 12] = mat[i*4 + 3];
	}
	return res;
}

template<typename T>
inline
Mat4<T> Mat4<T>::inverse() const
{
	// Source: http://www.intel.com/design/pentiumiii/sml/245043.htm
	Mat4<T> res;
	T tmp00, tmp01, tmp02, tmp03;
	T tmp04, tmp05, tmp06, tmp07;
	T tmp08, tmp09, tmp10, tmp11;

	tmp00 = _33 * _44; tmp01 = _43 * _34; tmp02 = _23 * _44; tmp03 = _43 * _24;
	tmp04 = _23 * _34; tmp05 = _33 * _24; tmp06 = _13 * _44; tmp07 = _43 * _14;
	tmp08 = _13 * _34; tmp09 = _33 * _14; tmp10 = _13 * _24; tmp11 = _23 * _14;
	res._11  = tmp00*_22 + tmp03*_32 + tmp04*_42;
	res._11 -= tmp01*_22 + tmp02*_32 + tmp05*_42;
	res._12  = tmp01*_12 + tmp06*_32 + tmp09*_42;
	res._12 -= tmp00*_12 + tmp07*_32 + tmp08*_42;
	res._13  = tmp02*_12 + tmp07*_22 + tmp10*_42;
	res._13 -= tmp03*_12 + tmp06*_22 + tmp11*_42;
	res._14  = tmp05*_12 + tmp08*_22 + tmp11*_32;
	res._14 -= tmp04*_12 + tmp09*_22 + tmp10*_32;
	res._21  = tmp01*_21 + tmp02*_31 + tmp05*_41;
	res._21 -= tmp00*_21 + tmp03*_31 + tmp04*_41;
	res._22  = tmp00*_11 + tmp07*_31 + tmp08*_41;
	res._22 -= tmp01*_11 + tmp06*_31 + tmp09*_41;
	res._23  = tmp03*_11 + tmp06*_21 + tmp11*_41;
	res._23 -= tmp02*_11 + tmp07*_21 + tmp10*_41;
	res._24  = tmp04*_11 + tmp09*_21 + tmp10*_31;
	res._24 -= tmp05*_11 + tmp08*_21 + tmp11*_31;

	tmp00 = _31 * _42; tmp01 = _41 * _32; tmp02 = _21 * _42; tmp03 = _41 * _22;
	tmp04 = _21 * _32; tmp05 = _31 * _22; tmp06 = _11 * _42; tmp07 = _41 * _12;
	tmp08 = _11 * _32; tmp09 = _31 * _12; tmp10 = _11 * _22; tmp11 = _21 * _12;
	res._31  = tmp00*_24 + tmp03*_34 + tmp04*_44;
	res._31 -= tmp01*_24 + tmp02*_34 + tmp05*_44;
	res._32  = tmp01*_14 + tmp06*_34 + tmp09*_44;
	res._32 -= tmp00*_14 + tmp07*_34 + tmp08*_44;
	res._33  = tmp02*_14 + tmp07*_24 + tmp10*_44;
	res._33 -= tmp03*_14 + tmp06*_24 + tmp11*_44;
	res._34  = tmp05*_14 + tmp08*_24 + tmp11*_34;
	res._34 -= tmp04*_14 + tmp09*_24 + tmp10*_34;
	res._41  = tmp02*_33 + tmp05*_43 + tmp01*_23;
	res._41 -= tmp04*_43 + tmp00*_23 + tmp03*_33;
	res._42  = tmp08*_43 + tmp00*_13 + tmp07*_33;
	res._42 -= tmp06*_33 + tmp09*_43 + tmp01*_13;
	res._43  = tmp06*_23 + tmp11*_43 + tmp03*_13;
	res._43 -= tmp10*_43 + tmp02*_13 + tmp07*_23;
	res._44  = tmp10*_33 + tmp04*_13 + tmp09*_23;
	res._44 -= tmp08*_23 + tmp11*_33 + tmp05*_13;

	T det = _11*res._11 + _21*res._12 + _31*res._13 + _41*res._14;
	det = 1.0/det;
	res._11 *= det; res._12 *= det; res._13 *= det; res._14 *= det;
	res._21 *= det; res._22 *= det; res._23 *= det; res._24 *= det;
	res._31 *= det; res._32 *= det; res._33 *= det; res._34 *= det;
	res._41 *= det; res._42 *= det; res._43 *= det; res._44 *= det;
	return res;
}

template<typename T>
inline
Mat4<T> Mat4<T>::orthonormalInverse() const
{
	T x = _41*_11 + _42*_12 + _43*_13;
	T y = _41*_21 + _42*_22 + _43*_23;
	T z = _41*_31 + _42*_32 + _43*_33;
	return Mat4<T>(_11, _21, _31, 0,
				   _12, _22, _32, 0,
				   _13, _23, _33, 0,
				    -x,  -y,  -z, 1 );
}

template<typename T>
inline
Vec3<T> Mat4<T>::eulerAngles() const
{
	float yaw = 0.0f, roll = 0.0f, pitch = 0.0f;

	if (_32 > 0.99995) {
		roll = 0.0f;
		pitch = -PI / 2.0f;
		yaw = -roll - atan2(_21, _11);
	} else if (_32 < -0.99995) {
		roll = 0.0f;
		pitch = PI / 2.0f;
		yaw = roll + atan2(_21, _11);
	} else {
		pitch = asin(-_32);
		T t = cos(pitch);
		yaw = atan2(_31 / t, _33 / t);
		roll = atan2(_12 / t, _22 / t);
	}

	return Vec3<T>(pitch, yaw, roll);
}

template<typename T>
inline
Mat4<T> Mat4<T>::identity()
{
	return Mat4<T>(1, 0, 0, 0,
				   0, 1, 0, 0,
				   0, 0, 1, 0,
				   0, 0, 0, 1);
}

template<typename T>
inline
Mat4<T> Mat4<T>::translate(const Vec3<T>& t)
{
	return Mat4<T>(1, 0, 0, 0,
				   0, 1, 0, 0,
				   0, 0, 1, 0,
				   t.x, t.y, t.z, 1);
}

template<typename T>
inline
Mat4<T> Mat4<T>::translate(const T& x, const T& y, const T& z)
{
	return Mat4<T>(1, 0, 0, 0,
				   0, 1, 0, 0,
				   0, 0, 1, 0,
				   x, y, z, 1);
}

template<typename T>
inline
Mat4<T> Mat4<T>::scale(const Vec3<T>& scale)
{
	return Mat4<T>(scale.x, 0, 0, 0,
				   0, scale.y, 0, 0,
				   0, 0, scale.z, 0,
				   0, 0, 0, 1);
}

template<typename T>
inline
Mat4<T> Mat4<T>::rotX(const T& angle)
{
	T _sin = sin(angle);
	T _cos = cos(angle);
	return Mat4<T>(1,    0,    0, 0,
				   0, _cos, _sin, 0,
				   0,-_sin, _cos, 0,
				   0,    0,    0, 1);
}

template<typename T>
inline
Mat4<T> Mat4<T>::rotY(const T& angle)
{
	T _sin = sin(angle);
	T _cos = cos(angle);
	return Mat4<T>(_cos, 0,-_sin, 0,
					  0, 1,    0, 0,
				   _sin, 0, _cos, 0,
				      0, 0,    0, 1);
}

template<typename T>
inline
Mat4<T> Mat4<T>::rotZ(const T& angle)
{
	T _sin = sin(angle);
	T _cos = cos(angle);
	return Mat4<T>( _cos, _sin, 0, 0,
				   -_sin, _cos, 0, 0,
				       0,    0, 1, 0,
				       0,    0, 0, 1);
}


template<typename T>
inline
Mat4<T> Mat4<T>::rotAxis(const Vec3<T>& axis, const T& angle)
{
	Vec3<T> a = axis.normalized();
	T _sin = sin(angle);
	T _cos = cos(angle);
	T _icos = 1.0 - _cos;

	return Mat4<T>(
		a.x * a.x * _icos + _cos, 		a.x * a.y * _icos - a.z * _sin, a.x * a.z * _icos + a.y * _sin, 0.0,
		a.y * a.x * _icos + a.z * _sin, 	  a.y * a.y * _icos + _cos, a.y * a.z * _icos - a.x * _sin, 0.0,
		a.z * a.x * _icos - a.y * _sin, a.z * a.y * _icos + a.x * _sin, 	  a.z * a.z * _icos + _cos, 0.0,
								   0.0,                            0.0,                            0.0, 1.0
	);
}

template<typename T>
inline
Mat4<T> Mat4<T>::perspective(const T& fovy, const T& aspect, const T& zNear, const T& zFar)
{
	T f = tan(PI/2.0 - fovy/2.0);
	return Mat4<T>(
		f/aspect, (T)0,                          (T)0, (T) 0,
		(T)0,        f,                          (T)0, (T) 0,
		(T)0,     (T)0,     (zFar+zNear)/(zNear-zFar), (T)-1,
		(T)0,     (T)0, (2.0*zFar*zNear)/(zNear-zFar), (T) 0);
}

template<typename T>
inline
Mat4<T> Mat4<T>::lookAt(const Vec3<T>& eye, const Vec3<T>& center, const Vec3<T>& up)
{
	Vec3<T> f = (center - eye).normalized();
	Vec3<T> s = (f % up.normalized()).normalized();
	Vec3<T> u = s % f;
	return Mat4<T>(   s.x,    u.x,  -f.x, (T)0,
				      s.y,    u.y,  -f.y, (T)0,
				      s.z,    u.z,  -f.z, (T)0,
				   -s*eye, -u*eye, f*eye, (T)1);
}

template<typename T>
inline
Mat4<T>& Mat4<T>::setX(const Vec3<T>& x)
{
	_11 = x.x; _12 = x.y; _13 = x.z;
	return *this;
}

template<typename T>
inline
Mat4<T>& Mat4<T>::setY(const Vec3<T>& y)
{
	_21 = y.x; _22 = y.y; _23 = y.z;
	return *this;
}

template<typename T>
inline
Mat4<T>& Mat4<T>::setZ(const Vec3<T>& z)
{
	_31 = z.x; _32 = z.y; _33 = z.z;
	return *this;
}

template<typename T>
inline
Mat4<T>& Mat4<T>::setW(const Vec3<T>& pos)
{
	_41 = pos.x; _42 = pos.y; _43 = pos.z;
	return *this;
}

template<typename T>
inline
Vec3<T> Mat4<T>::getX() const
{
	return Vec3<T>(_11, _12, _13);
}

template<typename T>
inline
Vec3<T> Mat4<T>::getY() const
{
	return Vec3<T>(_21, _22, _23);
}

template<typename T>
inline
Vec3<T> Mat4<T>::getZ() const
{
	return Vec3<T>(_31, _32, _33);
}

template<typename T>
inline
Vec3<T> Mat4<T>::getW() const
{
	return Vec3<T>(_41, _42, _43);
}

template<typename T>
inline
std::string Mat4<T>::str()
{
	std::stringstream sst;
	sst << *this;
	return sst.str();
}

template<typename T>
inline
void Mat4<T>::assign(std::string str)
{
	std::stringstream sst;
	sst << str;
	sst.seekg(0, std::ios::beg);
	sst >> *this;
}

template<typename T>
inline
T* Mat4<T>::operator[](const int i)
{
	return (&_11 + 4*i);
}

template<typename T>
inline
const T* Mat4<T>::operator[](const int i) const
{
	return (&_11 + 4*i);
}

#endif /* MAT4_HPP_ */
