/**
 * @author Markus Holtermann
 * @date Jul 2, 2011
 * @file util/adapter.hpp
 */

#ifndef LISTENER_HPP_
#define LISTENER_HPP_

#include <list>

namespace util {

template<class T>
class Adapter {
protected:
	std::list<T*> m_listeners;
public:
	Adapter<T>();
	virtual ~Adapter();

	/**
	 * Adds a new listener.
	 *
	 * @param listener The listener to register.
	 */
	void addListener(T* listener);

	/**
	 * Removes a registered listener. Note that the listener
	 * will not be destroyed.
	 *
	 * @param listener THe listener to remove
	 */
	void removeListener(T* listener);

};

template<class T>
inline Adapter<T>::Adapter() {
}

template<class T>
inline Adapter<T>::~Adapter() {
	m_listeners.clear();
}

template<class T>
inline void Adapter<T>::addListener(T* listener) {
	m_listeners.push_back(listener);
}

template<class T>
inline void Adapter<T>::removeListener(T* listener) {
	m_listeners.remove(listener);
}

}
#endif /* LISTENER_HPP_ */
