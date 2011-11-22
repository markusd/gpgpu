/**
 * @author Markus Doellinger, Markus Holtermann
 * @date May 25, 2011
 * @file util/inputadapters.cpp
 */

#include <util/inputadapters.hpp>
#include <iostream>

namespace util {

KeyAdapter::KeyAdapter()
	: m_alt(false),
	  m_ctrl(false),
	  m_shift(false)
{
}

void AsciiKeyAdapter::keyEvent(bool down, unsigned char key)
{
	if (down)
		m_pressed.insert(key);
	else
		m_pressed.erase(key);
}

MouseAdapter::MouseAdapter() : Adapter<MouseListener>()
{
	m_down[0] = m_down[1] = m_down[2] = 0;
}

bool MouseAdapter::isDown(Button button)
{
	return m_down[button];
}

void SimpleMouseAdapter::mouseMove(int x, int y)
{
	for (std::list<MouseListener*>::iterator itr = m_listeners.begin();
			itr != m_listeners.end(); ++itr)
		(*itr)->mouseMove(x, y);
	m_x = x;
	m_y = y;
}

void SimpleMouseAdapter::mouseButton(Button button, bool down, int x, int y)
{
	m_down[button] = down;
	for (std::list<MouseListener*>::iterator itr = m_listeners.begin();
			itr != m_listeners.end(); ++itr)
		(*itr)->mouseButton(button, down, x, y);
	m_x = x;
	m_y = y;
}

}
