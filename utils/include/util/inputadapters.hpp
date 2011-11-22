/**
 * @author Markus Doellinger, Markus Holtermann
 * @date May 25, 2011
 * @file util/inputadapters.hpp
 */

#ifndef INPUTADAPTERS_HPP_
#define INPUTADAPTERS_HPP_

#include <set>
#include <list>
#include <util/adapter.hpp>

namespace util {

/**
 * An input adapter class that monitors key events and provides
 * them to components independent from the windowing system.
 *
 * The actual monitoring functionality is implemented by the
 * classes inheriting from KeyAdapter.
 */
class KeyAdapter {
protected:
	bool m_alt;
	bool m_ctrl;
	bool m_shift;

	/** The set of pressed keys */
	std::set<unsigned char> m_pressed;

public:
	KeyAdapter();

	/**
	 * Returns the state of the given key in ASCII format.
	 *
	 * @param key The key to query
	 * @return    True, if the key is down, false otherwise
	 */
	inline bool isDown(unsigned char key) { return m_pressed.find(key) != m_pressed.end(); };

	/**
	 * @return True if shift is pressed, false otherwise
	 */
	inline bool shift() { return m_shift; };

	/**
	 * @return True if control is pressed, false otherwise
	 */
	inline bool ctrl() { return m_ctrl; };

	/**
	 * @return True if alt is pressed, false otherwise
	 */
	inline bool alt() { return m_alt; };
};

/**
 * A key adapter that monitors simple ASCII key events, for example
 * for the GLUT windowing system.
 */
class AsciiKeyAdapter : public KeyAdapter {
public:
	void keyEvent(bool down, unsigned char key);
	void changeAlt(bool alt) { m_alt = alt; };
	void changeCtrl(bool ctrl) { m_ctrl = ctrl; };
	void changeShift(bool shift) { m_shift = shift; };
};

/**
 * An enumeration defining the mouse buttons
 */
enum Button {
	LEFT = 0, //!< LEFT
	RIGHT = 1, //!< RIGHT
	MIDDLE = 2 //!< MIDDLE
};

/**
 * A listener interface that can be called when mouse events occur.
 */
class MouseListener {
public:
	/**
	 * This method is called whenever the mouse moves.
	 *
	 * @param x      The x position of the mouse
	 * @param y      The y position of the mouse
	 */
	virtual void mouseMove(int x, int y) = 0;

	/**
	 * This method is called whenever a mouse button is pressed or
	 * released.
	 *
	 * @param button The button that was pressed
	 * @param down   True, if the mouseButton is down
	 * @param x      The x position of the mouse
	 * @param y      The y position of the mouse
	 */
	virtual void mouseButton(Button button, bool down, int x, int y) = 0;

	/**
	 * This methid is called whenever a double click was made by the user.
	 *
	 * @param button The button that was pressed
	 * @param x      The x position of the mouse
	 * @param y      The y position of the mouse
	 */
	virtual void mouseDoubleClick(Button button, int x, int y) = 0;

	/**
	 * This method is called whenever the mouse wheel is rotated
	 *
	 * @param delta	 The delta of rotation in eighths of a degree
	 */
	virtual void mouseWheel(int delta) = 0;
};

/**
 * An input adapter class that monitors mouse events and provides
 * them to components independent from the windowing system. Listeners
 * can be registered whose methods will be called when a mouse event
 * occurs.
 *
 * The actual monitoring functionality is implemented by the
 * classes inheriting from MouseAdapter.
 */
class MouseAdapter: public Adapter<MouseListener> {
protected:
	/** Indicates the state of the three buttons */
	bool m_down[3];

	/** The x and y position of the mouse */
	int m_x, m_y;
public:
	MouseAdapter();

	/**
	 * Queries the state of a mouse button
	 *
	 * @param button The button of interest
	 * @return True, if the button is pressed, false otherwise
	 */
	bool isDown(Button button);

	/**
	 * @return The x position of the mouse.
	 */
	inline int getX() { return m_x; };

	/**
	 * @return The y position of the mouse.
	 */
	inline int getY() { return m_y; };
};

/**
 * A simple mouse adapter that provides generic methods to set
 * the position and button states.
 */
class SimpleMouseAdapter : public MouseAdapter {
public:
	/**
	 * Must be called whenever the mouse is moved.
	 *
	 * @param x      The x position of the mouse
	 * @param y      The y position of the mouse
	 */
	void mouseMove(int x, int y);

	/**
	 * Must be called whenever a button is pressed or released.
	 *
	 * @param button The button of the event
	 * @param down   True if the button was pressed, false otherwise
	 * @param x      The x position of the mouse
	 * @param y      The y position of the mouse
	 */
	void mouseButton(Button button, bool down, int x, int y);
};

}

#endif /* INPUTADAPTERS_HPP_ */
