/**
 * @author Markus Holtermann
 * @date July 17, 2011
 * @file util/config.hpp
 */

#ifndef CONFIG_HPP_
#define CONFIG_HPP_

#include <string>
#include <map>
#undef min
#undef max

#include <boost/lexical_cast.hpp>

namespace util {

class Config: public std::map<std::string, std::string> {
private:
	// Singleton
	static Config* s_instance;
	Config();
	virtual ~Config();

protected:
public:
	static Config& instance();
	void destroy();

	void save(const std::string& path);
	bool load(const std::string& path);

	void set(const std::string& key, const std::string& value);
	void set(const std::string& key, bool value);
	template<typename T> void set(const std::string& key, T value);

	bool get(const std::string& key, bool def);
	template<typename T> T get(const std::string& key, T def);
};

inline Config& Config::instance()
{
	if (!s_instance)
		s_instance = new Config();
	return *s_instance;
}

inline
bool Config::get(const std::string& key, bool def)
{
	// lexical_cast does not convert "true" and "false"
	std::map<std::string, std::string>::iterator itr = find(key);
	if (itr != end()) {
		if (itr->second == "true") return true;
		if (itr->second == "false") return false;
		return get<bool>(key, def);
	}
	return def;
}

template<typename T>
inline
T Config::get(const std::string& key, T def)
{
	try {
		std::map<std::string, std::string>::iterator itr = find(key);
		if (itr != end())
			return boost::lexical_cast<T, std::string>(itr->second);
	} catch (...) {
	}
	return def;
}

template<typename T>
inline
void Config::set(const std::string& key, T value)
{
	(*this)[key] = boost::lexical_cast<std::string, T>(value);
}


}


#endif /* CONFIG_HPP_ */
