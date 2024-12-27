#include "option_set.hpp"
#include "arg_parse.hpp"

using namespace Nix;

template <typename data_t>
void OptionSet::add_option(std::string option_name, std::string help_msg = "") {
    // Add option into option-list
    if (!is_option_exist(option_name))
        m_option_list.push_back(new Option<data_t>(option_name, help_msg));
}

template <typename data_t>
bool OptionSet::get_option_value(std::string option_value, data_t& slot) {
    if (!is_option_exist())
}

void OptionSet::show_help() {
    std::vector<OptionBase*>::iterator iter;
    for (iter = m_option_list.begin(); iter != m_option_list.end(); iter++) {
        std::cout << "--" << (*iter)->get_option_name() << " : \n";
        std::cout << "\t" << (*iter)->get_help_msg() << "\n";
    }
}

void OptionSet::parse_command_line(const int argc, const char **argv) {
    if (argc > 1) {
        int i = 1;
        while (i < argc) {
            
        }
    }
}

bool OptionSet::is_option_exist(std::string option_name) {
    std::vector<OptionBase*>::iterator iter;
    for (iter = m_option_list.begin(); iter != m_option_list.end(); iter++) {
        if ((*iter)->get_option_name() == option_name)
            return true;
    }
    return false;
}