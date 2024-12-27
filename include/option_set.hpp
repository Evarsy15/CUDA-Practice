#ifndef OPTION_SET_H
#define OPTION_SET_H

#include <string>
#include <vector>

namespace Nix {

class OptionSet {
public:
    OptionSet() {}

    template <typename data_t>
    bool add_option(std::string option_name, std::string help_msg = "");

    template <typename data_t>
    bool get_option_value(std::string option_name, data_t& slot);

    void parse_command_line(const int argc, const char **argv);
    void show_help();

private:
    // https://stackoverflow.com/questions/41893489/using-template-struct-in-vector-with-deffrent-types
    class Option {
    protected:
        std::string m_option_name;
        std::string m_help_msg;
               bool m_is_option_set;
    
    public:
        Option() {} // Not recommend to call this null constructor
        Option(std::string option_name) 
            : m_option_name(option_name) { m_help_msg = ""; }
        Option(std::string option_name, std::string help_msg)
            : m_option_name(option_name), m_help_msg(help_msg) {}
        
        void set_help_msg(std::string help_msg) { m_help_msg = help_msg; }
        std::string get_option_name() { return m_option_name; }
        std::string get_help_msg()    { return m_help_msg; }
        
        void activate_option() { m_is_option_set = true; }
        void deactivate_option() { m_is_option_set = false; }
        bool check_if_option_set() { return m_is_option_set; }
    };

    template <typename data_t>
    class OptionWithData : public Option {
    private:
        data_t m_option_value;
    
    public:
        OptionWithData() {} // Not recommend to call this null constructor
        OptionWithData(std::string option_name) 
            : Option(option_name) {}
        OptionWithData(std::string option_name, std::string help_msg)
            : Option(option_name, help_msg) {}
        
        void set_option_value(data_t option_value) { m_option_value = option_value; }
        void get_option_value(data_t& slot) { slot = m_option_value; }

    }

    std::vector<Option*> m_option_list;

    bool is_option_already_exist(std::string option_name);
};

}

#endif