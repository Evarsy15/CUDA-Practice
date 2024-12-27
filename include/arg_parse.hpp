#ifndef ARG_PARSE_H
#define ARG_PARSE_H

/*
    <arg_parse.hpp> : Argument Parsing Helpers

    Source code from : <Common/helper_string.h> in https://github.com/NVIDIA/cuda-samples.git
*/

#include <stdlib.h>
#include <string.h>

// stringRemoveDelimiter(...) returns first place very after the delimiters('-').
inline int stringRemoveDelimiter(char delimiter, const char *string) {
    // assert(delimiter != '\0');
    int string_start = 0;

    while (string[string_start] == delimiter)
        string_start++;

    if (string_start >= static_cast<int>(strlen(string) - 1))
        return 0;
    
    return string_start;
}


/*
    checkCmdLineFlag(argc, argv, string_ref) :
        Check whether option name 'string_ref' exists in the argument.
*/
inline bool checkCmdLineFlag(const int argc, const char **argv, const char *string_ref) {
    bool bFound = false;

    if (argc >= 1) {
        for (int i = 1; i < argc; i++) {
            int string_start = stringRemoveDelimiter('-', argv[i]);
            const char *string_argv = &argv[i][string_start];

            const char *equal_pos = strchr(string_argv, '=');
            int argv_length = static_cast<int>(
                  equal_pos == 0 ? strlen(string_argv) : equal_pos - string_argv);

            int length = static_cast<int>(strlen(string_ref));

            if (length == argv_length &&
                  !strncmp(string_argv, string_ref, length)) {
                bFound = true;
                continue;
            }
        }
    }

    return bFound;
}

inline int getCmdLineArgumentInt(const int argc, const char **argv,
                                 const char *string_ref) {
    bool bFound = false;
    int value = -1;

    if (argc >= 1) {
        for (int i = 1; i < argc; i++) {
            int string_start = stringRemoveDelimiter('-', argv[i]);
            const char *string_argv = &argv[i][string_start];
            int length = static_cast<int>(strlen(string_ref));

            if (!strncmp(string_argv, string_ref, length)) {
                if (length + 1 <= static_cast<int>(strlen(string_argv))) {
                    int auto_inc = (string_argv[length] == '=') ? 1 : 0;
                    value = atoi(&string_argv[length + auto_inc]);
                } else value = 0;
                bFound = true;
                continue;
            }
        }
    }

    return bFound ? value : 0;
}

inline float getCmdLineArgumentFloat(const int argc, const char **argv,
                                     const char *string_ref) {
    bool bFound = false;
    float value = -1;

    if (argc >= 1) {
        for (int i = 1; i < argc; i++) {
            int string_start = stringRemoveDelimiter('-', argv[i]);
            const char *string_argv = &argv[i][string_start];
            int length = static_cast<int>(strlen(string_ref));

            if (!strncmp(string_argv, string_ref, length)) {
                if (length + 1 <= static_cast<int>(strlen(string_argv))) {
                    int auto_inc = (string_argv[length] == '=') ? 1 : 0;
                    value = static_cast<float>(atof(&string_argv[length + auto_inc]));
                } else {
                    value = 0.f;
                }

                bFound = true;
                continue;
            }
        }
    }

    return bFound ? value : 0;
}

inline double getCmdLineArgumentDouble(const int argc, const char **argv,
                                       const char *string_ref) {
    bool bFound = false;
    double value = -1;

    if (argc >= 1) {
        for (int i = 1; i < argc; i++) {
            int string_start = stringRemoveDelimiter('-', argv[i]);
            const char *string_argv = &argv[i][string_start];
            int length = static_cast<int>(strlen(string_ref));

            if (!strncmp(string_argv, string_ref, length)) {
                if (length + 1 <= static_cast<int>(strlen(string_argv))) {
                    int auto_inc = (string_argv[length] == '=') ? 1 : 0;
                    value = static_cast<double>(atof(&string_argv[length + auto_inc]));
                } else {
                    value = 0.f;
                }

                bFound = true;
                continue;
            }
        }
    }

    return bFound ? value : 0;
}

inline bool getCmdLineArgumentString(const int argc, const char **argv,
                                     const char *string_ref,
                                     char **string_retval) {
    bool bFound = false;

    if (argc >= 1) {
        for (int i = 1; i < argc; i++) {
            int string_start = stringRemoveDelimiter('-', argv[i]);
            char *string_argv = const_cast<char *>(&argv[i][string_start]);
            int length = static_cast<int>(strlen(string_ref));

            if (!strncmp(string_argv, string_ref, length)) {
                *string_retval = &string_argv[length + 1];
                bFound = true;
                continue;
            }
        }
    }

    if (!bFound)
        *string_retval = NULL;
    return bFound;
}

#endif