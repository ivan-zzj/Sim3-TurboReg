#ifndef __UTILS_DEBUG_HPP__
#define __UTILS_DEBUG_HPP__

#include <iostream>

#define PRINT(...)                 \
    do                             \
    {                              \
        std::cout << "(";          \
        print_helper(__VA_ARGS__); \
        std::cout << ")\n";        \
    } while (0)

template <typename T>
void print_helper(const T &arg)
{
    std::cout << arg;
}

template <typename T, typename... Args>
void print_helper(const T &arg, const Args &...args)
{
    std::cout << arg << " | "; 
    print_helper(args...);     
}

#define IS_SHOW_LOG
#ifdef IS_SHOW_LOG
#define TIME_BLOCK(task_name, code_block)                                                \
    {                                                                                    \
        auto start = std::chrono::high_resolution_clock::now();                          \
        code_block auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>( \
                                      std::chrono::high_resolution_clock::now() - start) \
                                      .count();                                          \
        std::cout << "[TIME] " << task_name << ": " << time_ms << " (ms)"                \
                  << std::endl;                                                          \
    }
#else
#define TIME_BLOCK(task_name, code_block) code_block

#endif
#endif