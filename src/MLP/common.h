#ifndef COMMON_H
#define COMMON_H

#include <chrono>
#include <stdexcept>
#include <string>

namespace Common {
  /**
  * This class is used for timing the performance
  * Uncopyable and unmovable
  *
  * Adapted from WindyDarian(https://github.com/WindyDarian)
  */
    class PerformanceTimer {
    public:
        PerformanceTimer() {}
        ~PerformanceTimer() {}

        void startCpuTimer() {
            if (cpu_timer_started) { throw std::runtime_error("CPU timer already started"); }
            cpu_timer_started = true;

            time_start_cpu = std::chrono::high_resolution_clock::now();
        }

        void endCpuTimer() {
            time_end_cpu = std::chrono::high_resolution_clock::now();

            if (!cpu_timer_started) { throw std::runtime_error("CPU timer not started"); }

            std::chrono::duration<double, std::milli> duro = time_end_cpu - time_start_cpu;
            prev_elapsed_time_cpu_milliseconds =
                static_cast<decltype(prev_elapsed_time_cpu_milliseconds)>(duro.count());

            cpu_timer_started = false;
        }

        float getCpuElapsedTimeForPreviousOperation() //noexcept
        {
            return prev_elapsed_time_cpu_milliseconds;
        }

        float getGpuElapsedTimeForPreviousOperation() //noexcept
        {
            return prev_elapsed_time_gpu_milliseconds;
        }

        // remove copy and move functions
        PerformanceTimer(const PerformanceTimer&) = delete;
        PerformanceTimer(PerformanceTimer&&) = delete;
        PerformanceTimer& operator=(const PerformanceTimer&) = delete;
        PerformanceTimer& operator=(PerformanceTimer&&) = delete;

    private:
        using time_point_t = std::chrono::high_resolution_clock::time_point;
        time_point_t time_start_cpu;
        time_point_t time_end_cpu;

        bool cpu_timer_started = false;
        bool gpu_timer_started = false;

        float prev_elapsed_time_cpu_milliseconds = 0.f;
        float prev_elapsed_time_gpu_milliseconds = 0.f;
    };
}

#endif