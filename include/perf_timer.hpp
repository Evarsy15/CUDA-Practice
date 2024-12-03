#include <chrono>
#include <stdexcept>
#include <string>

namespace Nix {
/*
    Source for measuring kernel execution time

    Original code provided by :
        >> Adapted from WindyDarian (https://github.com/WindyDarian) <<
*/
class PerfTimer {
public:
    PerfTimer() {}
    ~PerfTimer() {}

    void startTimer() {
        if (isRunning) return;
        if (isPaused) {
            std::cout << "Warning : Called 'startTimer()' when the timer is paused.\n";
            std::cout << "Recommended to use 'resumeTimer()'.\n";
            resumeTimer();
        }

        isRunning = true;
        isPaused  = false;
        isEnded   = false;

        acc_elapsed_time_microseconds = 0.0;

        time_start = std::chrono::high_resolution_clock::now();
    }

    void pauseTimer() {
        time_end = std::chrono::high_resolution_clock::now();
        
        if (!isRunning) return;
      
        isRunning = false;
        isPaused  = true;
        isEnded   = false;

        std::chrono::duration<double, std::micro> duro = time_end - time_start;
        elapsed_time_microseconds = static_cast<double>(duro.count());
        acc_elapsed_time_microseconds += elapsed_time_microseconds;
    }

    void resumeTimer() {
        if (isRunning || isEnded) return;

        isRunning = true;
        isPaused  = false;
        isEnded   = false;

        time_start = std::chrono::high_resolution_clock::now();
    }

    void endTimer() {
        time_end = std::chrono::high_resolution_clock::now();
        
        if (!isRunning) return;
        if (!isPaused) {
            std::chrono::duration<double, std::micro> duro = time_end - time_start;
            elapsed_time_microseconds = static_cast<double>(duro.count());
            acc_elapsed_time_microseconds += elapsed_time_microseconds;
        }

        isRunning = false;
        isPaused  = false;
        isEnded   = true;
    }

    double getAccElapsedTime() {
        return acc_elapsed_time_microseconds;
    }

    void display() {
        if (isRunning) {
            throw std::runtime_error("Can't display timer while it is running.");
        }


    }

    // Remove copy and move functions
    PerfTimer(const PerfTimer&) = delete;
    PerfTimer(PerfTimer&&) = delete;
    PerfTimer& operator=(const PerfTimer&) = delete;
    PerfTimer& operator=(PerfTimer&&) = delete;

private:
    using time_point_t = std::chrono::high_resolution_clock::time_point;
    time_point_t time_start;
    time_point_t time_end;

    bool isRunning = false;
    bool isPaused  = false;
    bool isEnded   = true;

    double elapsed_time_microseconds = 0.0;
    double acc_elapsed_time_microseconds = 0.0;
};

}

#endif