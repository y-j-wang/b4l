#ifndef RANDOMUTIL_H
#define RANDOMUTIL_H

#include "tf/util/Base.h"
#include <random>

namespace RandomUtil {

    std::random_device rd;
    std::mt19937 gen(rd());

    void init_seed(){
        srand(time(NULL));
    }

    /**
     * Low value inclusive, high value exclusive
     * @param low
     * @param high
     * @return
     */
    inline int uniform_int(int low, int high) {
        std::uniform_int_distribution<> distribution(low, high - 1);
        return distribution(gen);
    }

    /**
     * Low value inclusive, high value exclusive
     * @param low
     * @param high
     * @return
     */
    inline value_type uniform_real(const value_type low=-0.1, const value_type high=0.1) {
        std::uniform_real_distribution<value_type> distribution(low, high);
        return distribution(gen);
    }

    /**
     * Thread-safe function that returns a random number between min and max (exclusive).
     * This function takes ~142% the time that calling rand() would take. For this extra
     * cost you get a better uniform distribution and thread-safety.
     * https://stackoverflow.com/questions/21237905/how-do-i-generate-thread-safe-uniform-random-numbers
     * @param min
     * @param max
     * @return
     */
    inline int randint_multithreaded(const int min, const int max) {
        static thread_local std::mt19937 *generator = nullptr;
        if (!generator) {
            generator = new std::mt19937(clock() + std::hash<std::thread::id>()(std::this_thread::get_id()));
        }
        std::uniform_int_distribution<int> distribution(min, max - 1);
        return distribution(*generator);
    }

};
#endif //RANDOMUTIL_H
