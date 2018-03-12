#ifndef THREADUTIL_H
#define THREADUTIL_H

#include "tf/util/Base.h"

namespace ThreadUtil {

    inline void execute_threads(std::function<void(int)> &func, const int num_of_thread) {
        std::thread *exec_threads = cache_aligned_allocator<std::thread>().allocate(num_of_thread);

        for (int thread_index = 0; thread_index < num_of_thread; thread_index++) {
            cache_aligned_allocator<std::thread>().construct(exec_threads + thread_index, func,
                                                             thread_index);
        }

        for (int thread_index = 0; thread_index < num_of_thread; thread_index++) {
            exec_threads[thread_index].join();
        }

        for (int thread_index = 0; thread_index < num_of_thread; thread_index++) {
            cache_aligned_allocator<std::thread>().destroy(exec_threads + thread_index);
        }

        cache_aligned_allocator<std::thread>().deallocate(exec_threads, num_of_thread);
    }
}
#endif //THREADUTIL_H
