#ifndef COMPAREUTIL_H
#define COMPAREUTIL_H

#include "tf/util/Base.h"

namespace CompareUtil {
    // This function returns true if the first pair is "larger"
    // than the second one according to some metric
    template<typename T>
    bool pairGreaterCompare(const std::pair<T, value_type> &firstElem, const std::pair<T, value_type> &secondElem) {
        return firstElem.second > secondElem.second;
    }

    // This function returns true if the first pair is "less"
    // than the second one according to some metric
    template<typename T>
    bool pairLessCompare(const std::pair<T, value_type> &firstElem, const std::pair<T, value_type> &secondElem) {
        return firstElem.second < secondElem.second;
    }
}

#endif //COMPAREUTIL_H
