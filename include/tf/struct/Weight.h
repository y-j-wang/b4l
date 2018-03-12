#ifndef WEIGHT_H
#define WEIGHT_H

#include "tf/util/Base.h"

class SimpleWeight{
public:
    value_type w1;
    value_type w2;
    value_type w3;

    SimpleWeight(){};

    SimpleWeight(value_type w1, value_type w2)
            : w1(w1), w2(w2) {}

    SimpleWeight(value_type w1, value_type w2, value_type w3)
            : w1(w1), w2(w2), w3(w3) {}

};
#endif //WEIGHT_H
