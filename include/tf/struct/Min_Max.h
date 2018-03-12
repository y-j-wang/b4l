#ifndef MIN_MAX_H
#define MIN_MAX_H

struct min_max {
    // if parameter->normalize, it is min and max. if parameter->znormalize, it is mean and std
    value_type max1;
    value_type min1;
    value_type max2;
    value_type min2;
    value_type max3;
    value_type min3;
};
#endif //MIN_MAX_H
