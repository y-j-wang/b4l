#ifndef TRIPLE_H
#define TRIPLE_H

#include "tf/util/Base.h"

template<typename T>
class Triple{
public:
    T subject;
    T relation;
    T object;

    Triple(){};
    Triple(T subject, T relation, T object):subject(subject), relation(relation), object(object) {}

    bool operator==(const Triple &b) const {
        if ((this->subject == b.subject) && (this->relation == b.relation) && (this->object == b.object)) {
            return true;
        } else {
            return false;
        }
    }
};
#endif //TRIPLE_H
