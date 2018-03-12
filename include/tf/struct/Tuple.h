#ifndef TUPLE_H
#define TUPLE_H

#include "tf/util/Base.h"

template<typename T>
class Tuple {
public:
    T subject;
    T object;

    Tuple() {}

    Tuple(int subject, int object) : subject(subject), object(object) {}

    bool operator==(const Tuple &b) const {
        if (this->subject == b.subject && this->object == b.object) {
            return true;
        } else {
            return false;
        }
    }

    bool operator>(const Tuple &b) const {
        if (subject > b.subject) {
            return true;
        } else if (subject < b.subject) {
            return false;
        } else {
            return (object > b.object);
        }
    }

    bool operator<(const Tuple &b) const {
        if (subject < b.subject) {
            return true;
        } else if (subject > b.subject) {
            return false;
        } else {
            return (object < b.object);
        }
    }
};

#endif //TUPLE_H
