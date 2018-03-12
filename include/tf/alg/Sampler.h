#ifndef SAMPLER_H
#define SAMPLER_H

#include "tf/util/Parameter.h"
#include "tf/util/Base.h"
#include "tf/struct/Tuple.h"
#include "tf/util/RandomUtil.h"
#include "tf/util/Data.h"
#include "tf/util/Calculator.h"

using namespace Calculator;

struct Sample {
    int relation_id;
    int p_sub;
    int p_obj;
    int n_sub;
    int n_obj;
};

namespace Sampler {

    inline void random_sample(Data &data, Sample &sample, const int random_idx) {
        Triple<int> &true_triple = data.training_triples[random_idx];
        sample.relation_id = true_triple.relation;
        sample.p_sub = true_triple.subject;
        sample.p_obj = true_triple.object;

        while (true) {
            int entity_id = RandomUtil::uniform_int(0, data.N);

            if (RandomUtil::uniform_int(0, 2) > 0) {
                sample.n_obj = sample.p_obj;
                sample.n_sub = entity_id;
            } else {
                sample.n_sub = sample.p_sub;
                sample.n_obj = entity_id;
            }

            if (data.faked_tuple_exist_train(sample.relation_id, sample.n_sub, sample.n_obj)) {
                continue;
            } else {
                break;
            }
        }
    }

    inline void random_sample_multithreaded(Data &data, Sample &sample, const int random_idx) {
        Triple<int> &true_triple = data.training_triples[random_idx];
        sample.relation_id = true_triple.relation;
        sample.p_sub = true_triple.subject;
        sample.p_obj = true_triple.object;

        while (true) {
            int entity_id = RandomUtil::randint_multithreaded(0, data.N);

            if (RandomUtil::randint_multithreaded(0, 2) > 0) {
                sample.n_obj = sample.p_obj;
                sample.n_sub = entity_id;
            } else {
                sample.n_sub = sample.p_sub;
                sample.n_obj = entity_id;
            }

            if (data.faked_tuple_exist_train(sample.relation_id, sample.n_sub, sample.n_obj)) {
                continue;
            } else {
                break;
            }
        }
    }
}

#endif //SAMPLER_H