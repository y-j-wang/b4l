#ifndef ENSEMBLE2_H
#define ENSEMBLE2_H

#include "tf/util/Base.h"
#include "tf/alg/RESCAL_RANK.h"
#include "tf/alg/TransE.h"

using namespace EvaluationUtil;
using namespace Calculator;
using namespace FileUtil;

class Ensemble2 : public TransE, public RESCAL_RANK {

private:

    vector<SimpleWeight> weights;

    void initialize() {
        current_epoch = 1;
        TransE::initialize();
        RESCAL_RANK::initialize();

        weights.resize(data->K);

        for (int i = 0; i < data->K; i++) {
            weights[i].w1 = 0.5;
            weights[i].w2 = 0.5;
        }
    }

    void update_weight(Sample &sample, vector<SimpleWeight> &weights) {
        SimpleWeight &weight = weights[sample.relation_id];

        weight.w1 -= parameter->step_size *
                     (cal_rescal_score(sample.relation_id, sample.p_sub, sample.p_obj, parameter->rescal_D,
                                       RESCAL_RANK::rescalA, RESCAL_RANK::rescalR) -
                      cal_rescal_score(sample.relation_id, sample.n_sub, sample.n_obj, parameter->rescal_D,
                                       RESCAL_RANK::rescalA, RESCAL_RANK::rescalR));
        weight.w2 += parameter->step_size *
                     (cal_transe_score(sample.p_sub, sample.p_obj, sample.relation_id, parameter->transe_D,
                                       parameter->L1_flag, TransE::transeA, TransE::transeR) -
                      cal_transe_score(sample.n_sub, sample.n_obj, sample.relation_id, parameter->transe_D,
                                       parameter->L1_flag, TransE::transeA, TransE::transeR));
    }

    void update(Sample &sample, const value_type weight = 1.0) {
        // violation will be the sum of two
        RESCAL_RANK::update(sample);
        TransE::update(sample);
        update_weight(sample, weights);
    }

    // ToDo: write its own evaluation
    string eval(const int epoch) {
        hit_rate measure = eval_hit_rate(Method::m_Ensemble, parameter, data, &(RESCAL_RANK::rescalA), &(RESCAL_RANK::rescalR), &(TransE::transeA),
                                         &(TransE::transeR), nullptr, nullptr, nullptr, &weights, parameter->output_path + "/ensemble2");

        string prefix = "Ensemble >>> ";

        print_hit_rate(prefix, parameter->hit_rate_topk, measure);

        if (parameter->eval_map) {
            pair<value_type, value_type> map = eval_MAP(m_Ensemble, parameter, data, &(RESCAL_RANK::rescalA), &(RESCAL_RANK::rescalR), &(TransE::transeA), &(TransE::transeR), nullptr, nullptr, nullptr,
                                                        &weights);

            string prefix = "MAP evalution >>> ";
            print_map(prefix, parameter->num_of_replaced_entities, map);
        }

        return "";
    }

    void output(const int epoch) {
        output_matrices(RESCAL_RANK::rescalA, RESCAL_RANK::rescalR, RESCAL_RANK::rescalA_G, RESCAL_RANK::rescalR_G,
                        epoch,
                        parameter->output_path + "/Ensemble2/RESCAL_RANK");
        output_matrices(TransE::transeA, TransE::transeR, data->N, parameter->transe_D, data->K, epoch,
                        parameter->output_path + "/Ensemble2/TransE");
        output_weights(weights, epoch, parameter->output_path + "/Weight");
    }

    // ToDo: write its own header
    string get_log_header() {
        return RESCAL_RANK::get_log_header();
    }

public:
    Ensemble2(Parameter *parameter, Data *data) : Optimizer(parameter, data), TransE(parameter, data),
                                                  RESCAL_RANK(parameter, data) {}

};

#endif //ENSEMBLE2_H
