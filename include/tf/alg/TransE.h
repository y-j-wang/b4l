#ifndef TransE_H
#define TransE_H

#include "tf/alg/Optimizer.h"
#include "tf/util/Base.h"
#include "tf/util/RandomUtil.h"
#include "tf/util/Data.h"
#include "tf/util/Monitor.h"
#include "tf/util/CompareUtil.h"
#include "tf/util/Calculator.h"
#include "tf/util/EvaluationUtil.h"
#include "tf/util/Parameter.h"

using namespace EvaluationUtil;
using namespace Calculator;
using namespace FileUtil;

class TransE : virtual public Optimizer {

public:

    TransE(){};

    TransE(Parameter *parameter, Data *data) : Optimizer(parameter, data) {}

protected:

    DenseMatrix transeR;
    DenseMatrix transeA;
    DenseMatrix transeR_G;
    DenseMatrix transeA_G;

    void initialize() {

        RandomUtil::init_seed();

        // ToDo: why SGD cannot run if min_not_zeros is not initialized
        min_not_zeros.resize(parameter->transe_D);
        for (int i = 0; i < parameter->transe_D; i++) {
            min_not_zeros(i) = min_not_zero_value;
        }

        if (parameter->optimization == "adagrad" || parameter->optimization == "adadelta") {

            transeR_G.resize(data->K, parameter->transe_D);
            transeR_G.clear();

            transeA_G.resize(data->N, parameter->transe_D);
            transeA_G.clear();
        }

        current_epoch = parameter->restore_epoch + 1;

        transeR.resize(data->K, parameter->transe_D);
        transeA.resize(data->N, parameter->transe_D);

        if (parameter->restore_epoch >= 1) {

            if(parameter->optimization=="sgd") {
                read_transe_matrices(transeA, transeR, parameter->transe_D, parameter->restore_epoch,
                                     parameter->restore_path);
            } else {
                read_transe_matrices(transeA, transeR, transeA_G, transeR_G, parameter->transe_D, parameter->restore_epoch,
                                     parameter->restore_path);
            }

            Monitor timer;
            timer.start();

            hit_rate measure = eval_hit_rate(Method::m_TransE, parameter, data, nullptr, nullptr,
                                             &transeA, &transeR, nullptr, nullptr, nullptr, nullptr, parameter->output_path + "/TransE_test_epoch_" + to_string(parameter->restore_epoch));
            timer.stop();

            string prefix = "[TransE] restore to epoch " + to_string(parameter->restore_epoch) + " ";

            print_hit_rate(prefix, parameter->hit_rate_topk, measure);

            cout << "evaluation time: " << timer.getElapsedTime() << " secs" << endl;

            cout << "------------------------" << endl;

        } else {

            for (int i = 0; i < data->N; ++i) {
                value_type *entity = transeA.data().begin() + i * parameter->transe_D;
                for (int d = 0; d < parameter->transe_D; ++d) {
                    entity[d] = RandomUtil::uniform_real() * 6 / sqrt(parameter->transe_D);
                }
//                normalize(entity, parameter->transe_D);
            }

            for (int j = 0; j < data->K; ++j) {
                value_type *relation = transeR.data().begin() + j * parameter->transe_D;
                for (int d = 0; d < parameter->transe_D; ++d) {
                    relation[d] = RandomUtil::uniform_real() * 6 / sqrt(parameter->transe_D);
                }

                normalize(relation, parameter->transe_D);
            }
        }
    }

    void update(Sample &sample, const value_type weight = 1.0) {

        bool subject_replace = (sample.n_sub != sample.p_sub); // true: subject is replaced, false: object is replaced.

        value_type positive_score = cal_transe_score(sample.p_sub, sample.p_obj, sample.relation_id, parameter->transe_D, parameter->L1_flag, transeA, transeR);
        value_type negative_score = cal_transe_score(sample.n_sub, sample.n_obj, sample.relation_id, parameter->transe_D, parameter->L1_flag, transeA, transeR);

        value_type margin = positive_score + parameter->margin - negative_score;

        if (margin > 0) {

            violations++;

            value_type *p_sub_vec = transeA.data().begin() + sample.p_sub * parameter->transe_D;
            value_type *p_obj_vec = transeA.data().begin() + sample.p_obj * parameter->transe_D;

            value_type *n_sub_vec = transeA.data().begin() + sample.n_sub * parameter->transe_D;
            value_type *n_obj_vec = transeA.data().begin() + sample.n_obj * parameter->transe_D;

            value_type *rel_vec = transeR.data().begin() + sample.relation_id * parameter->transe_D;

            Vec x = 2 * (row(transeA, sample.p_obj) - row(transeA, sample.p_sub) - row(transeR, sample.relation_id));

            if (parameter->L1_flag) {
                for (int i = 0; i < parameter->transe_D; i++) {
                    if (x(i) > 0) {
                        x(i) = 1;
                    } else {
                        x(i) = -1;
                    }
                }
            }

            (this->*update_grad)(rel_vec, x.data().begin(), transeR_G.data().begin() + sample.relation_id * parameter->transe_D, parameter->transe_D, weight);
            (this->*update_grad)(p_sub_vec, x.data().begin(), transeA_G.data().begin() + sample.p_sub * parameter->transe_D, parameter->transe_D, weight);
            (this->*update_grad)(p_obj_vec, x.data().begin(), transeA_G.data().begin() + sample.p_obj * parameter->transe_D, parameter->transe_D, - weight);

            x = 2 * (row(transeA, sample.n_obj) - row(transeA, sample.n_sub) - row(transeR, sample.relation_id));

            if (parameter->L1_flag) {
                for (int i = 0; i < parameter->transe_D; i++) {
                    if (x(i) > 0) {
                        x(i) = 1;
                    } else {
                        x(i) = -1;
                    }
                }
            }

            (this->*update_grad)(rel_vec, x.data().begin(), transeR_G.data().begin() + sample.relation_id * parameter->transe_D, parameter->transe_D, - weight);
            if (subject_replace) {
                (this->*update_grad)(n_sub_vec, x.data().begin(), transeA_G.data().begin() + sample.n_sub * parameter->transe_D, parameter->transe_D, - weight);
                (this->*update_grad)(p_obj_vec, x.data().begin(), transeA_G.data().begin() + sample.p_obj * parameter->transe_D, parameter->transe_D, weight);
            } else {
                (this->*update_grad)(p_sub_vec, x.data().begin(), transeA_G.data().begin() + sample.p_sub * parameter->transe_D, parameter->transe_D, - weight);
                (this->*update_grad)(n_obj_vec, x.data().begin(), transeA_G.data().begin() + sample.n_obj * parameter->transe_D, parameter->transe_D, weight);
            }

            normalizeOne(rel_vec, parameter->transe_D);
            normalizeOne(p_sub_vec, parameter->transe_D);
            normalizeOne(p_obj_vec, parameter->transe_D);

            if (subject_replace) {
                normalizeOne(n_sub_vec, parameter->transe_D);
            } else {
                normalizeOne(n_obj_vec, parameter->transe_D);
            }

        }

    }

    string eval(const int epoch){

        if (parameter->eval_train) {

            hit_rate train_measure = eval_transe_train(parameter, data,
                                                       transeA,
                                                       transeR,
                                                       parameter->output_path + "/TransE_train_epoch_" +
                                                       to_string(epoch));

            string prefix = "sampled training data >>> ";
            print_hit_rate_train(prefix, parameter->hit_rate_topk, train_measure);

        }

        hit_rate testing_measure = eval_hit_rate(Method::m_TransE, parameter, data, nullptr, nullptr,
                                                 &transeA, &transeR, nullptr, nullptr, nullptr, nullptr, parameter->output_path + "/TransE_test_epoch_" + to_string(epoch));

        string prefix = "testing data >>> ";
        print_hit_rate(prefix, parameter->hit_rate_topk, testing_measure);

        if (parameter->eval_rel) {

            hit_rate rel_measure = eval_relation_transe(parameter, data, transeA,
                                                        transeR,
                                                        parameter->output_path + "/TransE_test_epoch_" +
                                                        to_string(epoch));

            string prefix = "testing data relation evaluation >>>  ";
            print_hit_rate_rel(prefix, parameter->hit_rate_topk, rel_measure);
        }

        pair<value_type, value_type> map;
        map.first = -1;
        map.second = -1;

        if (parameter->eval_map) {
            map = eval_MAP(m_TransE, parameter, data, nullptr, nullptr, &transeA, &transeR, nullptr, nullptr, nullptr, nullptr);
            string prefix = "testing data MAP evalution >>> ";
            print_map(prefix, parameter->num_of_replaced_entities, map);
        }

        string log = "";
        log.append("TransE,");
        log.append(parameter->optimization + ",");
        log.append(to_string(epoch) + ",");
        log.append(to_string(parameter->transe_D) + ",");
        log.append(to_string(parameter->step_size) + ",");
        log.append(to_string(parameter->margin) + ",");
        log.append(to_string(parameter->L1_flag) + ",");
        if(parameter->optimization=="adadelta") {
            log.append(to_string(parameter->Rho) + ",");
        }

        string count_s = (testing_measure.count_s == -1? "Not Computed" : to_string(testing_measure.count_s));
        log.append(count_s + ",");

        string count_o = (testing_measure.count_o == -1? "Not Computed" : to_string(testing_measure.count_o));
        log.append(count_o + ",");

        string count_s_ranking = (testing_measure.count_s_ranking == -1? "Not Computed" : to_string(testing_measure.count_s_ranking));
        log.append(count_s_ranking + ",");

        string count_o_ranking = (testing_measure.count_o_ranking == -1? "Not Computed" : to_string(testing_measure.count_o_ranking));
        log.append(count_o_ranking + ",");

        log.append(to_string(testing_measure.count_s_filtering) + ",");
        log.append(to_string(testing_measure.count_o_filtering) + ",");
        log.append(to_string(testing_measure.count_s_ranking_filtering) + ",");
        log.append(to_string(testing_measure.count_o_ranking_filtering) + ",");

        string map1 = (map.first == -1 ? "Not Computed" : to_string(map.first));
        string map2 = (map.second == -1 ? "Not Computed" : to_string(map.second));

        log.append(map1 + ",");
        log.append(map2 + ",");

        string inv_count_s_ranking = (testing_measure.inv_count_s_ranking == -1? "Not Computed" : to_string(testing_measure.inv_count_s_ranking));
        log.append(count_s_ranking + ",");

        string inv_count_o_ranking = (testing_measure.inv_count_o_ranking == -1? "Not Computed" : to_string(testing_measure.inv_count_o_ranking));
        log.append(count_o_ranking + ",");

        log.append(to_string(testing_measure.inv_count_s_ranking_filtering) + ",");
        log.append(to_string(testing_measure.inv_count_o_ranking_filtering));

        return log;
    }

    string get_log_header() {
        string header = "Method,Optimization,epoch,Dimension,step size,margin,L1 flag,";
        header += ((parameter->optimization=="adadelta")?"Rho,":"");
        header += "hit_rate_subject@" +
                  to_string(parameter->hit_rate_topk) + ",hit_rate_object@" +
                  to_string(parameter->hit_rate_topk) +
                  ",subject_ranking,object_ranking,hit_rate_subject_filter@" +
                  to_string(parameter->hit_rate_topk) + ",hit_rate_object_filter@" +
                  to_string(parameter->hit_rate_topk) +
                  ",subject_ranking_filter,object_ranking_filter,MAP_subject@" +
                  to_string(parameter->num_of_replaced_entities) + ",MAP_object@" + to_string(parameter->num_of_replaced_entities) +
                  ",MRR_subject,MRR_object,MRR_subject_filter,MRR_object_filter";
        return header;
    }

    void output(const int epoch) {

        if(parameter->optimization=="sgd"){
            output_matrices(transeA, transeR, data->N, parameter->transe_D, data->K, epoch,
                            parameter->output_path);
        } else {
            output_matrices(transeA, transeR, transeA_G, transeR_G, data->N, parameter->transe_D, data->K, epoch,
                            parameter->output_path);
        }

    }
};

#endif //TransE_H