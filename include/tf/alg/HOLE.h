#ifndef HOLE_H
#define HOLE_H

#include "tf/alg/Optimizer.h"
#include "tf/util/Base.h"
#include "tf/util/RandomUtil.h"
#include "tf/util/Monitor.h"
#include "tf/util/FileUtil.h"
#include "tf/util/CompareUtil.h"
#include "tf/util/Data.h"
#include "tf/util/EvaluationUtil.h"
#include "tf/util/Parameter.h"
#include "tf/util/Calculator.h"

using namespace EvaluationUtil;
using namespace FileUtil;
using namespace Calculator;

class HOLE : virtual public Optimizer {

public:

    HOLE() {};

    HOLE(Parameter *parameter, Data *data) : Optimizer(parameter, data) {}

    ~HOLE() {
        DftiFreeDescriptor(&descriptor);
    }

protected:

    // for evaluation
    vector<DenseMatrix> rescalR;
    DFTI_DESCRIPTOR_HANDLE descriptor;
    DenseMatrix HoleE;
    DenseMatrix HoleP;
    DenseMatrix HoleE_G;
    DenseMatrix HoleP_G;

    void update(Sample &sample, const value_type weight = 1.0) {

        bool subject_replace = (sample.n_sub != sample.p_sub); // true: subject is replaced, false: object is replaced.

        value_type positive_score = cal_hole_score(sample.p_sub, sample.p_obj, sample.relation_id, parameter->hole_D, HoleE,
                                                   HoleP, descriptor);
        value_type negative_score = cal_hole_score(sample.n_sub, sample.n_obj, sample.relation_id, parameter->hole_D, HoleE,
                                                   HoleP, descriptor);

        value_type margin = positive_score - negative_score - parameter->margin;

        // compute the gradient then update the embeddings
        if (margin < 0) {

            violations++;

            value_type *e_ps = HoleE.data().begin() + sample.p_sub * parameter->hole_D;
            value_type *e_po = HoleE.data().begin() + sample.p_obj * parameter->hole_D;
            value_type *r_k = HoleP.data().begin() + sample.relation_id * parameter->hole_D;

            Vec grad_r_p;
            correlation(descriptor, e_ps, e_po, grad_r_p, parameter->hole_D);
            //calculate gradient of sub
            Vec grad_s_p;
            correlation(descriptor, r_k, e_po, grad_s_p, parameter->hole_D);
            //calculate gradient of obj
            Vec grad_o_p;
            convolution(descriptor, r_k, e_ps, grad_o_p, parameter->hole_D);

            value_type weight_p = g_sigmoid(positive_score);
            value_type weight_n = g_sigmoid(negative_score);

            if (subject_replace) {

                value_type *e_ns = HoleE.data().begin() + sample.n_sub * parameter->hole_D;

                //calculate gradient of r
                Vec grad_r_n;
                correlation(descriptor, e_ns, e_po, grad_r_n, parameter->hole_D);
                //calculate gradient of sub
                Vec grad_s_n;
                correlation(descriptor, r_k, e_po, grad_s_n, parameter->hole_D);
                //calculate gradient of obj
                Vec grad_o_n;
                convolution(descriptor, r_k, e_ns, grad_o_n, parameter->hole_D);

//                for (int i = 0; i < parameter->hole_D; i++) {
//                    r_k[i] += weight_p * grad_r_p(i) - weight_n * grad_r_n(i);
//                    e_ps[i] += weight_p * grad_s_p(i);
//                    e_po[i] += weight_p * grad_o_p(i) - weight_n * grad_o_n(i);
//                    e_ns[i] -= weight_n * grad_s_n(i);
//                }

                Vec grad_r_k = weight_p * grad_r_p - weight_n * grad_r_n - parameter->lambdaP * row(HoleP, sample.relation_id);
                Vec grad_e_ps = weight_p * grad_s_p - parameter->lambdaE * row(HoleE, sample.p_sub);
                Vec grad_e_po = weight_p * grad_o_p - weight_n * grad_o_n - parameter->lambdaE * row(HoleE, sample.p_obj);
                Vec grad_e_ns = - weight_n * grad_s_n - parameter->lambdaE * row(HoleE, sample.n_sub);

                (this->*update_grad)(r_k, grad_r_k.data().begin(), HoleP_G.data().begin() + sample.relation_id * parameter->hole_D, parameter->hole_D, weight);
                (this->*update_grad)(e_ps, grad_e_ps.data().begin(), HoleE_G.data().begin() + sample.p_sub * parameter->hole_D, parameter->hole_D, weight);
                (this->*update_grad)(e_po, grad_e_po.data().begin(), HoleE_G.data().begin() + sample.p_obj * parameter->hole_D, parameter->hole_D, weight);
                (this->*update_grad)(e_ns, grad_e_ns.data().begin(), HoleE_G.data().begin() + sample.n_sub * parameter->hole_D, parameter->hole_D, weight);

            } else {

                value_type *e_no = HoleE.data().begin() + sample.n_obj * parameter->hole_D;

                //calculate gradient of r
                Vec grad_r_n;
                correlation(descriptor, e_ps, e_no, grad_r_n, parameter->hole_D);
                //calculate gradient of sub
                Vec grad_s_n;
                correlation(descriptor, r_k, e_no, grad_s_n, parameter->hole_D);
                //calculate gradient of obj
                Vec grad_o_n;
                convolution(descriptor, r_k, e_ps, grad_o_n, parameter->hole_D);

//                for (int i = 0; i < parameter->hole_D; i++) {
//                    r_k[i] += weight_p * grad_r_p(i) - weight_n * grad_r_n(i);
//                    e_ps[i] += weight_p * grad_s_p(i) - weight_n * grad_s_n(i);
//                    e_po[i] += weight_p * grad_o_p(i);
//                    e_no[i] -= weight_n * grad_o_n(i);
//                }

                Vec grad_r_k = weight_p * grad_r_p - weight_n * grad_r_n - parameter->lambdaP * row(HoleP, sample.relation_id);
                Vec grad_e_ps = weight_p * grad_s_p - weight_n * grad_s_n - parameter->lambdaE * row(HoleE, sample.p_sub);
                Vec grad_e_po = weight_p * grad_o_p - parameter->lambdaE * row(HoleE, sample.p_obj);
                Vec grad_e_no = - weight_n * grad_o_n - parameter->lambdaE * row(HoleE, sample.n_obj);

                (this->*update_grad)(r_k, grad_r_k.data().begin(), HoleP_G.data().begin() + sample.relation_id * parameter->hole_D, parameter->hole_D, weight);
                (this->*update_grad)(e_ps, grad_e_ps.data().begin(), HoleE_G.data().begin() + sample.p_sub * parameter->hole_D, parameter->hole_D, weight);
                (this->*update_grad)(e_po, grad_e_po.data().begin(), HoleE_G.data().begin() + sample.p_obj * parameter->hole_D, parameter->hole_D, weight);
                (this->*update_grad)(e_no, grad_e_no.data().begin(), HoleE_G.data().begin() + sample.n_obj * parameter->hole_D, parameter->hole_D, weight);
            }
        }
    }

    void initialize() {

        // for fast evaluation
        parameter->rescal_D = parameter->hole_D;

        RandomUtil::init_seed();

#ifdef use_double
        DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_REAL, 1, parameter->hole_D);
#else
        DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_REAL, 1, parameter->hole_D);
#endif

        value_type scale = 1.0 / parameter->hole_D;
        DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, scale); //Scale down the output
        DftiSetValue(descriptor, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE); //Out of place FFT
        DftiCommitDescriptor(descriptor);

        HoleE.resize(data->N, parameter->hole_D);
        HoleP.resize(data->K, parameter->hole_D);

        // ToDo: why SGD cannot run if min_not_zeros is not initialized
//   if(parameter->optimization=="adagrad" || parameter->optimization=="adadelta"){
        min_not_zeros.resize(parameter->hole_D);
        for (int i = 0; i < parameter->hole_D; i++) {
            min_not_zeros(i) = min_not_zero_value;
        }

        HoleE_G.resize(data->N, parameter->hole_D);
        HoleP_G.resize(data->K, parameter->hole_D);
        HoleE_G.clear();
        HoleP_G.clear();

//   }

        current_epoch = parameter->restore_epoch + 1;

        if (parameter->restore_epoch >= 1) {

            if(parameter->optimization=="sgd") {
                read_transe_matrices(HoleE, HoleP, parameter->hole_D, parameter->restore_epoch, parameter->restore_path);
            } else {
                read_transe_matrices(HoleE, HoleP, HoleE_G, HoleP_G, parameter->hole_D, parameter->restore_epoch, parameter->restore_path);
            }

            // transform factors to rescal based factor and then evaluate to save time
            transform_hole2rescal(HoleP, rescalR);

            Monitor timer;
            timer.start();

            hit_rate measure = eval_hit_rate(Method::m_RESCAL_RANK, parameter, data, &HoleE, &rescalR,
                                             nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                             parameter->output_path + "/HOLE_test_epoch_" + to_string((parameter->restore_epoch)));

            timer.stop();

            string prefix = "[Hole] restore to epoch " + to_string(parameter->restore_epoch) + " ";

            print_hit_rate(prefix, parameter->hit_rate_topk, measure);

            cout << "evaluation time: " << timer.getElapsedTime() << " secs" << endl;

            cout << "------------------------" << endl;

        } else {

            value_type bnd = sqrt(6) / sqrt(data->N + parameter->hole_D);

            for (int row = 0; row < data->N; row++) {
                for (int col = 0; col < parameter->hole_D; col++) {
                    HoleE(row, col) = RandomUtil::uniform_real(-bnd, bnd);
                }
            }

            for (int row = 0; row < data->K; row++) {
                for (int col = 0; col < parameter->hole_D; col++) {
                    HoleP(row, col) = RandomUtil::uniform_real(-bnd, bnd);
                }
            }

        }

    }

    string eval(const int epoch) {

        // transform factors to rescal based factor and then evaluate to save time
        transform_hole2rescal(HoleP, rescalR);

        hit_rate testing_measure = eval_hit_rate(Method::m_RESCAL_RANK, parameter, data, &HoleE, &rescalR,
                                                 nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                 parameter->output_path + "/HOLE_test_epoch_" + to_string(epoch));

//        hit_rate testing_measure = eval_hit_rate(Method::m_HOLE, parameter, data, nullptr, nullptr,
//                                                 nullptr, nullptr, &HoleE, &HoleP, &descriptor, nullptr, nullptr,
//                                                 parameter->output_path + "/HOLE_test_epoch_" + to_string(epoch));

        string prefix = "testing data >>> ";

        print_hit_rate(prefix, parameter->hit_rate_topk, testing_measure);

        pair<value_type, value_type> map;
        map.first = -1;
        map.second = -1;

        if (parameter->eval_map) {
            map = eval_MAP(Method::m_RESCAL_RANK, parameter, data, &HoleE, &rescalR, nullptr, nullptr, nullptr, nullptr, &descriptor, nullptr);
            string prefix = "testing data MAP evalution >>> ";
            print_map(prefix, parameter->num_of_replaced_entities, map);
        }

        string log = "";
        log.append("HOLE,");
        log.append(parameter->optimization + ",");
        log.append(to_string(epoch) + ",");
        log.append(to_string(parameter->hole_D) + ",");
        log.append(to_string(parameter->step_size) + ",");
        log.append(to_string(parameter->margin) + ",");
        log.append(to_string(parameter->lambdaE) + ",");
        log.append(to_string(parameter->lambdaP) + ",");
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

    void output(const int epoch) {
        if(parameter->optimization=="sgd"){
            output_matrices(HoleE, HoleP, data->N, parameter->hole_D, data->K, epoch,
                            parameter->output_path);
        } else {
            output_matrices(HoleE, HoleP, HoleE_G, HoleP_G, data->N, parameter->hole_D, data->K, epoch,
                            parameter->output_path);
        }
    }

    string get_log_header() {
        string header = "Method,Optimization,epoch,Dimension,step size,margin,lambdaE,lambdaP,";
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

};

#endif //HOLE_H