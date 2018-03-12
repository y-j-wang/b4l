#ifndef RESCAL_RANK_H
#define RESCAL_RANK_H

#include "tf/alg/Optimizer.h"
#include "tf/util/Base.h"
#include "tf/util/RandomUtil.h"
#include "tf/util/Monitor.h"
#include "tf/util/FileUtil.h"
#include "tf/util/CompareUtil.h"
#include "tf/util/Data.h"
#include "tf/util/Calculator.h"
#include "tf/util/EvaluationUtil.h"
#include "tf/util/Parameter.h"

using namespace mf;
using namespace EvaluationUtil;
using namespace FileUtil;
using namespace Calculator;

class RESCAL_RANK: virtual public Optimizer {

protected:
    DFTI_DESCRIPTOR_HANDLE descriptor;
    DenseMatrix rescalA;
    vector<DenseMatrix> rescalR;
    DenseMatrix rescalA_G;
    vector<DenseMatrix> rescalR_G;

    value_type cal_loss() {
        return eval_rescal_train(parameter, data, rescalA, rescalR);
    }

    void init_G(const int D) {
        // ToDo: why SGD cannot run if min_not_zeros is not initialized
//        if(parameter->optimization=="adagrad" || parameter->optimization=="adadelta"){
            min_not_zeros.resize(D * D);
            for (int i = 0; i < D * D; i++) {
                min_not_zeros(i) = min_not_zero_value;
            }

            rescalA_G.resize(data->N, D);
            rescalR_G.resize(data->K, DenseMatrix(D, D));
            rescalA_G.clear();
            for (int i = 0; i < data->K; i++) {
                rescalR_G[i].clear();
            }
//        }
    }

    void initialize() {
        RandomUtil::init_seed();

        if (parameter->restore_epoch >= 1 && (!parameter->restore_from_transe) && (!parameter->restore_from_hole)) {

            current_epoch = parameter->restore_epoch + 1;

            rescalA.resize(data->N, parameter->rescal_D);
            rescalR.resize(data->K, DenseMatrix(parameter->rescal_D, parameter->rescal_D));

            init_G(parameter->rescal_D);

            if (parameter->optimization == "sgd") {
                read_rescal_matrices(rescalA, rescalR, parameter->restore_epoch, parameter->restore_path);
            } else {
                read_rescal_rank_matrices(rescalA, rescalR, rescalA_G, rescalR_G, parameter->restore_epoch,
                                          parameter->restore_path);
            }

            Monitor timer;
            timer.start();

            hit_rate measure = eval_hit_rate(Method::m_RESCAL_RANK, parameter, data, &rescalA, &rescalR,
                                             nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, parameter->output_path + "/RESCAL_RANK_test_epoch_" + to_string(parameter->restore_epoch));
            timer.stop();

            string prefix = "[RESCAL_RANK] restore to epoch " + to_string(parameter->restore_epoch) + ", ";

            print_hit_rate(prefix, parameter->hit_rate_topk, measure);

            cout << "evaluation time: " << timer.getElapsedTime() << " secs" << endl;

            cout << "------------------------" << endl;

        } else if (parameter->restore_epoch >= 1 && parameter->restore_from_transe && (!parameter->restore_from_hole)) {

            current_epoch = 1;

            parameter->transe_D = parameter->rescal_D;

            DenseMatrix transeA(data->N, parameter->transe_D);
            DenseMatrix transeR(data->K, parameter->transe_D);

            read_transe_matrices(transeA, transeR, parameter->transe_D, parameter->restore_epoch, parameter->restore_path);

            Monitor timer;

            if(parameter->init_check){

                timer.start();

                hit_rate transe_measure = eval_hit_rate(Method::m_TransE, parameter, data, nullptr, nullptr,
                                                        &transeA, &transeR, nullptr, nullptr, nullptr, nullptr, parameter->output_path + "/TransE_test_epoch_" + to_string(parameter->restore_epoch));
                timer.stop();

                string prefix = "[TransE] restore to epoch " + to_string(parameter->restore_epoch) + " ";

                print_hit_rate(prefix, parameter->hit_rate_topk, transe_measure);

                cout << "evaluation time: " << timer.getElapsedTime() << " secs" << endl;
            }

            parameter->rescal_D = transform_transe2rescal(transeA, transeR, rescalA, rescalR);

            init_G(parameter->rescal_D);

            cout << "Transe Dimensionality: " << parameter->transe_D << ", RESCAL_RANK Dimensionality: " << parameter->rescal_D << endl;

            if(parameter->init_check) {
                timer.start();

                hit_rate measure = eval_hit_rate(Method::m_RESCAL_RANK, parameter, data, &rescalA, &rescalR,
                                                 nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                 parameter->output_path + "/RESCAL_restore_from_transE_epoch_" +
                                                 to_string(parameter->restore_epoch));
                timer.stop();

                string prefix2 = "[RESCAL_RANK] restore from TransE to epoch " + to_string(parameter->restore_epoch) + ", ";

                print_hit_rate(prefix2, parameter->hit_rate_topk, measure);

                cout << "evaluation time: " << timer.getElapsedTime() << " secs" << endl;

                cout << "------------------------" << endl;

            }

            if(parameter->init_check) {

                Sample sample;
                int test_violation_num = 0;
                for (int n = 0; n < data->num_of_training_triples; n++) {

                    Sampler::random_sample(*data, sample, n);

                    value_type positive_score = cal_rescal_score(sample.relation_id, sample.p_sub, sample.p_obj,
                                                                 parameter->rescal_D, rescalA, rescalR);
                    value_type negative_score = cal_rescal_score(sample.relation_id, sample.n_sub, sample.n_obj,
                                                                 parameter->rescal_D, rescalA, rescalR);

                    if (positive_score - parameter->margin - negative_score < 0) {
                        test_violation_num++;
                    }
                }

                cout << "violations: " << test_violation_num << endl;
            }

        } else if (parameter->restore_epoch >= 1 && (!parameter->restore_from_transe) && parameter->restore_from_hole) {

            current_epoch = 1;

            parameter->hole_D = parameter->rescal_D;

            rescalA.resize(data->N, parameter->rescal_D);
            DenseMatrix HoleP(data->K, parameter->rescal_D);

            read_transe_matrices(rescalA, HoleP, parameter->rescal_D, parameter->restore_epoch, parameter->restore_path);

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

            Monitor timer;

            // too slow, but it is correct
//            if(parameter->init_check) {
//                timer.start();
//
//                hit_rate hole_measure = eval_hit_rate(Method::m_HOLE, parameter, data, nullptr, nullptr,
//                                                      nullptr, nullptr, &rescalA, &HoleP, &descriptor, nullptr,
//                                                      parameter->output_path + "/Hole_test_epoch_" +
//                                                      to_string(parameter->restore_epoch));
//                timer.stop();
//
//                string prefix = "[Hole] restore to epoch " + to_string(parameter->restore_epoch) + " ";
//
//                print_hit_rate(prefix, parameter->hit_rate_topk, hole_measure);
//
//                cout << "evaluation time: " << timer.getElapsedTime() << " secs" << endl;
//            }

            init_G(parameter->rescal_D);

            cout << "Hole Dimensionality: " << parameter->hole_D << ", RESCAL_RANK Dimensionality: " << parameter->rescal_D << endl;

            transform_hole2rescal(HoleP, rescalR);

            if(parameter->init_check) {

                timer.start();

                hit_rate measure = eval_hit_rate(Method::m_RESCAL_RANK, parameter, data, &rescalA, &rescalR,
                                                 nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                 parameter->output_path + "/RESCAL_restore_from_hole_epoch_" +
                                                 to_string(parameter->restore_epoch));
                timer.stop();

                string prefix2 = "[RESCAL_RANK] restore from Hole to epoch " + to_string(parameter->restore_epoch) + ", ";

                print_hit_rate(prefix2, parameter->hit_rate_topk, measure);

                cout << "evaluation time: " << timer.getElapsedTime() << " secs" << endl;

                cout << "------------------------" << endl;
            }

            if(parameter->init_check) {
                Sample sample;
                int test_violation_num = 0;
                for (int n = 0; n < data->num_of_training_triples; n++) {

                    Sampler::random_sample(*data, sample, n);

                    value_type positive_score = cal_rescal_score(sample.relation_id, sample.p_sub, sample.p_obj,
                                                                 parameter->rescal_D, rescalA, rescalR);
                    value_type negative_score = cal_rescal_score(sample.relation_id, sample.n_sub, sample.n_obj,
                                                                 parameter->rescal_D, rescalA, rescalR);

                    positive_score = sigmoid(positive_score);
                    negative_score = sigmoid(negative_score);

                    if (positive_score - negative_score < parameter->margin) {
                        test_violation_num++;
                    }

                }

                cout << "violations: " << test_violation_num << endl;
            }
        } else {

            current_epoch = 1;

            rescalA.resize(data->N, parameter->rescal_D);
            rescalR.resize(data->K, DenseMatrix(parameter->rescal_D, parameter->rescal_D));
            init_G(parameter->rescal_D);

            value_type bnd = sqrt(6) / sqrt(data->N + parameter->rescal_D);

            for (int row = 0; row < data->N; row++) {
                for (int col = 0; col < parameter->rescal_D; col++) {
                    rescalA(row, col) = RandomUtil::uniform_real(-bnd, bnd);
                }
            }

            bnd = sqrt(6) / sqrt(parameter->rescal_D + parameter->rescal_D);

            for (int R_i = 0; R_i < data->K; R_i++) {
                DenseMatrix &sub_R = rescalR[R_i];
                for (int row = 0; row < parameter->rescal_D; row++) {
                    for (int col = 0; col < parameter->rescal_D; col++) {
                        sub_R(row, col) = RandomUtil::uniform_real(-bnd, bnd);
                    }
                }
            }

        }
    }

    void update(Sample &sample, const value_type weight = 1.0) {

        value_type positive_score = cal_rescal_score(sample.relation_id, sample.p_sub, sample.p_obj, parameter->rescal_D, rescalA, rescalR);
        value_type negative_score = cal_rescal_score(sample.relation_id, sample.n_sub, sample.n_obj, parameter->rescal_D, rescalA, rescalR);

        value_type p_pre = 1;
        value_type n_pre = 1;

        // ToDo: ???
//        if (parameter->restore_from_hole) {
//            positive_score = sigmoid(positive_score);
//            negative_score = sigmoid(negative_score);
//            p_pre = g_sigmoid(positive_score);
//            n_pre = g_sigmoid(negative_score);
//        }

        if(positive_score - negative_score >= parameter->margin) {
            return;
        }

        violations++;

        DenseMatrix grad4R(parameter->rescal_D, parameter->rescal_D);
        unordered_map<int, Vec> grad4A_map;

        // Step 1: compute gradient descent

        update_4_R(sample, grad4R, p_pre, n_pre, weight);
        update_4_A(sample, grad4A_map, p_pre, n_pre, weight);

        // Step 2: do the update
        (this->*update_grad)(rescalR[sample.relation_id].data().begin(), grad4R.data().begin(), rescalR_G[sample.relation_id].data().begin(),
           parameter->rescal_D * parameter->rescal_D , weight);

        for (auto ptr = grad4A_map.begin(); ptr != grad4A_map.end(); ptr++) {
#ifdef use_mkl
            Vec A_grad(parameter->rescal_D);
            cblas_xcopy(parameter->rescal_D, ptr->second.data().begin(), 1, A_grad.data().begin(), 1);
            cblas_xaxpy(parameter->rescal_D, - parameter->lambdaA, rescalA.data().begin()+ptr->first * parameter->rescal_D, 1, A_grad.data().begin(), 1);
#else
            Vec A_grad = ptr->second - parameter->lambdaA * row(rescalA, ptr->first);
#endif

            (this->*update_grad)(rescalA.data().begin() + parameter->rescal_D * ptr->first, A_grad.data().begin(), rescalA_G.data().begin() + parameter->rescal_D * ptr->first,
                                 parameter->rescal_D, weight);
        }

    }

    string eval(const int epoch){

//        if (parameter->eval_train) {
//
//            hit_rate train_measure = eval_rescal_train(parameter, data, rescalA, rescalR, parameter->output_path + "/RESCAL_RANK_train_epoch_" + to_string(epoch));
//
//            string prefix = "sampled training data >>> ";
//
//            print_hit_rate_train(prefix, parameter->hit_rate_topk, train_measure);
//
//        }

        hit_rate testing_measure = eval_hit_rate(Method::m_RESCAL_RANK, parameter, data, &rescalA, &rescalR,
                                                 nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, parameter->output_path + "/RESCAL_RANK_test_epoch_" + to_string(epoch));

        string prefix = "testing data >>> ";

        print_hit_rate(prefix, parameter->hit_rate_topk, testing_measure);

        if (parameter->eval_rel) {

            hit_rate rel_measure = eval_relation_rescal(parameter, data, rescalA, rescalR, parameter->output_path + "/RESCAL_RANK_test_epoch_" + to_string(epoch));

            string prefix = "testing data relation evalution >>> ";

            print_hit_rate_rel(prefix, parameter->hit_rate_topk, rel_measure);
        }

        pair<value_type, value_type> map;
        map.first = -1;
        map.second = -1;

        if (parameter->eval_map) {
            map = eval_MAP(m_RESCAL_RANK, parameter, data, &rescalA, &rescalR, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
            string prefix = "testing data MAP evalution >>> ";
            print_map(prefix, parameter->num_of_replaced_entities, map);
        }

        string log = "";
        log.append("RESCAL_RANK,");
        log.append(parameter->optimization + ",");
        log.append(to_string(epoch) + ",");
        log.append(to_string(parameter->rescal_D) + ",");
        log.append(to_string(parameter->step_size) + ",");
        log.append(to_string(parameter->margin) + ",");
        log.append(to_string(parameter->lambdaA) + ",");
        log.append(to_string(parameter->lambdaR) + ",");
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
            output_matrices(rescalA, rescalR, epoch, parameter->output_path);
        }else{
            output_matrices(rescalA, rescalR, rescalA_G, rescalR_G, epoch, parameter->output_path);
        }

    }

    string get_log_header() {

        string header = "Method,Optimization,epoch,Dimension,step size,margin,lambdaA,lambdaR,";
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


private:

    void update_4_A(Sample &sample, unordered_map<int, Vec> &grad4A_map, const value_type p_pre, const value_type n_pre, const value_type weight) {

        DenseMatrix &R_k = rescalR[sample.relation_id];

#ifdef use_mkl
        Vec p_tmp1(parameter->rescal_D);
        cblas_xgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    parameter->rescal_D, 1, parameter->rescal_D, 1.0, R_k.data().begin(), parameter->rescal_D,
                    rescalA.data().begin() + sample.p_obj * parameter->rescal_D, 1, 0.0, p_tmp1.data().begin(), 1);

        Vec p_tmp2(parameter->rescal_D);
        cblas_xgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    1, parameter->rescal_D, parameter->rescal_D, 1.0,
                    rescalA.data().begin() + sample.p_sub * parameter->rescal_D, parameter->rescal_D,
                    R_k.data().begin(), parameter->rescal_D, 0.0, p_tmp2.data().begin(), parameter->rescal_D);

        Vec n_tmp1(parameter->rescal_D);
        cblas_xgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    parameter->rescal_D, 1, parameter->rescal_D, 1.0, R_k.data().begin(), parameter->rescal_D,
                    rescalA.data().begin() + sample.n_obj * parameter->rescal_D, 1, 0.0, n_tmp1.data().begin(), 1);

        Vec n_tmp2(parameter->rescal_D);
        cblas_xgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    1, parameter->rescal_D, parameter->rescal_D, 1.0,
                    rescalA.data().begin() + sample.n_sub * parameter->rescal_D, parameter->rescal_D,
                    R_k.data().begin(), parameter->rescal_D, 0.0, n_tmp2.data().begin(), parameter->rescal_D);

#else
        Vec p_tmp1 = prod(R_k, row(rescalA, sample.p_obj));
        Vec p_tmp2 = prod(row(rescalA, sample.p_sub), R_k);
        Vec n_tmp1 = prod(R_k, row(rescalA, sample.n_obj));
        Vec n_tmp2 = prod(row(rescalA, sample.n_sub), R_k);
#endif

        grad4A_map[sample.p_sub] = p_tmp1;

        unordered_map<int, Vec>::iterator ptr = grad4A_map.find(sample.p_obj);
        if (ptr == grad4A_map.end()) {
            grad4A_map[sample.p_obj] = p_pre * p_tmp2;
        } else {
#ifdef use_mkl
            cblas_xaxpy(parameter->rescal_D, p_pre, p_tmp2.data().begin(), 1, grad4A_map[sample.p_obj].data().begin(), 1);
//            vxAdd(parameter->rescal_D, grad4A_map[sample.p_obj].data().begin(), p_tmp2.data().begin(), grad4A_map[sample.p_obj].data().begin());
#else
            grad4A_map[sample.p_obj] += p_pre * p_tmp2;
#endif
        }

        ptr = grad4A_map.find(sample.n_sub);
        if (ptr == grad4A_map.end()) {
            grad4A_map[sample.n_sub] = n_pre * (-n_tmp1);
        } else {

#ifdef use_mkl
            cblas_xaxpy(parameter->rescal_D, - n_pre, n_tmp1.data().begin(), 1, grad4A_map[sample.n_sub].data().begin(), 1);
//            vxSub(parameter->rescal_D, grad4A_map[sample.n_sub].data().begin(), n_tmp1.data().begin(), grad4A_map[sample.n_sub].data().begin());
#else
            grad4A_map[sample.n_sub] += n_pre * (-n_tmp1);
#endif
        }

        ptr = grad4A_map.find(sample.n_obj);
        if (ptr == grad4A_map.end()) {
            grad4A_map[sample.n_obj] = n_pre * (-n_tmp2);
        } else {

#ifdef use_mkl
            cblas_xaxpy(parameter->rescal_D, - n_pre, n_tmp2.data().begin(), 1, grad4A_map[sample.n_obj].data().begin(), 1);
//            vxSub(parameter->rescal_D, grad4A_map[sample.n_obj].data().begin(), n_tmp2.data().begin(), grad4A_map[sample.n_obj].data().begin());
#else
            grad4A_map[sample.n_obj] += n_pre * (-n_tmp2);
#endif
        }
    }

    void update_4_R(Sample &sample, DenseMatrix &grad4R, const value_type p_pre, const value_type n_pre, const value_type weight) {

        value_type *p_sub = rescalA.data().begin() + sample.p_sub * parameter->rescal_D;
        value_type *p_obj = rescalA.data().begin() + sample.p_obj * parameter->rescal_D;

        value_type *n_sub = rescalA.data().begin() + sample.n_sub * parameter->rescal_D;
        value_type *n_obj = rescalA.data().begin() + sample.n_obj * parameter->rescal_D;

#ifdef use_mkl
        cblas_xgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    parameter->rescal_D, parameter->rescal_D, 1, p_pre, p_sub, 1, p_obj, parameter->rescal_D, 0.0, grad4R.data().begin(), parameter->rescal_D);

        cblas_xgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    parameter->rescal_D, parameter->rescal_D, 1, - n_pre, n_sub, 1, n_obj, parameter->rescal_D, 1.0, grad4R.data().begin(), parameter->rescal_D);

        cblas_xaxpy(parameter->rescal_D *  parameter->rescal_D, - parameter->lambdaR, rescalR[sample.relation_id].data().begin(), 1, grad4R.data().begin(), 1);
#else

        grad4R.clear();

        for (int i = 0; i < parameter->rescal_D; i++) {
            for (int j = 0; j < parameter->rescal_D; j++) {
                grad4R(i, j) += p_pre * p_sub[i] * p_obj[j] - n_pre * n_sub[i] * n_obj[j];
            }
        }

        grad4R += - parameter->lambdaR * rescalR[sample.relation_id];
#endif
    }

public:

    RESCAL_RANK(){};
    RESCAL_RANK(Parameter *parameter, Data *data) : Optimizer(parameter, data) {}
    ~RESCAL_RANK() {
        if(parameter->restore_from_hole) {
            DftiFreeDescriptor(&descriptor);
        }
    }
};

#endif //RESCAL_RANK_H
