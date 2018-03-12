#ifndef RESCAL_H
#define RESCAL_H

#include "tf/util/Base.h"
#include "tf/util/Calculator.h"
#include "tf/util/FileUtil.h"
#include "tf/util/RandomUtil.h"
#include "tf/util/Calculator.h"
#include "tf/util/Monitor.h"
#include "tf/util/Data.h"
#include "tf/util/CompareUtil.h"
#include "tf/util/ThreadUtil.h"
#include "tf/util/EvaluationUtil.h"
#include "tf/util/Parameter.h"

using namespace EvaluationUtil;
using namespace FileUtil;
using namespace Calculator;

class RESCAL {

private:

    Parameter *parameter = nullptr;
    Data *data = nullptr;
    int current_epoch;
    value_type sumNorm;

    DenseMatrix lambdaR_matrix;
    DenseMatrix lambdaA_matrix;

    // updateA
    DenseMatrix F;
    DenseMatrix E;
    DenseMatrix AtA;

    // updateR
    DenseMatrix Vt;
    DenseMatrix U;
    DenseMatrix Shat;

    value_type *S_array = nullptr;
    value_type *superb = nullptr;

#ifdef use_mkl
    DenseMatrix tmp_ND;
    DenseMatrix tmp_ND2;
    DenseMatrix tmp_DD;
    DenseMatrix tmp_DD2;
    DenseMatrix tmp_KD;
    DenseMatrix tmp_DN;
#endif

    // compute fit
    DenseMatrix ARAt;

    inline void init() {

        RandomUtil::init_seed();

        A.resize(data->N, parameter->rescal_D);
        R.resize(data->K, DenseMatrix(parameter->rescal_D, parameter->rescal_D));

        Vt.resize(parameter->rescal_D, parameter->rescal_D);
        U.resize(data->N, parameter->rescal_D);
        Shat.resize(parameter->rescal_D, parameter->rescal_D);

        S_array = new value_type[parameter->rescal_D];
        superb = new value_type[parameter->rescal_D - 1];

        lambdaA_matrix.resize(parameter->rescal_D, parameter->rescal_D);
        lambdaA_matrix.clear();
        lambdaR_matrix.resize(parameter->rescal_D, parameter->rescal_D);
        lambdaR_matrix.clear();

        for (int i = 0; i < parameter->rescal_D; i++) {
            for (int j = 0; j < parameter->rescal_D; j++) {
                if (j == i) {
                    lambdaA_matrix(i, j) = parameter->lambdaA;
                }
                lambdaR_matrix(i, j) = parameter->lambdaR;
            }
        }

        F.resize(data->N, parameter->rescal_D);
        E.resize(parameter->rescal_D, parameter->rescal_D);
        AtA.resize(parameter->rescal_D, parameter->rescal_D);

#ifdef use_mkl
        tmp_ND.resize(data->N, parameter->rescal_D);
        tmp_ND2.resize(data->N, parameter->rescal_D);
        tmp_DD.resize(parameter->rescal_D, parameter->rescal_D);
        tmp_DD2.resize(parameter->rescal_D, parameter->rescal_D);
        tmp_KD.resize(data->K, parameter->rescal_D);
        tmp_DN.resize(parameter->rescal_D, data->N);
#endif

        ARAt.resize(data->N, data->N);

        sumNorm = 0.0;
        value_type normX;

        for (int i = 0; i < data->K; i++) {
            sumNorm += data->relation2tupleList_mapping[i].size();
        }

    }

    inline void init_dense_matrix(bool nvecs=false) {

        current_epoch = parameter->restore_epoch + 1;

        if (parameter->restore_epoch >= 1){

            read_rescal_matrices(A, R, parameter->restore_epoch, parameter->restore_path);
            value_type obj = cal_obj();

            Monitor timer;
            timer.start();

            hit_rate measure = eval_hit_rate(Method::m_RESCAL, parameter, data, &A, &R,
                                             nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                             parameter->output_path + "/RESCAL_test_epoch_" +
                                             to_string(parameter->restore_epoch));
            timer.stop();

            string prefix = "restore to epoch " + to_string(parameter->restore_epoch) + ", obj: " + to_string(obj) + ", ";

            print_hit_rate(prefix, parameter->hit_rate_topk, measure);

            cout << "evaluation time: " << timer.getElapsedTime() << " secs" << endl;

            return;

        } else {
            if (nvecs) {

                DenseMatrix *S = new DenseMatrix(data->N, data->N);
                S->clear();

#ifdef use_mkl
                for (int i = 0; i < data->relation2tupleList_mapping.size(); i++) {

                    int nnz = data->relation2tupleList_mapping[i].size();
                    int workload = nnz / parameter->num_of_thread + (nnz % parameter->num_of_thread == 0) ? 0 : 1;

                    std::function<void(int)> add_func = [&](int thread_index) -> void {
                        int start = thread_index * workload;
                        int end = std::min(thread_index * workload + workload, nnz);
                        int current_row = 0;
                        int end_index_current_row = data->X_pointer[i][1];
                        for (int rating_index = start; rating_index < end; rating_index++) {
                            int col = data->X_columns[i][rating_index];
                            if (rating_index > end_index_current_row) {
                                current_row++;
                                end_index_current_row = data->X_pointer[i][current_row + 1];
                            }
                            (*S)(current_row, col) += 1;
                            (*S)(col, current_row) += 1;
                        }
                    };

                    ThreadUtil::execute_threads(add_func, parameter->num_of_thread);

                }

#else
                for (int i = 0; i < data->X.size(); i++) {
                    (*S) += data->X[i];
                    (*S) += data->XT[i];
                }
#endif

                value_type abstol = LAPACKE_xlamch('S');
                int found_num;
                value_type *w = new value_type[data->N];
                value_type *z = new value_type[data->N*data->N];
                int *isupp = new int[2 * data->N];

                LAPACKE_xsyevr(LAPACK_ROW_MAJOR, 'V', 'A', 'U', data->N, S->data().begin(), data->N,
                               0, 0, 0, 0, abstol, &found_num, w, z, data->N, isupp);

                for (int row = 0; row < data->N; row++) {
                    std::copy(z + row * data->N, z + row * data->N + parameter->rescal_D, A.data().begin() + row * data->N);
                }

                S->clear();
                delete S;
                delete[] w;
                delete[] z;
                delete[] isupp;


            } else {

                for (int row = 0; row < data->N; row++) {
                    for (int col = 0; col < parameter->rescal_D; col++) {
                        A(row, col) = RandomUtil::uniform_real();
                    }
                }

                for (int R_i = 0; R_i < data->K; R_i++) {
                    DenseMatrix &sub_R = R[R_i];
                    for (int i = 0; i < parameter->rescal_D; i++) {
                        for (int j = 0; j < parameter->rescal_D; j++) {
                            sub_R(i, j) = RandomUtil::uniform_real();
                        }
                    }
                }

            }
        }

    }

    void updateR() {

        Vt.clear();
        U.clear();

        DenseMatrix tmp_A = A;

        int info = LAPACKE_xgesvd(LAPACK_ROW_MAJOR, 'S', 'S', data->N, parameter->rescal_D, tmp_A.data().begin(), parameter->rescal_D,
                                  S_array, U.data().begin(), parameter->rescal_D, Vt.data().begin(), parameter->rescal_D, superb);

        if (info > 0) {
            cerr << "The algorithm computing SVD failed to converge!" << endl;
            exit(1);
        }

        DenseMatrix V = trans(Vt);
        DenseMatrix Ut = trans(U);

        Shat.clear();
        kronecker_product(S_array, S_array, Shat, parameter->rescal_D);

#ifdef use_mkl
        vxMul(parameter->rescal_D * parameter->rescal_D, Shat.data().begin(), Shat.data().begin(), tmp_DD.data().begin());
        vxAdd(parameter->rescal_D * parameter->rescal_D, tmp_DD.data().begin(), lambdaR_matrix.data().begin(), tmp_DD.data().begin());
        vxDiv(parameter->rescal_D * parameter->rescal_D, Shat.data().begin(), tmp_DD.data().begin(), Shat.data().begin());

        char trans = 'N';
        char matdescra[6] = {'g', 'x', 'x', 'c', 'x', 'x'};
        value_type alpha = 1.0;
        value_type beta = 0.0;

#else
        Shat = element_div(Shat, element_prod(Shat, Shat) + lambdaR_matrix);
#endif

        R.clear();
        R.resize(data->K);

        for (int i = 0; i < data->K; i++) {

            DenseMatrix &Ri = R[i];
            Ri.resize(parameter->rescal_D, parameter->rescal_D);

#ifdef use_mkl

            mkl_xcsrmm(&trans, &data->N, &parameter->rescal_D, &data->N, &alpha, matdescra, data->X_values[i], data->X_columns[i], data->X_pointer[i],
                       data->X_pointer[i] + 1, U.data().begin(), &parameter->rescal_D, &beta, tmp_ND.data().begin(), &parameter->rescal_D);

            cblas_xgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        parameter->rescal_D, parameter->rescal_D, data->N, 1.0, Ut.data().begin(), data->N, tmp_ND.data().begin(), parameter->rescal_D, 0.0, tmp_DD.data().begin(), parameter->rescal_D);

            vxMul(parameter->rescal_D * parameter->rescal_D, Shat.data().begin(), tmp_DD.data().begin(), Ri.data().begin());

            cblas_xgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        parameter->rescal_D, parameter->rescal_D, parameter->rescal_D, 1.0, Ri.data().begin(), parameter->rescal_D, Vt.data().begin(), parameter->rescal_D, 0.0, tmp_DD.data().begin(), parameter->rescal_D);

            cblas_xgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        parameter->rescal_D, parameter->rescal_D, parameter->rescal_D, 1.0, V.data().begin(), parameter->rescal_D, tmp_DD.data().begin(), parameter->rescal_D, 0.0, Ri.data().begin(), parameter->rescal_D);

#else
            DenseMatrix Xi(data->X[i]);
            // Q6 in http://www.boost.org/doc/libs/1_63_0/libs/numeric/ublas/doc/index.html
            Ri = element_prod(Shat, prod(Ut, DenseMatrix(prod(Xi, U))));
            Ri = prod(V, DenseMatrix(prod(Ri, Vt)));
#endif

        }
    }

    void updateA() {

        F.clear();
        E.clear();

#ifdef use_mkl
        cblas_xgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    parameter->rescal_D, parameter->rescal_D, data->N, 1.0, A.data().begin(), parameter->rescal_D, A.data().begin(), parameter->rescal_D, 0.0, AtA.data().begin(), parameter->rescal_D);

        char trans1 = 'N';
        char trans2 = 'T';
        char matdescra[6] = {'g', 'x', 'x', 'c', 'x', 'x'};
        value_type alpha = 1.0;
        value_type beta = 0.0;

#else
        DenseMatrix AtA = prod(trans(A), A);
#endif

        for (int i = 0; i < data->K; i++) {

            DenseMatrix &Ri = R[i];

#ifdef use_mkl

            // F += prod(X[i], DenseMatrix(prod(A, Rt))) + prod(XT[i], DenseMatrix(prod(A, Ri)));
            cblas_xgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        data->N, parameter->rescal_D, parameter->rescal_D, 1.0, A.data().begin(), parameter->rescal_D, Ri.data().begin(), parameter->rescal_D, 0.0, tmp_ND.data().begin(), parameter->rescal_D);

            mkl_xcsrmm(&trans1, &data->N, &parameter->rescal_D, &data->N, &alpha, matdescra, data->X_values[i], data->X_columns[i], data->X_pointer[i],
                       data->X_pointer[i] + 1, tmp_ND.data().begin(), &parameter->rescal_D, &beta, tmp_ND2.data().begin(), &parameter->rescal_D);

            vxAdd(data->N * parameter->rescal_D, F.data().begin(), tmp_ND2.data().begin(), F.data().begin());

            cblas_xgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        data->N, parameter->rescal_D, parameter->rescal_D, 1.0, A.data().begin(), parameter->rescal_D, Ri.data().begin(), parameter->rescal_D, 0.0, tmp_ND.data().begin(), parameter->rescal_D);

            mkl_xcsrmm(&trans2, &data->N, &parameter->rescal_D, &data->N, &alpha, matdescra, data->X_values[i], data->X_columns[i], data->X_pointer[i],
                       data->X_pointer[i] + 1, tmp_ND.data().begin(), &parameter->rescal_D, &beta, tmp_ND2.data().begin(), &parameter->rescal_D);

            vxAdd(data->N * parameter->rescal_D, F.data().begin(), tmp_ND2.data().begin(), F.data().begin());

            // E += prod(R[i], DenseMatrix(prod(AtA, Rt))) + prod(Rt, DenseMatrix(prod(AtA, Ri)));
            cblas_xgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        parameter->rescal_D, parameter->rescal_D, parameter->rescal_D, 1.0, AtA.data().begin(), parameter->rescal_D, Ri.data().begin(), parameter->rescal_D, 0.0, tmp_DD.data().begin(), parameter->rescal_D);

            cblas_xgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        parameter->rescal_D, parameter->rescal_D, parameter->rescal_D, 1.0, Ri.data().begin(), parameter->rescal_D, tmp_DD.data().begin(), parameter->rescal_D, 0.0, tmp_DD2.data().begin(), parameter->rescal_D);

            vxAdd(parameter->rescal_D * parameter->rescal_D, E.data().begin(), tmp_DD2.data().begin(), E.data().begin());

            cblas_xgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        parameter->rescal_D, parameter->rescal_D, parameter->rescal_D, 1.0, AtA.data().begin(), parameter->rescal_D, Ri.data().begin(), parameter->rescal_D, 0.0, tmp_DD.data().begin(), parameter->rescal_D);

            cblas_xgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        parameter->rescal_D, parameter->rescal_D, parameter->rescal_D, 1.0, Ri.data().begin(), parameter->rescal_D, tmp_DD.data().begin(), parameter->rescal_D, 0.0, tmp_DD2.data().begin(), parameter->rescal_D);

            vxAdd(parameter->rescal_D * parameter->rescal_D, E.data().begin(), tmp_DD2.data().begin(), E.data().begin());
#else

            DenseMatrix Xi(data->X[i]);
            DenseMatrix XiT = trans(Xi);

            DenseMatrix Rt = trans(Ri);

            F += prod(Xi, DenseMatrix(prod(A, Rt))) + prod(XiT, DenseMatrix(prod(A, Ri)));
            E += prod(R[i], DenseMatrix(prod(AtA, Rt))) + prod(Rt, DenseMatrix(prod(AtA, Ri)));
#endif


        }

        int *ipiv = new int[parameter->rescal_D];

        DenseMatrix Ft = trans(F);
        DenseMatrix Et = trans(E);

#ifdef use_mkl
        DenseMatrix tmp(parameter->rescal_D, parameter->rescal_D);
        vxAdd(parameter->rescal_D * parameter->rescal_D, lambdaA_matrix.data().begin(), Et.data().begin(), tmp.data().begin());
#else
        DenseMatrix tmp = lambdaA_matrix + Et;
#endif

        int info = LAPACKE_xgesv(LAPACK_ROW_MAJOR, parameter->rescal_D, data->N,
                                 tmp.data().begin(), parameter->rescal_D, ipiv,
                                 Ft.data().begin(),
                                 data->N);

        /* Check for the exact singularity */
        if (info > 0) {
            cerr << "the solution could not be computed." << endl;
            delete[] ipiv;
            exit(1);
        }

        delete[] ipiv;
        A = DenseMatrix(trans(Ft));
    }

    // Compute fit for full slices
    value_type cal_obj() {
        value_type f = 0;
        value_type tmp;

#ifdef use_mkl
#else
        DenseMatrix At = DenseMatrix(trans(A));
#endif

        for (int i = 0; i < data->K; i++) {

#ifdef use_mkl
            cblas_xgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        parameter->rescal_D, data->N, parameter->rescal_D, 1.0, R[i].data().begin(), parameter->rescal_D, A.data().begin(), parameter->rescal_D, 0.0, tmp_DN.data().begin(), data->N);

            cblas_xgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        data->N, data->N, parameter->rescal_D, 1.0, A.data().begin(), parameter->rescal_D, tmp_DN.data().begin(), data->N, 0.0, ARAt.data().begin(), data->N);

            // ToDo:: I do not find a way to subtract a dense matrix from a sparse matrix in MKL
//            vxSub(data->N * data->N, Xi.data().begin(), ARAt.data().begin(), ARAt.data().begin());

            int nnz = data->relation2tupleList_mapping[i].size();
            int workload = nnz / parameter->num_of_thread + (nnz % parameter->num_of_thread == 0) ? 0 : 1;

            std::function<void(int)> subtract_func = [&](int thread_index) -> void {
                int start = thread_index * workload;
                int end = std::min(thread_index * workload + workload, nnz);
                int current_row = 0;
                int end_index_current_row = data->X_pointer[i][1];
                for (int rating_index = start; rating_index < end; rating_index++) {
                    int col = data->X_columns[i][rating_index];
                    if (rating_index > end_index_current_row) {
                        current_row++;
                        end_index_current_row = data->X_pointer[i][current_row + 1];
                    }
                    ARAt(current_row, col) = 1.0 - ARAt(current_row, col);
                }
            };

            ThreadUtil::execute_threads(subtract_func, parameter->num_of_thread);

            tmp = cblas_xnrm2(data->N * data->N, ARAt.data().begin(), 1);

#else
            DenseMatrix Xi(data->X[i]);
            ARAt = DenseMatrix(prod(A, DenseMatrix(prod(R[i], At))));
            tmp = norm_frobenius(DenseMatrix(Xi - ARAt));
#endif

            f += tmp * tmp;
        }

        cout << f << ", " << sumNorm << endl;
        return 1.0 - f / sumNorm;
    }

    string get_log_header() {
        return "Method,Optimization,epoch,Dimension,margin,lambdaA,lambdaR,hit_rate_subject@" +
               to_string(parameter->hit_rate_topk) + ",hit_rate_object@" +
               to_string(parameter->hit_rate_topk) +
               ",subject_ranking,object_ranking,hit_rate_subject_filter@" +
               to_string(parameter->hit_rate_topk) + ",hit_rate_object_filter@" +
               to_string(parameter->hit_rate_topk) +
               ",subject_ranking_filter,object_ranking_filter,MAP_subject@" +
               to_string(parameter->num_of_replaced_entities) + ",MAP_object@" +
               to_string(parameter->num_of_replaced_entities) + ",MRR_subject,MRR_object,MRR_subject_filter,MRR_object_filter";
    }

public:

    DenseMatrix A;
    vector<DenseMatrix> R;

    ~RESCAL(){
        delete[] S_array;
        delete[] superb;
    }

    RESCAL(Parameter *parameter, Data *data) : parameter(parameter), data(data) {}

    value_type als() {

        init();

        // initialize A
        init_dense_matrix();

        // initialize R
        updateR();

        // compute factorization
        Monitor timer;
        value_type fit = 0;
        value_type fitold = 0;
        value_type fitchange = 0;
        value_type f = 0;

        ofstream log_file(parameter->log_path.c_str(), std::ofstream::out | std::ofstream::app);
        log_file << get_log_header() << endl;
        log_file.close();

        for (int e = current_epoch; e < parameter->epoch; e++) {

            timer.start();

            fitold = fit;

            updateA();

            updateR();

            if (parameter->compute_fit) {
                fit = cal_obj();
            } else {
                fit = e;
            }

            fitchange = fabs(fitold - fit);

            timer.stop();

            cout << boost::format{"[%3d] fit: %0.5f | delta: %7.1e | secs: %.5f"} % e %
                    fit % fitchange % timer.getElapsedTime() << endl;

            if ((e + 1) % parameter->print_epoch == 0) {

                timer.start();

                if(parameter->eval_train) {
                    hit_rate train_measure = eval_rescal_train(parameter, data, A, R, parameter->output_path + "/RESCAL_train_epoch_" + to_string(e));

                    string prefix = "sampled training data >>> ";

                    print_hit_rate_train(prefix, parameter->hit_rate_topk, train_measure);
                }

                hit_rate testing_measure = eval_hit_rate(Method::m_RESCAL, parameter, data, &A, &R,
                                                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, parameter->output_path + "/RESCAL_test_epoch_" + to_string(e));
                string prefix = "testing data >>> ";

                print_hit_rate(prefix, parameter->hit_rate_topk, testing_measure);

                if(parameter->eval_rel){

                    hit_rate rel_measure = eval_relation_rescal(parameter, data, A, R, parameter->output_path + "/RESCAL_test_epoch_" + to_string(e));

                    string prefix = "testing data relation evalution >>> ";

                    print_hit_rate_rel(prefix, parameter->hit_rate_topk, rel_measure);
                }

                pair<value_type, value_type> map;
                map.first = -1;
                map.second = -1;

                if (parameter->eval_map) {
                    map = eval_MAP(m_RESCAL, parameter, data, &A, &R, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
                    string prefix = "testing data MAP evalution >>> ";
                    print_map(prefix, parameter->num_of_replaced_entities, map);
                }

                timer.stop();

                cout << "evaluation time: " << timer.getElapsedTime() << " secs"<< endl;

                string log = "";
                log.append("RESCAL,");
                log.append(parameter->optimization + ",");
                log.append(to_string(e) + ",");
                log.append(to_string(parameter->rescal_D) + ",");
                log.append(to_string(parameter->margin) + ",");
                log.append(to_string(parameter->lambdaA) + ",");
                log.append(to_string(parameter->lambdaR) + ",");
                log.append(to_string(parameter->hit_rate_topk) + ",");
                log.append(to_string(parameter->num_of_replaced_entities) + ",");

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

                ofstream log_file(parameter->log_path.c_str(), std::ofstream::out | std::ofstream::app);
                log_file << log << endl;
                log_file.close();
            }

            if ((e + 1) % parameter->output_epoch == 0) {
                output_matrices(A, R, e, parameter->output_path);
            }

        }

    }

};

#endif //RESCAL_H