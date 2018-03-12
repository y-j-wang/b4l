#ifndef CALCULATOR_H
#define CALCULATOR_H

#include "tf/util/Base.h"
#include "tf/util/Parameter.h"
#include "tf/struct/Weight.h"
#include "tf/struct/Min_Max.h"

namespace Calculator {

    inline void cal_mean_std(vector<value_type > &scores, value_type &mean, value_type &std){
        value_type sum = std::accumulate(std::begin(scores), std::end(scores), 0.0);
        mean = sum / scores.size();

        std = 0.0;
        std::for_each (std::begin(scores), std::end(scores), [&](const double d) {
            std += (d - mean) * (d - mean);
        });

        std = sqrt(std / scores.size());
    }

    inline value_type vec_len(vector<value_type> &a) {
        value_type res = 0;
        for (int i = 0; i < a.size(); i++) {
            res += a[i] * a[i];
        }
        res = sqrt(res);
        return res;
    }

    inline value_type vec_len(value_type *a, const int size) {
        value_type vec_norm = 0;
        for (int i = 0; i < size; i++) {
            vec_norm += a[i] * a[i];
        }
        vec_norm = sqrt(vec_norm);
        return vec_norm;
    }

    inline void normalize(value_type *a, const int size) {
        value_type vec_norm = vec_len(a, size);
        for (int i = 0; i < size; i++) {
            a[i] /= vec_norm;
        }
    }

    // only normalize those with len less than one
    inline void normalizeOne(value_type *a, const int size) {
        value_type vec_norm = vec_len(a, size);
        if(vec_norm > 1) {
            for (int i = 0; i < size; i++) {
                a[i] /= vec_norm;
            }
        }
    }

    inline value_type inner_product(value_type *A, value_type *B, const int size) {
        value_type result = 0;
        for (int i = 0; i < size; i++) {
            result += A[i] * B[i];
        }
        return result;
    }

    inline void array_sqrt(value_type *A, const int size) {
        for (int i = 0; i < size; i++) {
            A[i] = sqrt(A[i]);
        }
    }

    inline value_type sqr(value_type x) {
        return x * x;
    }

    inline value_type cal_transe_score(int e1, int e2, int rel, const int D, const bool L1_flag, DenseMatrix &transeA, DenseMatrix &transeR) {

        value_type *transE_entity_vec = transeA.data().begin();
        value_type *transE_rel_vec = transeR.data().begin();

        value_type sum = 0;
        if (L1_flag) {
            for (int i = 0; i < D; i++) {
                sum += fabs(transE_entity_vec[e2 * D + i] - transE_entity_vec[e1 * D + i] - transE_rel_vec[rel * D + i]);
            }
        } else {
            for (int i = 0; i < D; i++) {
                sum += sqr(transE_entity_vec[e2 * D + i] - transE_entity_vec[e1 * D + i] - transE_rel_vec[rel * D + i]);
            }
        }
        return sum;
    }

    inline value_type cal_rescal_score(const int rel_id, const int sub_id, const int obj_id, const int D, DenseMatrix &A, vector<DenseMatrix> &R){

#ifdef use_mkl
        Vec d_v(D);
        cblas_xgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, D, D, 1.0,
                    A.data().begin() + D * sub_id, D, R[rel_id].data().begin(), D, 0.0,
                    d_v.data().begin(), D);

        return cblas_xdot(D, d_v.data().begin(), 1, A.data().begin() + D * obj_id, 1);

#else
        DenseMatrix &R_k = R[rel_id];

        Vec p_tmp = prod(row(A, sub_id), R_k);
        return inner_product(p_tmp.data().begin(), A.data().begin() + D * obj_id, D);
#endif
    }

    inline int transform_transe2rescal(DenseMatrix &transeA, DenseMatrix &transeR, DenseMatrix &rescalA, vector<DenseMatrix> &rescalR){

        int rescal_d =  2 * transeA.size2() + 1;

        rescalA.resize(transeA.size1(), rescal_d);
        rescalR.resize(transeR.size1(), DenseMatrix(rescal_d, rescal_d));

        for (int entity_index = 0; entity_index < transeA.size1(); entity_index++) {
            for (int d = 0; d < transeA.size2(); d++) {
                rescalA(entity_index, d) = 1;
            }

            for (int d = transeA.size2(); d < 2 * transeA.size2(); d++) {
                rescalA(entity_index, d) = transeA(entity_index, d - transeA.size2());
            }

            rescalA(entity_index, 2 * transeA.size2()) = inner_product(
                    transeA.data().begin() + entity_index * transeA.size2(),
                    transeA.data().begin() + entity_index * transeA.size2(), transeA.size2());
        }

        for (int rel_id = 0; rel_id < transeR.size1(); rel_id++) {
            DenseMatrix &R = rescalR[rel_id];

            R.clear();

            R(0, 0) = - inner_product(transeR.data().begin() + rel_id * transeA.size2(), transeR.data().begin() + rel_id * transeA.size2(), transeA.size2());

            for (int row = 0; row < transeA.size2(); row++) {
                R(row, row + transeA.size2()) = 2 * transeR(rel_id, row);
            }

            R(0, 2 * transeA.size2()) = -1;

            for (int row = transeA.size2(); row < 2 * transeA.size2(); row++) {
                R(row, row - transeA.size2()) = -2 * transeR(rel_id, row - transeA.size2());
            }

            for (int row = transeA.size2(); row < 2 * transeA.size2(); row++) {
                R(row, row) = 2;
            }

            R(2 * transeA.size2(), 0) = -1;

        }

//        for (int rel_id = 0; rel_id < transeR.size1(); rel_id++) {
//            DenseMatrix &R = rescalR[rel_id];
//
//            R.clear();
//
//            R(0, 0) = inner_product(transeR.data().begin() + rel_id * transeA.size2(), transeR.data().begin() + rel_id * transeA.size2(), transeA.size2());
//
//            for (int row = 0; row < transeA.size2(); row++) {
//                R(row, row + transeA.size2()) = -2 * transeR(rel_id, row);
//            }
//
//            R(0, 2 * transeA.size2()) = 1;
//
//            for (int row = transeA.size2(); row < 2 * transeA.size2(); row++) {
//                R(row, row - transeA.size2()) = 2 * transeR(rel_id, row - transeA.size2());
//            }
//
//            for (int row = transeA.size2(); row < 2 * transeA.size2(); row++) {
//                R(row, row) = -2;
//            }
//
//            R(2 * transeA.size2(), 0) = 1;
//
//        }

        return rescal_d;
    }

    inline void transform_hole2rescal(DenseMatrix &HoleP, vector<DenseMatrix> &rescalR){

        rescalR.resize(HoleP.size1(), DenseMatrix(HoleP.size2(), HoleP.size2()));

        for (int rel_id = 0; rel_id < HoleP.size1(); rel_id++) {
            DenseMatrix &R = rescalR[rel_id];

            for (int row = 0; row < HoleP.size2(); row++) {
                for (int col = 0; col < row; col++) {
                    R(row, col) = HoleP(rel_id, HoleP.size2() - row + col);
                }

                for (int col = row; col < HoleP.size2(); col++) {
                    R(row, col) = HoleP(rel_id, col - row);
                }
            }
        }
    }

    inline value_type sigmoid(value_type x) {
        return 1.0 / (1 + exp(-x));
    }

    inline value_type g_sigmoid(value_type x) {
        return x * (1 - x);
    }

    inline void involution(value_type *in, Vec &out, const int d) {
        out.resize(d);
        for (int i = 0; i < d; ++i) {
            out(i) = in[d - i - 1];
        }
    }

    // in-place right shift
    inline void right_shift(Vec &result) {
        value_type last = result(result.size() - 1);
        for (int i = result.size() - 1; i >=1; i--) {
            result(i) = result(i - 1);
        }
        result(0) = last;
    }

    inline void element_prod_complex(value_type _Complex *a, value_type _Complex *b, value_type _Complex *results, const int d){
        for (int i = 0; i < d; i++) {
            results[i] = a[i] * b[i];
        }
    }

    inline void convolution(DFTI_DESCRIPTOR_HANDLE &descriptor, value_type *left, value_type *right, Vec &result, const int d) {
        //compute DFT
        value_type _Complex *spt1 = new value_type _Complex[d];
        value_type _Complex *spt2 = new value_type _Complex[d];

        DftiComputeForward(descriptor, left, spt1);
        DftiComputeForward(descriptor, right, spt2);

        value_type _Complex *spt_mult = new value_type _Complex[d];
        element_prod_complex(spt1, spt2, spt_mult, d);

        result.resize(d);
        DftiComputeBackward(descriptor, spt_mult, result.data().begin());

        delete[] spt1;
        delete[] spt2;
        delete[] spt_mult;

    }

    inline void correlation(DFTI_DESCRIPTOR_HANDLE &descriptor, value_type *left, value_type *right, Vec &result, const int d){
        Vec inl(d);
        involution(left, inl, d);
        convolution(descriptor, inl.data().begin(), right, result, d);
        right_shift(result);
    }

    inline value_type cal_hole_score(int sub, int obj, int rel, const int D, DenseMatrix &E, DenseMatrix &P, DFTI_DESCRIPTOR_HANDLE &descriptor, const bool do_sigmoid=true) {
        value_type *e_s = E.data().begin() + sub * D;
        value_type *e_o = E.data().begin() + obj * D;
        value_type *r_k = P.data().begin() + rel * D;

        Vec result;

        correlation(descriptor, e_s, e_o, result, D);
        value_type score = inner_product(r_k, result.data().begin(), D);

        return do_sigmoid?sigmoid(score):score;
    }

//    inline value_type cal_pipeline_ensemble_score(Parameter &parameter, vector<Weight> &pipelineEnsembleWeights, DenseMatrix &rescalA,
//                                                  vector<DenseMatrix> &rescalR, DenseMatrix &transeA, DenseMatrix &transeR,
//                                                  const int sub_id, const int rel_id, const int obj_id, const bool sub_weight) {
//
//        value_type rescal_score = cal_rescal_score(rel_id, sub_id, obj_id, parameter.rescal_D, rescalA, rescalR);
//        value_type transeE_score = cal_transe_score(sub_id, obj_id, rel_id, parameter.transe_D, parameter.L1_flag, transeA, transeR);
//        Weight &weight = pipelineEnsembleWeights[rel_id];
//        if(sub_weight){
//            return weight.s_w1 * rescal_score + weight.s_w2 * transeE_score + weight.s_bias;
//        } else {
//            return weight.o_w1 * rescal_score + weight.o_w2 * transeE_score + weight.o_bias;
//        }
//    }

    inline value_type cal_ensemble_score(Parameter &parameter, vector<SimpleWeight> &ensembleWeights, DenseMatrix &rescalA,
                                         vector<DenseMatrix> &rescalR, DenseMatrix &transeA, DenseMatrix &transeR,
                                         const int sub_id, const int rel_id, const int obj_id, vector<min_max> *min_max_values = nullptr) {

        value_type rescal_score = cal_rescal_score(rel_id, sub_id, obj_id, parameter.rescal_D, rescalA, rescalR);
        value_type transeE_score = cal_transe_score(sub_id, obj_id, rel_id, parameter.transe_D, parameter.L1_flag, transeA, transeR);
        if(parameter.normalize){
            rescal_score = (rescal_score - (*min_max_values)[rel_id].min1)/ ((*min_max_values)[rel_id].max1 - (*min_max_values)[rel_id].min1);
            transeE_score = (transeE_score - (*min_max_values)[rel_id].min2) / ((*min_max_values)[rel_id].max2 - (*min_max_values)[rel_id].min2);
        }
        SimpleWeight &weight = ensembleWeights[rel_id];
        return weight.w1 * rescal_score + weight.w2 * transeE_score;
    }

    inline value_type cal_score(Method method, Parameter &parameter, const int sub_id, const int rel_id, const int obj_id,
              DenseMatrix *rescalA, vector<DenseMatrix> *rescalR,
              DenseMatrix *transeA, DenseMatrix *transeR, DenseMatrix *holeE, DenseMatrix *holeP,
              DFTI_DESCRIPTOR_HANDLE *descriptor, vector<SimpleWeight> *ensembleWeight,
              vector<min_max> *min_max_values = nullptr) {

        value_type score;

        switch(method) {
            case m_RESCAL:
            case m_RESCAL_RANK:
                score = cal_rescal_score(rel_id, sub_id, obj_id, parameter.rescal_D, *rescalA, *rescalR);
                break;
            case m_TransE:
                score = cal_transe_score(sub_id, obj_id, rel_id, parameter.transe_D, parameter.L1_flag, *transeA,
                                         *transeR);
                break;
            case m_HOLE:
                score = cal_hole_score(sub_id, obj_id, rel_id, parameter.hole_D, *holeE, *holeP, *descriptor);
                break;
            case m_RTLREnsemble:
            case m_HTLREnsemble:
            case m_Ensemble:
                score = cal_ensemble_score(parameter, *ensembleWeight, *rescalA, *rescalR, *transeA, *transeR, sub_id,
                                           rel_id, obj_id, min_max_values);
                break;
            case m_RHLREnsemble: {
                value_type rescal_score = cal_rescal_score(rel_id, sub_id, obj_id, parameter.rescal_D, *rescalA,
                                                           *rescalR);
                value_type hole_score = cal_hole_score(sub_id, obj_id, rel_id, parameter.hole_D, *holeE, *holeP,
                                                       *descriptor, false);
                if(parameter.normalize){
                    rescal_score = (rescal_score - (*min_max_values)[rel_id].min1)/ ((*min_max_values)[rel_id].max1 - (*min_max_values)[rel_id].min1);
                    hole_score = (hole_score - (*min_max_values)[rel_id].min2) / ((*min_max_values)[rel_id].max2 - (*min_max_values)[rel_id].min2);
                }
                score = ensembleWeight->at(rel_id).w1 * rescal_score + ensembleWeight->at(rel_id).w2 * hole_score;
                break;
            }
            case m_RHTLREnsemble: {
                value_type rescal_score = cal_rescal_score(rel_id, sub_id, obj_id, parameter.rescal_D, *rescalA,
                                                           *rescalR);
                value_type hole_score = cal_hole_score(sub_id, obj_id, rel_id, parameter.hole_D, *holeE, *holeP,
                                                       *descriptor, false);
                value_type transe_score = cal_transe_score(sub_id, obj_id, rel_id, parameter.transe_D, parameter.L1_flag, *transeA, *transeR);

                if(parameter.normalize){
                    rescal_score = (rescal_score - (*min_max_values)[rel_id].min1)/ ((*min_max_values)[rel_id].max1 - (*min_max_values)[rel_id].min1);
                    hole_score = (hole_score - (*min_max_values)[rel_id].min2) / ((*min_max_values)[rel_id].max2 - (*min_max_values)[rel_id].min2);
                    transe_score = (transe_score - (*min_max_values)[rel_id].min3) / ((*min_max_values)[rel_id].max3 - (*min_max_values)[rel_id].min3);
                }

                score = ensembleWeight->at(rel_id).w1 * rescal_score + ensembleWeight->at(rel_id).w2 * hole_score + ensembleWeight->at(rel_id).w3 * transe_score;
                break;
            }
            default:
                cerr << "unrecognized method!" << endl;
                exit(1);
        }

        return score;
    }

    // slow
    inline void kronecker_product(const Vec &A, const Vec &B, DenseMatrix &C, const int size) {
        C = outer_prod(A, B);
    }

    // faster
    inline void kronecker_product(value_type *A, value_type *B, DenseMatrix &C, const int size) {
        C.resize(size, size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                C(i, j) = A[i] * B[j];
            }
        }
    }

    // faster
    inline void kronecker_product(value_type *A, value_type *B, value_type *C, const int size) {

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                C[i * size + j] = A[i] * B[j];
            }
        }
    }
};
#endif //CALCULATOR_H
