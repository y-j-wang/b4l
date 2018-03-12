#ifndef EVALUATIONUTIL_H
#define EVALUATIONUTIL_H

#include "tf/util/Base.h"
#include "tf/util/Data.h"
#include "tf/util/CompareUtil.h"
#include "tf/util/ThreadUtil.h"
#include "tf/util/RandomUtil.h"
#include "tf/util/Calculator.h"
#include "tf/util/Parameter.h"
#include "tf/alg/Sampler.h"
#include "tf/struct/Min_Max.h"

using namespace Calculator;

struct hit_rate {
    value_type count_s;
    value_type count_o;
    value_type count_s_ranking;
    value_type count_o_ranking;
    value_type count_s_filtering;
    value_type count_o_filtering;
    value_type count_s_ranking_filtering;
    value_type count_o_ranking_filtering;
    value_type inv_count_s_ranking;
    value_type inv_count_o_ranking;
    value_type inv_count_s_ranking_filtering;
    value_type inv_count_o_ranking_filtering;
};

namespace EvaluationUtil {

    inline void print_hit_rate(string prefix, const int hit_rate_topk, hit_rate result) {
        if (result.count_s != -1) {
            cout << prefix << "hit_rate_subject@" << hit_rate_topk << ": " << result.count_s
                 << ", hit_rate_object@" << hit_rate_topk << ": " << result.count_o << ", subject_ranking: "
                 << result.count_s_ranking << ", object_ranking: " << result.count_o_ranking << endl;
        }
        cout << prefix << "hit_rate_subject_filter@" << hit_rate_topk << ": " << result.count_s_filtering
             << ", hit_rate_object_filter@" << hit_rate_topk << ": " << result.count_o_filtering << ", subject_ranking_filter: "
             << result.count_s_ranking_filtering << ", object_ranking_filter: " << result.count_o_ranking_filtering << endl;

        if(result.inv_count_s_ranking!=-1){
            cout << prefix << "subject_MRR: " << result.inv_count_s_ranking << ", object_MRR: " << result.inv_count_o_ranking << endl;
        }

        cout << prefix << "subject_MRR_filter: " << result.inv_count_s_ranking_filtering << ", object_MRR_filter: " << result.inv_count_o_ranking_filtering << endl;
    }

    inline void print_hit_rate_train(string prefix, const int hit_rate_topk, hit_rate result) {
        cout << prefix << "hit_rate_subject@" << hit_rate_topk << ": " << result.count_s
             << ", hit_rate_object@" << hit_rate_topk << ": " << result.count_o << ", subject_ranking: "
             << result.count_s_ranking << ", object_ranking: " << result.count_o_ranking << endl;

    }

    inline void print_hit_rate_rel(string prefix, const int hit_rate_topk, hit_rate result) {
        cout << prefix << "hit_rate_relation@" << hit_rate_topk << ": "
             << result.count_s << ", relation_ranking: " << result.count_s_ranking
             << ", hit_rate_relation_filter@" << hit_rate_topk << ": " << result.count_s_filtering
             << ", relation_ranking_filter: " << result.count_s_ranking_filtering
             << endl;
    }

    inline void print_map(string prefix, const int num_of_replaced_entities, pair<value_type, value_type> result) {
        cout << prefix << "MAP_subject@" << num_of_replaced_entities << ": "
             << result.first << ", MAP_object@" << num_of_replaced_entities << ": " << result.second
             << endl;
    }

    inline hit_rate eval_hit_rate(Method method, Parameter *parameter, Data *data, DenseMatrix *rescalA, vector<DenseMatrix> *rescalR,
                                  DenseMatrix *transeA, DenseMatrix *transeR, DenseMatrix *HOLE_E, DenseMatrix *HOLE_R, DFTI_DESCRIPTOR_HANDLE *descriptor,
                                  vector<SimpleWeight> *ensembleWeights,
                                  const string output_folder, vector<min_max> *min_max_values=nullptr) {

        if(method==Method::m_HOLE){
            cerr << "EvaluationUtil: eval_hit_rate should transform HOLE to RESCAL_RANK first than call m_RESCAL_RANK for evaluation!" << endl;
            exit(1);
        } else if(method==Method::m_HTLREnsemble){
            cerr << "EvaluationUtil: eval_hit_rate should transform HOLE to RESCAL_RANK first than call m_RTLREnsemble for evaluation!" << endl;
            exit(1);
        }

#ifdef detailed_eval
        boost::filesystem::create_directories(output_folder);
#endif

        // thread -> (relation, count)

#ifdef detailed_eval
        vector<vector<ull> > rel_counts_s(parameter->num_of_thread_eval, vector<ull>(data->K, 0));
        vector<vector<ull> > rel_counts_o(parameter->num_of_thread_eval, vector<ull>(data->K, 0));

        vector<vector<ull> > rel_counts_s_ranking(parameter->num_of_thread_eval, vector<ull>(data->K, 0));
        vector<vector<ull> > rel_counts_o_ranking(parameter->num_of_thread_eval, vector<ull>(data->K, 0));

        vector<vector<value_type> > inv_rel_counts_s_ranking(parameter->num_of_thread_eval, vector<value_type>(data->K, 0));
        vector<vector<value_type> > inv_rel_counts_o_ranking(parameter->num_of_thread_eval, vector<value_type>(data->K, 0));

#endif
        vector<vector<ull> > rel_counts_s_filtering(parameter->num_of_thread_eval, vector<ull>(data->K, 0));
        vector<vector<ull> > rel_counts_o_filtering(parameter->num_of_thread_eval, vector<ull>(data->K, 0));

        vector<vector<ull> > rel_counts_s_ranking_filtering(parameter->num_of_thread_eval, vector<ull>(data->K, 0));
        vector<vector<ull> > rel_counts_o_ranking_filtering(parameter->num_of_thread_eval, vector<ull>(data->K, 0));

        vector<vector<value_type> > inv_rel_counts_s_ranking_filtering(parameter->num_of_thread_eval, vector<value_type>(data->K, 0));
        vector<vector<value_type> > inv_rel_counts_o_ranking_filtering(parameter->num_of_thread_eval, vector<value_type>(data->K, 0));


        int testing_size = data->num_of_testing_triples;

        value_type *AR = nullptr;
        value_type *RAT = nullptr;

        if (method != Method::m_TransE) {
            AR = cache_aligned_allocator<value_type>().allocate(data->N * parameter->rescal_D);
            RAT = cache_aligned_allocator<value_type>().allocate(parameter->rescal_D * data->N);
        }

        for (int relation_id = 0; relation_id < data->K; relation_id++) {
            vector<Tuple<int> > &tuples = data->relation2tupleTestList_mapping[relation_id];

            if (method != Method::m_TransE && method != Method::m_HOLE) {
                // AR
                cblas_xgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            data->N, parameter->rescal_D, parameter->rescal_D, 1.0, (*rescalA).data().begin(),
                            parameter->rescal_D, (*rescalR)[relation_id].data().begin(), parameter->rescal_D, 0.0, AR,
                            parameter->rescal_D);

                // RAT
                cblas_xgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            parameter->rescal_D, data->N, parameter->rescal_D, 1.0,
                            (*rescalR)[relation_id].data().begin(),
                            parameter->rescal_D, (*rescalA).data().begin(), parameter->rescal_D, 0.0, RAT, data->N);
            }

            int size4relation = tuples.size();

            int workload = size4relation / parameter->num_of_thread_eval +
                           ((size4relation % parameter->num_of_thread_eval == 0) ? 0 : 1);
#ifdef detailed_eval
            vector<string> outputs(parameter->num_of_thread_eval, "");
#endif
            std::function<void(int)> compute_func = [&](int thread_index) -> void {

                int start = thread_index * workload;
                int end = std::min(start + workload, size4relation);
#ifdef detailed_eval
                vector<ull> &counts_s = rel_counts_s[thread_index];
                vector<ull> &counts_o = rel_counts_o[thread_index];
                vector<ull> &counts_s_ranking = rel_counts_s_ranking[thread_index];
                vector<ull> &counts_o_ranking = rel_counts_o_ranking[thread_index];
                vector<value_type> &inv_counts_s_ranking = inv_rel_counts_s_ranking[thread_index];
                vector<value_type> &inv_counts_o_ranking = inv_rel_counts_o_ranking[thread_index];
#endif
                vector<ull> &counts_s_filtering = rel_counts_s_filtering[thread_index];
                vector<ull> &counts_o_filtering = rel_counts_o_filtering[thread_index];
                vector<ull> &counts_s_ranking_filtering = rel_counts_s_ranking_filtering[thread_index];
                vector<ull> &counts_o_ranking_filtering = rel_counts_o_ranking_filtering[thread_index];
                vector<value_type> &inv_counts_s_ranking_filtering = inv_rel_counts_s_ranking_filtering[thread_index];
                vector<value_type> &inv_counts_o_ranking_filtering = inv_rel_counts_o_ranking_filtering[thread_index];

                // first: entity id, second: score
#ifdef detailed_eval
                vector<pair<int, value_type> > result(data->N);
#endif
                vector<pair<int, value_type> > result_filtering(data->N);
                int result_filtering_index = 0;

                value_type *tmp = nullptr;
                if (method != Method::m_TransE && method != Method::m_HOLE) {
                    tmp = new value_type[data->N];
                }

                for (int n = start; n < end; n++) {
#ifdef detailed_eval
                    outputs[thread_index].append("-------------------------\ntesting tuple: ");
#endif

                    Tuple<int> &test_tuple = tuples[n];
#ifdef detailed_eval
                    outputs[thread_index].append(data->entity_decoder[test_tuple.subject] + " " +
                                                 data->entity_decoder[test_tuple.object] + "\ntop" +
                                                 to_string(parameter->hit_rate_topk) + " for replacing subject:\n");
#endif

                    // first replace subject with other entities
                    if (method != Method::m_TransE && method != Method::m_HOLE) {
                        // AR*A_jt
                        cblas_xgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                    data->N, 1, parameter->rescal_D, 1.0, AR, parameter->rescal_D,
                                    (*rescalA).data().begin() + test_tuple.object * parameter->rescal_D, 1, 0.0, tmp,
                                    1);
                    }

                    result_filtering_index = 0;

                    for (int index = 0; index < data->N; index++) {
                        value_type score;
                        if (method == Method::m_TransE) {
                            score = cal_transe_score(index, test_tuple.object, relation_id, parameter->transe_D,
                                                     parameter->L1_flag, *transeA, *transeR);
                        } else if (method == Method::m_RESCAL || method == Method::m_RESCAL_RANK) {
                            score = tmp[index];
                        } else if (method == Method::m_RTLREnsemble || method == Method::m_HTLREnsemble || method == Method::m_Ensemble) {
                            value_type transE_score = cal_transe_score(index, test_tuple.object, relation_id,
                                                                       parameter->transe_D, parameter->L1_flag,
                                                                       *transeA, *transeR);
                            if (parameter->normalize) {
                                score = (*ensembleWeights)[relation_id].w1 * (tmp[index] - (*min_max_values)[relation_id].min1) /
                                        ((*min_max_values)[relation_id].max1 - (*min_max_values)[relation_id].min1) +
                                        (*ensembleWeights)[relation_id].w2 * (transE_score - (*min_max_values)[relation_id].min2)  /
                                        ((*min_max_values)[relation_id].max2 - (*min_max_values)[relation_id].min2);
                            } else if (parameter->znormalize) {
                                score = (*ensembleWeights)[relation_id].w1 * (tmp[index] - (*min_max_values)[relation_id].min1) /
                                        ((*min_max_values)[relation_id].max1) +
                                        (*ensembleWeights)[relation_id].w2 * (transE_score - (*min_max_values)[relation_id].min2)  /
                                        ((*min_max_values)[relation_id].max2);
                            } else {
                                score = (*ensembleWeights)[relation_id].w1 * tmp[index] +
                                        (*ensembleWeights)[relation_id].w2 * transE_score;
                            }

                        } else if (method == Method::m_RHLREnsemble) {
                            value_type hole_score = cal_hole_score(index, test_tuple.object, relation_id, parameter->hole_D, *HOLE_E, *HOLE_R, *descriptor, false);

                            if (parameter->normalize) {
                                score = (*ensembleWeights)[relation_id].w1 * (tmp[index] - (*min_max_values)[relation_id].min1) /
                                        ((*min_max_values)[relation_id].max1 - (*min_max_values)[relation_id].min1) +
                                        (*ensembleWeights)[relation_id].w2 * (hole_score - (*min_max_values)[relation_id].min2)  /
                                        ((*min_max_values)[relation_id].max2 - (*min_max_values)[relation_id].min2);
                            } else if (parameter->znormalize) {
                                score = (*ensembleWeights)[relation_id].w1 * (tmp[index] - (*min_max_values)[relation_id].min1) /
                                        ((*min_max_values)[relation_id].max1) +
                                        (*ensembleWeights)[relation_id].w2 * (hole_score - (*min_max_values)[relation_id].min2)  /
                                        ((*min_max_values)[relation_id].max2);
                            } else {
                                score = (*ensembleWeights)[relation_id].w1 * tmp[index] +
                                        (*ensembleWeights)[relation_id].w2 * hole_score;
                            }

                        } else if (method == Method::m_RHTLREnsemble) {

                            value_type hole_score = cal_hole_score(index, test_tuple.object, relation_id,
                                                                   parameter->hole_D, *HOLE_E, *HOLE_R, *descriptor, false);
                            value_type transE_score = cal_transe_score(index, test_tuple.object, relation_id,
                                                                       parameter->transe_D, parameter->L1_flag,
                                                                       *transeA, *transeR);

                            if (parameter->normalize) {

                                score = (*ensembleWeights)[relation_id].w1 * (tmp[index] - (*min_max_values)[relation_id].min1) /
                                        ((*min_max_values)[relation_id].max1 - (*min_max_values)[relation_id].min1) +
                                        (*ensembleWeights)[relation_id].w2 * (hole_score - (*min_max_values)[relation_id].min2)  /
                                        ((*min_max_values)[relation_id].max2 - (*min_max_values)[relation_id].min2) +
                                        (*ensembleWeights)[relation_id].w3 * (transE_score - (*min_max_values)[relation_id].min3)  /
                                        ((*min_max_values)[relation_id].max3 - (*min_max_values)[relation_id].min3);

                            } else if (parameter->znormalize) {
                                score = (*ensembleWeights)[relation_id].w1 * (tmp[index] - (*min_max_values)[relation_id].min1) /
                                        ((*min_max_values)[relation_id].max1) +
                                        (*ensembleWeights)[relation_id].w2 * (hole_score - (*min_max_values)[relation_id].min2)  /
                                        ((*min_max_values)[relation_id].max2) +
                                        (*ensembleWeights)[relation_id].w3 * (transE_score - (*min_max_values)[relation_id].min3)  /
                                        ((*min_max_values)[relation_id].max3);
                            } else {
                                score = (*ensembleWeights)[relation_id].w1 * tmp[index] + (*ensembleWeights)[relation_id].w2 * hole_score +
                                        (*ensembleWeights)[relation_id].w3 * transE_score;
                            }

                        } else if (method==Method::m_HOLE) {
                            score = cal_hole_score(index, test_tuple.object, relation_id, parameter->hole_D, *HOLE_E,
                                                   *HOLE_R, *descriptor);
                        } else {
                            cerr << "eval_hit_rate: unrecognized method!" << endl;
                            exit(1);
                        }
#ifdef detailed_eval
                        result[index] = make_pair(index, score);
#endif
                        if ((index == test_tuple.subject) ||
                            (!data->faked_s_tuple_exist(index, relation_id, test_tuple.object))) {
                            result_filtering[result_filtering_index] = make_pair(index, score);
                            result_filtering_index++;
                        }
                    }

                    // sort
                    if (method == Method::m_TransE) {
#ifdef detailed_eval
                        std::sort(result.begin(), result.end(), &CompareUtil::pairLessCompare<int>);
#endif
                        std::sort(result_filtering.begin(), result_filtering.begin() + result_filtering_index, &CompareUtil::pairLessCompare<int>);
                    } else {
#ifdef detailed_eval
                        std::sort(result.begin(), result.end(), &CompareUtil::pairGreaterCompare<int>);
#endif
                        std::sort(result_filtering.begin(), result_filtering.begin() + result_filtering_index,
                                  &CompareUtil::pairGreaterCompare<int>);
                    }

#ifdef detailed_eval
                    bool found = false;
#endif
                    bool found_filtering = false;

                    for (int i = 0; i < parameter->hit_rate_topk; i++) {
#ifdef detailed_eval
                        outputs[thread_index].append(
                                data->entity_decoder[result[i].first] + " " + to_string(result[i].second) + "\n");

                        if (result[i].first == test_tuple.subject) {
                            counts_s[relation_id]++;
                            counts_s_ranking[relation_id] += i + 1;
                            inv_counts_s_ranking[relation_id] += 1.0 / (i + 1);
                            found = true;
                        }
#endif

                        if (result_filtering[i].first == test_tuple.subject) {
                            counts_s_filtering[relation_id]++;
                            counts_s_ranking_filtering[relation_id] += i + 1;
                            inv_counts_s_ranking_filtering[relation_id] += 1.0 / (i + 1);
                            found_filtering = true;
                        }
                    }

#ifdef detailed_eval
                    if (!found) {
                        for (int i = parameter->hit_rate_topk; i < data->N; i++) {
                            if (result[i].first == test_tuple.subject) {
                                counts_s_ranking[relation_id] += i + 1;
                                inv_counts_s_ranking[relation_id] += 1.0 / (i + 1);
                                found = true;
                                break;
                            }
                        }
                    }
#endif

                    if (!found_filtering) {
                        for (int i = parameter->hit_rate_topk; i < result_filtering_index; i++) {
                            if (result_filtering[i].first == test_tuple.subject) {
                                counts_s_ranking_filtering[relation_id] += i + 1;
                                inv_counts_s_ranking_filtering[relation_id] += 1.0 / (i + 1);
                                found_filtering = true;
                                break;
                            }
                        }
                    }

#ifdef detailed_eval
                    if ((!found) || (!found_filtering)) {
                        cerr << "logical error in eval_hit_rate! (1)" << endl;
                        exit(1);
                    }
#else
                    if (!found_filtering) {
                        cerr << "logical error in eval_hit_rate! (1)" << endl;
                        exit(1);
                    }
#endif

                    // then replace object with other entities
                    if (method != Method::m_TransE && method!= Method::m_HOLE) {
                        // A_i * RAT
                        cblas_xgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                    1, data->N, parameter->rescal_D, 1.0,
                                    (*rescalA).data().begin() + test_tuple.subject * parameter->rescal_D,
                                    parameter->rescal_D, RAT, data->N, 0.0, tmp, data->N);
                    }

                    result_filtering_index = 0;

                    for (int index = 0; index < data->N; index++) {

                        value_type score;

                        if (method == Method::m_TransE) {
                            score = cal_transe_score(test_tuple.subject, index, relation_id, parameter->transe_D,
                                                     parameter->L1_flag, *transeA, *transeR);
                        } else if (method == Method::m_RESCAL || method == Method::m_RESCAL_RANK) {
                            score = tmp[index];
                        } else if (method == Method::m_RTLREnsemble || method == Method::m_HTLREnsemble || method == Method::m_Ensemble) {
                            value_type transE_score = cal_transe_score(test_tuple.subject, index, relation_id,
                                                                       parameter->transe_D, parameter->L1_flag,
                                                                       *transeA, *transeR);
                            if (parameter->normalize) {
                                score = (*ensembleWeights)[relation_id].w1 * (tmp[index] - (*min_max_values)[relation_id].min1) /
                                        ((*min_max_values)[relation_id].max1 - (*min_max_values)[relation_id].min1) +
                                        (*ensembleWeights)[relation_id].w2 * (transE_score - (*min_max_values)[relation_id].min2)  /
                                        ((*min_max_values)[relation_id].max2 - (*min_max_values)[relation_id].min2);
                            } else if (parameter->znormalize){
                                score = (*ensembleWeights)[relation_id].w1 * (tmp[index] - (*min_max_values)[relation_id].min1) /
                                        ((*min_max_values)[relation_id].max1) +
                                        (*ensembleWeights)[relation_id].w2 * (transE_score - (*min_max_values)[relation_id].min2)  /
                                        ((*min_max_values)[relation_id].max2);
                            } else {
                                score = (*ensembleWeights)[relation_id].w1 * tmp[index] +
                                        (*ensembleWeights)[relation_id].w2 * transE_score;
                            }

                        } else if (method == Method::m_RHLREnsemble) {
                            value_type hole_score = cal_hole_score(test_tuple.subject, index, relation_id, parameter->hole_D, *HOLE_E, *HOLE_R, *descriptor, false);

                            if (parameter->normalize) {
                                score = (*ensembleWeights)[relation_id].w1 * (tmp[index] - (*min_max_values)[relation_id].min1) /
                                        ((*min_max_values)[relation_id].max1 - (*min_max_values)[relation_id].min1) +
                                        (*ensembleWeights)[relation_id].w2 * (hole_score - (*min_max_values)[relation_id].min2)  /
                                        ((*min_max_values)[relation_id].max2 - (*min_max_values)[relation_id].min2);
                            } else if (parameter->znormalize) {
                                score = (*ensembleWeights)[relation_id].w1 * (tmp[index] - (*min_max_values)[relation_id].min1) /
                                        ((*min_max_values)[relation_id].max1) +
                                        (*ensembleWeights)[relation_id].w2 * (hole_score - (*min_max_values)[relation_id].min2)  /
                                        ((*min_max_values)[relation_id].max2);
                            } else {
                                score = (*ensembleWeights)[relation_id].w1 * tmp[index] +
                                        (*ensembleWeights)[relation_id].w2 * hole_score;
                            }

                        } else if (method == Method::m_RHTLREnsemble) {

                            value_type hole_score = cal_hole_score(test_tuple.subject, index, relation_id, parameter->hole_D, *HOLE_E, *HOLE_R, *descriptor, false);
                            value_type transE_score = cal_transe_score(test_tuple.subject, index, relation_id,
                                                                       parameter->transe_D, parameter->L1_flag,
                                                                       *transeA, *transeR);

                            if (parameter->normalize) {
                                score = (*ensembleWeights)[relation_id].w1 * (tmp[index] - (*min_max_values)[relation_id].min1) /
                                        ((*min_max_values)[relation_id].max1 - (*min_max_values)[relation_id].min1) +
                                        (*ensembleWeights)[relation_id].w2 * (hole_score - (*min_max_values)[relation_id].min2)  /
                                        ((*min_max_values)[relation_id].max2 - (*min_max_values)[relation_id].min2) +
                                        (*ensembleWeights)[relation_id].w3 * (transE_score - (*min_max_values)[relation_id].min3)  /
                                        ((*min_max_values)[relation_id].max3 - (*min_max_values)[relation_id].min3);
                            } else if(parameter->znormalize) {
                                score = (*ensembleWeights)[relation_id].w1 * (tmp[index] - (*min_max_values)[relation_id].min1) /
                                        ((*min_max_values)[relation_id].max1) +
                                        (*ensembleWeights)[relation_id].w2 * (hole_score - (*min_max_values)[relation_id].min2)  /
                                        ((*min_max_values)[relation_id].max2) +
                                        (*ensembleWeights)[relation_id].w3 * (transE_score - (*min_max_values)[relation_id].min3)  /
                                        ((*min_max_values)[relation_id].max3);
                            } else {
                                score = (*ensembleWeights)[relation_id].w1 * tmp[index] + (*ensembleWeights)[relation_id].w2 * hole_score +
                                        (*ensembleWeights)[relation_id].w3 * transE_score;
                            }

                        } else if (method==Method::m_HOLE) {
                            score = cal_hole_score(test_tuple.subject, index, relation_id, parameter->hole_D, *HOLE_E,
                                                   *HOLE_R, *descriptor);
                        } else {
                            cerr << "eval_hit_rate: unrecognized method!" << endl;
                            exit(1);
                        }

#ifdef detailed_eval
                        result[index] = make_pair(index, score);
#endif

                        if ((index == test_tuple.object) ||
                            (!data->faked_o_tuple_exist(test_tuple.subject, relation_id, index))) {
                            result_filtering[result_filtering_index] = make_pair(index, score);
                            result_filtering_index++;
                        }
                    }

                    // sort
                    if (method == Method::m_TransE) {
#ifdef detailed_eval
                        std::sort(result.begin(), result.end(), &CompareUtil::pairLessCompare<int>);
#endif
                        std::sort(result_filtering.begin(), result_filtering.begin() + result_filtering_index, &CompareUtil::pairLessCompare<int>);
                    } else {
#ifdef detailed_eval
                        std::sort(result.begin(), result.end(), &CompareUtil::pairGreaterCompare<int>);
#endif
                        std::sort(result_filtering.begin(), result_filtering.begin() + result_filtering_index,
                                  &CompareUtil::pairGreaterCompare<int>);
                    }

                    found_filtering = false;
#ifdef detailed_eval
                    found = false;
                    outputs[thread_index].append(
                            "top" + to_string(parameter->hit_rate_topk) + " for replacing object:\n");
#endif

                    for (int i = 0; i < parameter->hit_rate_topk; i++) {
#ifdef detailed_eval
                        outputs[thread_index].append(
                                data->entity_decoder[result[i].first] + " " + to_string(result[i].second) + "\n");

                        if (result[i].first == test_tuple.object) {
                            counts_o[relation_id]++;
                            counts_o_ranking[relation_id] += i + 1;
                            inv_counts_o_ranking[relation_id] += 1.0 / (i + 1);
                            found = true;
                        }
#endif

                        if (result_filtering[i].first == test_tuple.object) {
                            counts_o_filtering[relation_id]++;
                            counts_o_ranking_filtering[relation_id] += i + 1;
                            inv_counts_o_ranking_filtering[relation_id] += 1.0 / (i + 1);
                            found_filtering = true;
                        }
                    }

#ifdef detailed_eval
                    if (!found) {
                        for (int i = parameter->hit_rate_topk; i < data->N; i++) {
                            if (result[i].first == test_tuple.object) {
                                counts_o_ranking[relation_id] += i + 1;
                                inv_counts_o_ranking[relation_id] += 1.0 / (i + 1);
                                found = true;
                                break;
                            }
                        }
                    }
#endif

                    if (!found_filtering) {
                        for (int i = parameter->hit_rate_topk; i < result_filtering_index; i++) {
                            if (result_filtering[i].first == test_tuple.object) {
                                counts_o_ranking_filtering[relation_id] += i + 1;
                                inv_counts_o_ranking_filtering[relation_id] += 1.0 / (i + 1);
                                found_filtering = true;
                                break;
                            }
                        }
                    }

#ifdef detailed_eval
                    if ((!found) || (!found_filtering)) {
                        cerr << "logical error in eval_hit_rate! (2)" << endl;
                        exit(1);
                    }
#else
                    if (!found_filtering) {
                        cerr << "logical error in eval_hit_rate! (2)" << endl;
                        exit(1);
                    }
#endif
                }

                if (method != Method::m_TransE) {
                    delete[] tmp;
                }
            };

            ThreadUtil::execute_threads(compute_func, parameter->num_of_thread_eval);

#ifdef detailed_eval
            ofstream output(output_folder + "/topk_results_relation_" + to_string(relation_id) + ".dat");
            for (auto s:outputs) {
                output << s;
            }
            output.close();
#endif
        }

        if (method != Method::m_TransE) {
            cache_aligned_allocator<value_type>().deallocate(AR, data->N * parameter->rescal_D);
            cache_aligned_allocator<value_type>().deallocate(RAT, parameter->rescal_D * data->N);
        }

        hit_rate measure;
#ifdef detailed_eval
        measure.count_s = 0;
        measure.count_o = 0;
        measure.count_s_ranking = 0;
        measure.count_o_ranking = 0;
        measure.inv_count_s_ranking = 0;
        measure.inv_count_o_ranking = 0;
        vector<ull> rel_s(data->K, 0);
        vector<ull> rel_o(data->K, 0);
        vector<ull> rel_s_ranking(data->K, 0);
        vector<ull> rel_o_ranking(data->K, 0);
        vector<value_type > inv_rel_s_ranking(data->K, 0);
        vector<value_type > inv_rel_o_ranking(data->K, 0);
#else
        measure.count_s = -1;
        measure.count_o = -1;
        measure.count_s_ranking = -1;
        measure.count_o_ranking = -1;
        measure.inv_count_s_ranking = -1;
        measure.inv_count_o_ranking = -1;
#endif
        measure.count_s_filtering = 0;
        measure.count_o_filtering = 0;
        measure.count_s_ranking_filtering = 0;
        measure.count_o_ranking_filtering = 0;
        measure.inv_count_s_ranking_filtering = 0;
        measure.inv_count_o_ranking_filtering = 0;

        vector<ull> rel_s_filtering(data->K, 0);
        vector<ull> rel_o_filtering(data->K, 0);
        vector<ull> rel_s_ranking_filtering(data->K, 0);
        vector<ull> rel_o_ranking_filtering(data->K, 0);

        vector<value_type > inv_rel_s_ranking_filtering(data->K, 0);
        vector<value_type > inv_rel_o_ranking_filtering(data->K, 0);

        for (int thread_id = 0; thread_id < parameter->num_of_thread_eval; thread_id++) {
#ifdef detailed_eval
            vector<ull> &counts_s = rel_counts_s[thread_id];
            vector<ull> &counts_o = rel_counts_o[thread_id];
            vector<ull> &counts_s_ranking = rel_counts_s_ranking[thread_id];
            vector<ull> &counts_o_ranking = rel_counts_o_ranking[thread_id];
            vector<value_type> &inv_counts_s_ranking = inv_rel_counts_s_ranking[thread_id];
            vector<value_type> &inv_counts_o_ranking = inv_rel_counts_o_ranking[thread_id];
#endif
            vector<ull> &counts_s_filtering = rel_counts_s_filtering[thread_id];
            vector<ull> &counts_o_filtering = rel_counts_o_filtering[thread_id];
            vector<ull> &counts_s_ranking_filtering = rel_counts_s_ranking_filtering[thread_id];
            vector<ull> &counts_o_ranking_filtering = rel_counts_o_ranking_filtering[thread_id];
            vector<value_type> &inv_counts_s_ranking_filtering = inv_rel_counts_s_ranking_filtering[thread_id];
            vector<value_type> &inv_counts_o_ranking_filtering = inv_rel_counts_o_ranking_filtering[thread_id];

            for (int relation_id = 0; relation_id < data->K; relation_id++) {
#ifdef detailed_eval
                measure.count_s += counts_s[relation_id];
                measure.count_o += counts_o[relation_id];
                measure.count_s_ranking += counts_s_ranking[relation_id];
                measure.count_o_ranking += counts_o_ranking[relation_id];
                measure.inv_count_s_ranking += inv_counts_s_ranking[relation_id];
                measure.inv_count_o_ranking += inv_counts_o_ranking[relation_id];

                rel_s[relation_id] += counts_s[relation_id];
                rel_o[relation_id] += counts_o[relation_id];
                rel_s_ranking[relation_id] += counts_s_ranking[relation_id];
                rel_o_ranking[relation_id] += counts_o_ranking[relation_id];
                inv_rel_s_ranking[relation_id] += inv_counts_s_ranking[relation_id];
                inv_rel_o_ranking[relation_id] += inv_counts_o_ranking[relation_id];
#endif
                measure.count_s_filtering += counts_s_filtering[relation_id];
                measure.count_o_filtering += counts_o_filtering[relation_id];
                measure.count_s_ranking_filtering += counts_s_ranking_filtering[relation_id];
                measure.count_o_ranking_filtering += counts_o_ranking_filtering[relation_id];
                measure.inv_count_s_ranking_filtering += inv_counts_s_ranking_filtering[relation_id];
                measure.inv_count_o_ranking_filtering += inv_counts_o_ranking_filtering[relation_id];

                rel_s_filtering[relation_id] += counts_s_filtering[relation_id];
                rel_o_filtering[relation_id] += counts_o_filtering[relation_id];
                rel_s_ranking_filtering[relation_id] += counts_s_ranking_filtering[relation_id];
                rel_o_ranking_filtering[relation_id] += counts_o_ranking_filtering[relation_id];
                inv_rel_s_ranking_filtering[relation_id] += inv_counts_s_ranking_filtering[relation_id];
                inv_rel_o_ranking_filtering[relation_id] += inv_counts_o_ranking_filtering[relation_id];
            }
        }

#ifdef detailed_eval
        measure.count_s /= testing_size;
        measure.count_o /= testing_size;
        measure.count_s_ranking /= testing_size;
        measure.count_o_ranking /= testing_size;
        measure.inv_count_s_ranking /= testing_size;
        measure.inv_count_o_ranking /= testing_size;
#endif

        measure.count_s_filtering /= testing_size;
        measure.count_o_filtering /= testing_size;
        measure.count_s_ranking_filtering /= testing_size;
        measure.count_o_ranking_filtering /= testing_size;
        measure.inv_count_s_ranking_filtering /= testing_size;
        measure.inv_count_o_ranking_filtering /= testing_size;


#ifdef detailed_eval
        ofstream output(output_folder + "/statistics.dat");

        output << "hit_rate_subject@" << parameter->hit_rate_topk << ": " << measure.count_s << endl;
        output << "hit_rate_object@" << parameter->hit_rate_topk << ": " << measure.count_o << endl;
        output << "subject_ranking: " << measure.count_s_ranking << endl;
        output << "object_ranking: " << measure.count_o_ranking << endl;
        output << "subject_MRR: " << measure.inv_count_s_ranking << endl;
        output << "object_MRR: " << measure.inv_count_o_ranking << endl;

        output << "hit_rate_subject_filter@" << parameter->hit_rate_topk << ": " << measure.count_s_filtering << endl;
        output << "hit_rate_object_filter@" << parameter->hit_rate_topk << ": " << measure.count_o_filtering << endl;
        output << "subject_ranking_filter: " << measure.count_s_ranking_filtering << endl;
        output << "object_ranking_filter: " << measure.count_o_ranking_filtering << endl;
        output << "subject_MRR_filter: " << measure.inv_count_s_ranking_filtering << endl;
        output << "object_MRR_filter: " << measure.inv_count_o_ranking_filtering << endl;


        output << "-------------------" << endl;
        for (int relation_id = 0; relation_id < data->K; relation_id++) {
            value_type size = data->relation2tupleTestList_mapping[relation_id].size();
            output << "relation id: " << relation_id << endl;
            output << "hit_rate_subject@" << parameter->hit_rate_topk << ": " << rel_s[relation_id] / size << endl;
            output << "hit_rate_object@" << parameter->hit_rate_topk << ": " << rel_o[relation_id] / size << endl;
            output << "subject_ranking: " << rel_s_ranking[relation_id] / size << endl;
            output << "object_ranking: " << rel_o_ranking[relation_id] / size << endl;
            output << "subject_MRR: " << inv_rel_s_ranking[relation_id] / size << endl;
            output << "object_MRR: " << inv_rel_o_ranking[relation_id] / size << endl;

            output << "hit_rate_subject_filter@" << parameter->hit_rate_topk << ": " << rel_s_filtering[relation_id] / size << endl;
            output << "hit_rate_object_filter@" << parameter->hit_rate_topk << ": " << rel_o_filtering[relation_id] / size << endl;
            output << "subject_ranking_filter: " << rel_s_ranking_filtering[relation_id] / size << endl;
            output << "object_ranking_filter: " << rel_o_ranking_filtering[relation_id] / size << endl;
            output << "subject_MRR_filter: " << inv_rel_s_ranking_filtering[relation_id] / size << endl;
            output << "object_MRR_filter: " << inv_rel_o_ranking_filtering[relation_id] / size << endl;
            output << "-------------------" << endl;
        }

        output.close();
#endif

        return measure;
    }

    inline value_type cal_MAP(Method method, Parameter *parameter, Data *data, const bool replace_sub, DenseMatrix *rescalA,
            vector<DenseMatrix> *rescalR,
            DenseMatrix *transeA, DenseMatrix *transeR, DenseMatrix *holeE, DenseMatrix *holeP,
            DFTI_DESCRIPTOR_HANDLE *descriptor, vector<SimpleWeight> *ensembleWeight, vector<min_max> *min_max_values=nullptr) {

        // Y is replaced
        map<pair<int, int>, vector<int> > &testXRelY = replace_sub ? data->testObjRel2Sub : data->testSubRel2Obj;
        int total_size = testXRelY.size();

        int workload = total_size / parameter->num_of_thread_eval + ((total_size % parameter->num_of_thread_eval == 0) ? 0 : 1);
        vector<value_type> replaced_map(parameter->num_of_thread_eval, 0);

        std::function<void(int)> compute_func = [&](int thread_index) -> void {

            int start = thread_index * workload;
            int end = std::min(start + workload, total_size);

            for (int i = start; i < end; i++) {

                pair<int, int> &XRelKey = replace_sub ? data->testObjRel2SubKeys[i] : data->testSubRel2ObjKeys[i];

                vector<int> &existed_Ys = testXRelY.find(XRelKey)->second;
                unordered_set<string> trueKeys;

                vector<pair<string, value_type> > replaceYResult(existed_Ys.size() * (parameter->num_of_replaced_entities + 1));
                int replaceYResultIndex = 0;

                for (int j = 0; j < existed_Ys.size(); j++) {
                    int existed_Y = existed_Ys[j];

                    string str_key;
                    value_type score;

                    if(replace_sub){
                        str_key = to_string(existed_Y) + "," + to_string(XRelKey.second) + "," + to_string(XRelKey.first);
                        score = cal_score(method, *parameter, existed_Y, XRelKey.second, XRelKey.first, rescalA, rescalR, transeA, transeR, holeE, holeP, descriptor, ensembleWeight, min_max_values);
                    } else {
                        str_key = to_string(XRelKey.first) + "," + to_string(XRelKey.second) + "," + to_string(existed_Y);
                        score = cal_score(method, *parameter, XRelKey.first, XRelKey.second, existed_Y, rescalA, rescalR, transeA, transeR, holeE, holeP, descriptor, ensembleWeight, min_max_values);
                    }

                    // put true tuple
                    trueKeys.insert(str_key);

                    replaceYResult[replaceYResultIndex] = make_pair(str_key, score);
                    replaceYResultIndex++;

                    for (int k = 0; k < parameter->num_of_replaced_entities; k++) {
                        int random_Y_id = RandomUtil::uniform_int(0, data->N);
                        value_type score;
                        string str_key;

                        if(replace_sub){
                            while(data->faked_s_tuple_exist(random_Y_id, XRelKey.second, XRelKey.first)){
                                random_Y_id = RandomUtil::uniform_int(0, data->N);
                            }
                            score = cal_score(method, *parameter, random_Y_id, XRelKey.second, XRelKey.first, rescalA, rescalR, transeA, transeR, holeE, holeP, descriptor, ensembleWeight, min_max_values);
                            str_key = to_string(random_Y_id) + "," + to_string(XRelKey.second) + "," + to_string(XRelKey.first);
                        } else {
                            while(data->faked_o_tuple_exist(XRelKey.first, XRelKey.second, random_Y_id)){
                                random_Y_id = RandomUtil::uniform_int(0, data->N);
                            }
                            score = cal_score(method, *parameter, XRelKey.first, XRelKey.second, random_Y_id, rescalA, rescalR, transeA, transeR, holeE, holeP, descriptor, ensembleWeight, min_max_values);
                            str_key = to_string(XRelKey.first) + "," + to_string(XRelKey.second) + "," + to_string(random_Y_id);
                        }

                        replaceYResult[replaceYResultIndex] = make_pair(str_key, score);
                        replaceYResultIndex++;
                    }
                }

                // sort
                if(method == m_TransE){
                    std::sort(replaceYResult.begin(), replaceYResult.end(), &CompareUtil::pairLessCompare<string>);
                } else {
                    std::sort(replaceYResult.begin(), replaceYResult.end(), &CompareUtil::pairGreaterCompare<string>);
                }

                value_type map_value = 0;
                int true_key_found = 0;
                for (int list_index = 0; list_index < replaceYResult.size(); list_index++) {
                    if(trueKeys.find(replaceYResult[list_index].first)!=trueKeys.end()){
                        true_key_found++;
                        map_value += true_key_found / (list_index + 1.0);
                    }
                }

                replaced_map[thread_index] += map_value / true_key_found;
            }

        };

        ThreadUtil::execute_threads(compute_func, parameter->num_of_thread_eval);

        value_type global_map = 0;

        for(int thread_index=0;thread_index<parameter->num_of_thread_eval;thread_index++){
            global_map += replaced_map[thread_index];
        }

        return replace_sub ? (global_map/data->testObjRel2Sub.size()) : (global_map/data->testSubRel2Obj.size());
    }

    inline pair<value_type, value_type> eval_MAP(Method method, Parameter *parameter, Data *data, DenseMatrix *rescalA, vector<DenseMatrix> *rescalR,
                                                 DenseMatrix *transeA, DenseMatrix *transeR, DenseMatrix *holeE, DenseMatrix *holeP, DFTI_DESCRIPTOR_HANDLE *descriptor,
                                                 vector<SimpleWeight> *ensembleWeight, vector<min_max> *min_max_values=nullptr) {

        if(method==Method::m_HOLE){
            cerr << "EvaluationUtil: eval_hit_rate should transform HOLE to RESCAL_RANK first than call m_RESCAL_RANK for evaluation!" << endl;
            exit(1);
        } else if(method==Method::m_HTLREnsemble){
            cerr << "EvaluationUtil: eval_hit_rate should transform HOLE to RESCAL_RANK first than call m_RTLREnsemble for evaluation!" << endl;
            exit(1);
        }

        value_type global_sub_map = cal_MAP(method, parameter, data, true, rescalA, rescalR, transeA, transeR, holeE, holeP, descriptor, ensembleWeight, min_max_values);
        value_type global_obj_map = cal_MAP(method, parameter, data, false, rescalA, rescalR, transeA, transeR, holeE, holeP, descriptor, ensembleWeight, min_max_values);

        return make_pair(global_obj_map, global_sub_map);
    }

    set<pair<int, int> > sub_obj_set;
    map<pair<int, int>, int> id_mapping;
    bool init = false;

    inline hit_rate eval_relation_rescal(Parameter *parameter, Data *data, DenseMatrix &A, vector<DenseMatrix> &R, const string output_folder) {

        cerr << "There is a bug in eval_relation_rescal. Not fixed." << endl;
        exit(1);

        boost::filesystem::create_directories(output_folder);

        // thread -> (relation, count)
        vector<vector<ull> > rel_counts(parameter->num_of_thread_eval, vector<ull>(R.size(), 0));
        vector<vector<ull> > rel_counts_ranking(parameter->num_of_thread_eval, vector<ull>(R.size(), 0));
        vector<vector<ull> > rel_counts_filtering(parameter->num_of_thread_eval, vector<ull>(R.size(), 0));
        vector<vector<ull> > rel_counts_ranking_filtering(parameter->num_of_thread_eval, vector<ull>(R.size(), 0));

        int testing_size = (*data).testing_triple_strs.size();

        if (!init) {
            for (int relation_id = 0; relation_id < R.size(); relation_id++) {
                vector<Tuple<int> > &tuples = data->relation2tupleTestList_mapping[relation_id];
                for (auto tuple:tuples) {
                    sub_obj_set.insert(make_pair(tuple.subject, tuple.object));
                }
            }
            for (auto p:sub_obj_set) {
                id_mapping[p] = id_mapping.size();
            }
            init = true;
        }

        value_type *tmp_outer_product = cache_aligned_allocator<value_type>().allocate(id_mapping.size() * parameter->rescal_D * parameter->rescal_D);

        for (map<pair<int, int>, int>::const_iterator itr = id_mapping.begin();
             itr != id_mapping.end(); itr++) {
            kronecker_product(A.data().begin() + itr->first.second * parameter->rescal_D,
                              A.data().begin() + itr->first.first * parameter->rescal_D,
                              tmp_outer_product + itr->second * parameter->rescal_D * parameter->rescal_D, parameter->rescal_D);
        }

        for (int relation_id = 0; relation_id < R.size(); relation_id++) {
            vector<Tuple<int> > &tuples = data->relation2tupleTestList_mapping[relation_id];

            int size4relation = tuples.size();

            int workload = size4relation / parameter->num_of_thread_eval + ((size4relation % parameter->num_of_thread_eval == 0) ? 0 : 1);

            std::function<void(int)> compute_func = [&](int thread_index) -> void {

                int start = thread_index * workload;
                int end = std::min(start + workload, size4relation);

                vector<ull> &counts = rel_counts[thread_index];
                vector<ull> &counts_ranking = rel_counts_ranking[thread_index];

                vector<ull> &counts_filtering = rel_counts_filtering[thread_index];
                vector<ull> &counts_ranking_filtering = rel_counts_ranking_filtering[thread_index];

                // first: relation id, second: score
                vector<pair<int, value_type> > result(R.size());
                vector<pair<int, value_type> > result_filtering;

                for (int n = start; n < end; n++) {

                    Tuple<int> &test_tuple = tuples[n];
                    result_filtering.clear();

                    // replace relation_id with other relation_id
                    for (int replace_relation_id = 0; replace_relation_id < R.size(); replace_relation_id++) {
                        value_type score = cblas_xdot(parameter->rescal_D * parameter->rescal_D, tmp_outer_product +
                                                                                                 id_mapping[make_pair(test_tuple.subject,
                                                                                                                      test_tuple.object)] * parameter->rescal_D * parameter->rescal_D, 1,
                                                      R[replace_relation_id].data().begin(), 1);
                        result[replace_relation_id] = make_pair(replace_relation_id, score);
                        if ((replace_relation_id == relation_id) ||
                            (!data->faked_tuple_exist_test(replace_relation_id, test_tuple))) {
                            result_filtering.push_back(make_pair(replace_relation_id, score));
                        }
                    }

                    // sort
                    std::sort(result.begin(), result.end(), &CompareUtil::pairGreaterCompare<int>);
                    std::sort(result_filtering.begin(), result_filtering.end(), &CompareUtil::pairGreaterCompare<int>);

                    bool found = false;
                    bool found_filtering = false;

                    for (int i = 0; i < parameter->hit_rate_topk; i++) {

                        if (result[i].first == relation_id) {
                            counts[relation_id]++;
                            counts_ranking[relation_id] += i + 1;
                            found = true;
                        }

                        if (result_filtering[i].first == relation_id) {
                            counts_filtering[relation_id]++;
                            counts_ranking_filtering[relation_id] += i + 1;
                            found_filtering = true;
                        }
                    }

                    if (!found) {
                        for (int i = parameter->hit_rate_topk; i < R.size(); i++) {
                            if (result[i].first == relation_id) {
                                counts_ranking[relation_id] += i + 1;
                                found = true;
                                break;
                            }
                        }
                    }

                    if (!found_filtering) {
                        for (int i = parameter->hit_rate_topk; i < R.size(); i++) {
                            if (result_filtering[i].first == relation_id) {
                                counts_ranking_filtering[relation_id] += i + 1;
                                found_filtering = true;
                                break;
                            }
                        }
                    }

                    if ((!found) || (!found_filtering)) {
                        cerr << "logical error in eval_hit_rate! (1)" << endl;
                        exit(1);
                    }
                }
            };

            ThreadUtil::execute_threads(compute_func, parameter->num_of_thread_eval);

        }

        cache_aligned_allocator<value_type>().deallocate(tmp_outer_product, id_mapping.size() * parameter->rescal_D * parameter->rescal_D);

        hit_rate measure;
        measure.count_s = 0;
        measure.count_s_ranking = 0;
        measure.count_s_filtering = 0;
        measure.count_s_ranking_filtering = 0;

        vector<ull> rel(R.size(), 0);
        vector<ull> rel_ranking(R.size(), 0);

        vector<ull> rel_filtering(R.size(), 0);
        vector<ull> rel_ranking_filtering(R.size(), 0);

        for (int thread_id = 0; thread_id < parameter->num_of_thread_eval; thread_id++) {

            vector<ull> &counts = rel_counts[thread_id];
            vector<ull> &counts_ranking = rel_counts_ranking[thread_id];

            vector<ull> &counts_filtering = rel_counts_filtering[thread_id];
            vector<ull> &counts_ranking_filtering = rel_counts_ranking_filtering[thread_id];

            for (int relation_id = 0; relation_id < R.size(); relation_id++) {
                measure.count_s += counts[relation_id];
                measure.count_s_ranking += counts_ranking[relation_id];

                measure.count_s_filtering += counts_filtering[relation_id];
                measure.count_s_ranking_filtering += counts_ranking_filtering[relation_id];

                rel[relation_id] += counts[relation_id];
                rel_ranking[relation_id] += counts_ranking[relation_id];

                rel_filtering[relation_id] += counts_filtering[relation_id];
                rel_ranking_filtering[relation_id] += counts_ranking_filtering[relation_id];
            }
        }

        measure.count_s /= testing_size;
        measure.count_s_ranking /= testing_size;

        measure.count_s_filtering /= testing_size;
        measure.count_s_ranking_filtering /= testing_size;

        ofstream output(output_folder + "/relation_statistics.dat");

        output << "hit_rate_relation@" << parameter->hit_rate_topk << ": " << measure.count_s << endl;
        output << "relation_ranking" << parameter->hit_rate_topk << ": " << measure.count_s_ranking << endl;

        output << "hit_rate_relation_filter@" << parameter->hit_rate_topk << ": " << measure.count_s_filtering << endl;
        output << "relation_ranking_filter" << parameter->hit_rate_topk << ": " << measure.count_s_ranking_filtering << endl;

        output << "-------------------" << endl;
        for (int relation_id = 0; relation_id < R.size(); relation_id++) {
            value_type size = data->relation2tupleTestList_mapping[relation_id].size();
            output << "relation id: " << relation_id << endl;

            output << "hit_rate_relation@" << parameter->hit_rate_topk << ": " << rel[relation_id] / size << endl;
            output << "relation_ranking" << parameter->hit_rate_topk << ": " << rel_ranking[relation_id] / size << endl;

            output << "hit_rate_relation_filter@" << parameter->hit_rate_topk << ": " << rel_filtering[relation_id] / size << endl;
            output << "relation_ranking_filter" << parameter->hit_rate_topk << ": " << rel_ranking_filtering[relation_id] / size << endl;

            output << "-------------------" << endl;
        }

        output.close();

        return measure;

    }

    inline value_type eval_rescal_train(Parameter *parameter, Data *data, DenseMatrix &A, vector<DenseMatrix> &R) {

        value_type loss = 0;

        Sample sample;
        for (int i = 0; i < data->training_triples.size(); i++) {
            Sampler::random_sample(*data, sample, i);
            value_type positive_score = cal_rescal_score(sample.relation_id, sample.p_sub, sample.p_obj, parameter->rescal_D, A, R);
            value_type negative_score = cal_rescal_score(sample.relation_id, sample.n_sub, sample.n_obj, parameter->rescal_D, A, R);
            value_type tmp = positive_score - negative_score - parameter->margin;
            if(tmp < 0) {
                loss -= tmp;
            }
        }

        return loss;
    }

    inline hit_rate eval_rescal_train(Parameter *parameter, Data *data, DenseMatrix &A, vector<DenseMatrix> &R, const string output_folder) {

        boost::filesystem::create_directories(output_folder);

        // thread -> (relation, count)
        vector<vector<ull> > rel_counts_s(parameter->num_of_thread_eval, vector<ull>(R.size(), 0));
        vector<vector<ull> > rel_counts_o(parameter->num_of_thread_eval, vector<ull>(R.size(), 0));

        vector<vector<ull> > rel_counts_s_ranking(parameter->num_of_thread_eval, vector<ull>(R.size(), 0));
        vector<vector<ull> > rel_counts_o_ranking(parameter->num_of_thread_eval, vector<ull>(R.size(), 0));

        int sample_size = (*data).training_triple_strs.size() * parameter->train_sample_percentage;
        vector<int> sample_training_triple_ids(sample_size);

        unordered_map<int, vector<Tuple<int> > > relation2tupleList_mapping;

        for (int i = 0; i < sample_size; i++) {
            int random_id = RandomUtil::uniform_int(0, data->num_of_training_triples);
            Triple<int> &trainTriple = data->training_triples[random_id];
            if (relation2tupleList_mapping.find(trainTriple.relation) == relation2tupleList_mapping.end()) {
                relation2tupleList_mapping[trainTriple.relation] = vector<Tuple<int> >();
            }
            relation2tupleList_mapping[trainTriple.relation].push_back(Tuple<int>(trainTriple.subject, trainTriple.object));
        }

        value_type *AR = new value_type[data->N * parameter->rescal_D];
        value_type *RAT = new value_type[parameter->rescal_D * data->N];

        for (unordered_map<int, vector<Tuple<int> > >::iterator itr = relation2tupleList_mapping.begin();
             itr != relation2tupleList_mapping.end(); itr++) {

            int relation_id = itr->first;
            vector<Tuple<int> > &tuples = itr->second;

            // AR
            cblas_xgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        data->N, parameter->rescal_D, parameter->rescal_D, 1.0, A.data().begin(), parameter->rescal_D, R[relation_id].data().begin(), parameter->rescal_D, 0.0, AR, parameter->rescal_D);

            // RAT
            cblas_xgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        parameter->rescal_D, data->N, parameter->rescal_D, 1.0, R[relation_id].data().begin(), parameter->rescal_D, A.data().begin(), parameter->rescal_D, 0.0, RAT, data->N);

            int size4relation = tuples.size();

            int workload = size4relation / parameter->num_of_thread_eval + ((size4relation % parameter->num_of_thread_eval == 0) ? 0 : 1);

            vector<string> outputs(parameter->num_of_thread_eval, "");

            std::function<void(int)> compute_func = [&](int thread_index) -> void {

                int start = thread_index * workload;
                int end = std::min(start + workload, size4relation);

                vector<ull> &counts_s = rel_counts_s[thread_index];
                vector<ull> &counts_o = rel_counts_o[thread_index];
                vector<ull> &counts_s_ranking = rel_counts_s_ranking[thread_index];
                vector<ull> &counts_o_ranking = rel_counts_o_ranking[thread_index];

                // first: entity id, second: score
                vector<pair<int, value_type> > result(data->N);

                value_type *tmp = new value_type[data->N];

                for (int n = start; n < end; n++) {

                    outputs[thread_index].append("-------------------------\ntesting tuple: ");

                    Tuple<int> &test_tuple = tuples[n];

                    outputs[thread_index].append(data->entity_decoder[test_tuple.subject] + " " +
                                                 data->entity_decoder[test_tuple.object] + "\ntop" +
                                                 to_string(parameter->hit_rate_topk) + " for replacing subject:\n");

                    // first replace subject with other entities

                    // AR*A_jt
                    cblas_xgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                data->N, 1, parameter->rescal_D, 1.0, AR, parameter->rescal_D, A.data().begin() + test_tuple.object * parameter->rescal_D, 1, 0.0, tmp, 1);

                    for (int index = 0; index < data->N; index++) {
                        result[index] = make_pair(index, tmp[index]);
                    }

                    // sort
                    std::sort(result.begin(), result.end(), &CompareUtil::pairGreaterCompare<int>);

                    bool found = false;

                    for (int i = 0; i < parameter->hit_rate_topk; i++) {

                        outputs[thread_index].append(
                                data->entity_decoder[result[i].first] + " " + to_string(result[i].second) + "\n");

                        if (result[i].first == test_tuple.subject) {
                            counts_s[relation_id]++;
                            counts_s_ranking[relation_id] += i + 1;
                            found = true;
                        }
                    }

                    if (!found) {
                        for (int i = parameter->hit_rate_topk; i < data->N; i++) {
                            if (result[i].first == test_tuple.subject) {
                                counts_s_ranking[relation_id] += i + 1;
                                found = true;
                                break;
                            }
                        }
                    }

                    if (!found) {
                        cerr << "logical error in eval_hit_rate! (1)" << endl;
                        exit(1);
                    }

                    // then replace object with other entities

                    // A_i * RAT
                    cblas_xgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                1, data->N, parameter->rescal_D, 1.0, A.data().begin() + test_tuple.subject * parameter->rescal_D, parameter->rescal_D, RAT, data->N, 0.0, tmp, data->N);

                    for (int index = 0; index < data->N; index++) {
                        result[index] = make_pair(index, tmp[index]);
                    }

                    // sort
                    std::sort(result.begin(), result.end(), &CompareUtil::pairGreaterCompare<int>);

                    found = false;

                    outputs[thread_index].append("top" + to_string(parameter->hit_rate_topk) + " for replacing object:\n");

                    for (int i = 0; i < parameter->hit_rate_topk; i++) {

                        outputs[thread_index].append(
                                data->entity_decoder[result[i].first] + " " + to_string(result[i].second) + "\n");

                        if (result[i].first == test_tuple.object) {
                            counts_o[relation_id]++;
                            counts_o_ranking[relation_id] += i + 1;
                            found = true;
                        }
                    }

                    if (!found) {
                        for (int i = parameter->hit_rate_topk; i < data->N; i++) {
                            if (result[i].first == test_tuple.object) {
                                counts_o_ranking[relation_id] += i + 1;
                                found = true;
                                break;
                            }
                        }
                    }

                    if (!found) {
                        cerr << "logical error in eval_hit_rate! (2)" << endl;
                        exit(1);
                    }
                }

                delete[] tmp;
            };

            ThreadUtil::execute_threads(compute_func, parameter->num_of_thread_eval);

            ofstream output(output_folder + "/topk_results_relation_" + to_string(relation_id) + ".dat");
            for (auto s:outputs) {
                output << s;
            }
            output.close();
        }

        delete[] AR;
        delete[] RAT;

        hit_rate measure;
        measure.count_s = 0;
        measure.count_o = 0;
        measure.count_s_ranking = 0;
        measure.count_o_ranking = 0;

        vector<ull> rel_s(R.size(), 0);
        vector<ull> rel_o(R.size(), 0);
        vector<ull> rel_s_ranking(R.size(), 0);
        vector<ull> rel_o_ranking(R.size(), 0);

        for (int thread_id = 0; thread_id < parameter->num_of_thread_eval; thread_id++) {

            vector<ull> &counts_s = rel_counts_s[thread_id];
            vector<ull> &counts_o = rel_counts_o[thread_id];
            vector<ull> &counts_s_ranking = rel_counts_s_ranking[thread_id];
            vector<ull> &counts_o_ranking = rel_counts_o_ranking[thread_id];

            for (int relation_id = 0; relation_id < R.size(); relation_id++) {
                measure.count_s += counts_s[relation_id];
                measure.count_o += counts_o[relation_id];
                measure.count_s_ranking += counts_s_ranking[relation_id];
                measure.count_o_ranking += counts_o_ranking[relation_id];

                rel_s[relation_id] += counts_s[relation_id];
                rel_o[relation_id] += counts_o[relation_id];
                rel_s_ranking[relation_id] += counts_s_ranking[relation_id];
                rel_o_ranking[relation_id] += counts_o_ranking[relation_id];
            }
        }

        measure.count_s /= sample_size;
        measure.count_o /= sample_size;
        measure.count_s_ranking /= sample_size;
        measure.count_o_ranking /= sample_size;

        ofstream output(output_folder + "/statistics.dat");

        output << "hit_rate_subject@" << parameter->hit_rate_topk << ": " << measure.count_s << endl;
        output << "hit_rate_object@" << parameter->hit_rate_topk << ": " << measure.count_o << endl;
        output << "subject_ranking" << parameter->hit_rate_topk << ": " << measure.count_s_ranking << endl;
        output << "object_ranking" << parameter->hit_rate_topk << ": " << measure.count_o_ranking << endl;

        output << "-------------------" << endl;
        for (int relation_id = 0; relation_id < R.size(); relation_id++) {
            value_type size = data->relation2tupleTestList_mapping[relation_id].size();
            output << "relation id: " << relation_id << endl;
            output << "hit_rate_subject@" << parameter->hit_rate_topk << ": " << rel_s[relation_id] / size << endl;
            output << "hit_rate_object@" << parameter->hit_rate_topk << ": " << rel_o[relation_id] / size << endl;
            output << "subject_ranking" << parameter->hit_rate_topk << ": " << rel_s_ranking[relation_id] / size << endl;
            output << "object_ranking" << parameter->hit_rate_topk << ": " << rel_o_ranking[relation_id] / size << endl;
            output << "-------------------" << endl;
        }

        output.close();

        return measure;

    }

    inline hit_rate eval_relation_transe(Parameter *parameter, Data *data, DenseMatrix &transeA, DenseMatrix &transeR, const string output_folder) {

        boost::filesystem::create_directories(output_folder);

        // thread -> (relation, count)
        vector<vector<ull> > rel_counts(parameter->num_of_thread_eval, vector<ull>(data->K, 0));

        vector<vector<ull> > rel_counts_ranking(parameter->num_of_thread_eval, vector<ull>(data->K, 0));

        vector<vector<ull> > rel_counts_filtering(parameter->num_of_thread_eval, vector<ull>(data->K, 0));

        vector<vector<ull> > rel_counts_ranking_filtering(parameter->num_of_thread_eval, vector<ull>(data->K, 0));

        int testing_size = data->testing_triple_strs.size();

        for (int relation_id = 0; relation_id < data->K; relation_id++) {
            vector<Tuple<int> > &tuples = data->relation2tupleTestList_mapping[relation_id];

            int size4relation = tuples.size();
            int workload = size4relation / parameter->num_of_thread_eval + ((size4relation % parameter->num_of_thread_eval == 0) ? 0 : 1);

            std::function<void(int)> compute_func = [&](int thread_index) -> void {

                int start = thread_index * workload;
                int end = std::min(start + workload, size4relation);

                vector<ull> &counts = rel_counts[thread_index];
                vector<ull> &counts_ranking = rel_counts_ranking[thread_index];

                vector<ull> &counts_filtering = rel_counts_filtering[thread_index];
                vector<ull> &counts_ranking_filtering = rel_counts_ranking_filtering[thread_index];

                // first: entity id, second: score
                vector<pair<int, value_type>> result(data->K);
                vector<pair<int, value_type> > result_filtering;

                for (int n = start; n < end; n++) {

                    Tuple<int> &test_tuple = tuples[n];

                    result_filtering.clear();
                    // replace relation_id with other relation_id
                    for (int replace_relation_id = 0; replace_relation_id < data->K; replace_relation_id++) {
                        value_type dis = cal_transe_score(test_tuple.subject, test_tuple.object, replace_relation_id, parameter->transe_D, parameter->L1_flag, transeA, transeR);

                        result[replace_relation_id] = make_pair(replace_relation_id, dis);

                        if ((replace_relation_id == relation_id) ||
                            (!data->faked_tuple_exist_test(replace_relation_id, test_tuple))) {
                            result_filtering.push_back(make_pair(replace_relation_id, dis));
                        }
                    }

                    // sort
                    std::sort(result.begin(), result.end(), &CompareUtil::pairLessCompare<int>);
                    std::sort(result_filtering.begin(), result_filtering.end(), &CompareUtil::pairLessCompare<int>);

                    bool found = false;
                    bool found_filtering = false;

                    for (int i = 0; i < parameter->hit_rate_topk; i++) {

                        if (result[i].first == relation_id) {
                            counts[relation_id]++;
                            counts_ranking[relation_id] += i + 1;
                            found = true;
                        }

                        if (result_filtering[i].first == relation_id) {
                            counts_filtering[relation_id]++;
                            counts_ranking_filtering[relation_id] += i + 1;
                            found_filtering = true;
                        }
                    }

                    if (!found) {
                        for (int i = parameter->hit_rate_topk; i < data->K; i++) {
                            if (result[i].first == relation_id) {
                                counts_ranking[relation_id] += i + 1;
                                found = true;
                                break;
                            }
                        }
                    }

                    if (!found_filtering) {
                        for (int i = parameter->hit_rate_topk; i < data->K; i++) {
                            if (result_filtering[i].first == relation_id) {
                                counts_ranking_filtering[relation_id] += i + 1;
                                found_filtering = true;
                                break;
                            }
                        }
                    }

                    if ((!found) || (!found_filtering)) {
                        cerr << "logical error in eval_hit_rate! (1)" << endl;
                        exit(1);
                    }

                }
            };

            ThreadUtil::execute_threads(compute_func, parameter->num_of_thread_eval);
        }

        hit_rate measure;
        measure.count_s = 0;
        measure.count_s_ranking = 0;
        measure.count_s_filtering = 0;
        measure.count_s_ranking_filtering = 0;

        vector<ull> rel(data->K, 0);
        vector<ull> rel_ranking(data->K, 0);

        vector<ull> rel_filtering(data->K, 0);
        vector<ull> rel_ranking_filtering(data->K, 0);

        for (int thread_id = 0; thread_id < parameter->num_of_thread_eval; thread_id++) {

            vector<ull> &counts = rel_counts[thread_id];
            vector<ull> &counts_ranking = rel_counts_ranking[thread_id];

            vector<ull> &counts_filtering = rel_counts_filtering[thread_id];
            vector<ull> &counts_ranking_filtering = rel_counts_ranking_filtering[thread_id];

            for (int relation_id = 0; relation_id < data->K; relation_id++) {
                measure.count_s += counts[relation_id];
                measure.count_s_ranking += counts_ranking[relation_id];

                measure.count_s_filtering += counts_filtering[relation_id];
                measure.count_s_ranking_filtering += counts_ranking_filtering[relation_id];

                rel[relation_id] += counts[relation_id];
                rel_ranking[relation_id] += counts_ranking[relation_id];

                rel_filtering[relation_id] += counts_filtering[relation_id];
                rel_ranking_filtering[relation_id] += counts_ranking_filtering[relation_id];
            }
        }

        measure.count_s /= testing_size;
        measure.count_s_ranking /= testing_size;

        measure.count_s_filtering /= testing_size;
        measure.count_s_ranking_filtering /= testing_size;

        ofstream output(output_folder + "/relation_statistics.dat");

        output << "hit_rate_relation@" << parameter->hit_rate_topk << ": " << measure.count_s << endl;
        output << "relation_ranking" << parameter->hit_rate_topk << ": " << measure.count_s_ranking << endl;

        output << "hit_rate_relation_filter@" << parameter->hit_rate_topk << ": " << measure.count_s_filtering << endl;
        output << "relation_ranking_filter" << parameter->hit_rate_topk << ": " << measure.count_s_ranking_filtering << endl;

        output << "-------------------" << endl;
        for (int relation_id = 0; relation_id < data->K; relation_id++) {
            value_type size = data->relation2tupleTestList_mapping[relation_id].size();
            output << "relation id: " << relation_id << endl;

            output << "hit_rate_relation@" << parameter->hit_rate_topk << ": " << rel[relation_id] / size << endl;
            output << "relation_ranking" << parameter->hit_rate_topk << ": " << rel_ranking[relation_id] / size << endl;

            output << "hit_rate_relation_filter@" << parameter->hit_rate_topk << ": " << rel_filtering[relation_id] / size << endl;
            output << "relation_ranking_filter" << parameter->hit_rate_topk << ": " << rel_ranking_filtering[relation_id] / size << endl;

            output << "-------------------" << endl;
        }

        output.close();

        return measure;
    }

    inline hit_rate eval_transe_train(Parameter *parameter, Data *data, DenseMatrix &transeA, DenseMatrix &transeR, const string output_folder) {

        boost::filesystem::create_directories(output_folder);

        // thread -> (relation, count)
        vector<vector<ull> > rel_counts_s(parameter->num_of_thread_eval, vector<ull>(data->K, 0));
        vector<vector<ull> > rel_counts_o(parameter->num_of_thread_eval, vector<ull>(data->K, 0));

        vector<vector<ull> > rel_counts_s_ranking(parameter->num_of_thread_eval, vector<ull>(data->K, 0));
        vector<vector<ull> > rel_counts_o_ranking(parameter->num_of_thread_eval, vector<ull>(data->K, 0));

        int sample_size = (*data).training_triple_strs.size() * parameter->train_sample_percentage;
        vector<int> sample_training_triple_ids(sample_size);

        unordered_map<int, vector<Tuple<int> > > relation2tupleList_mapping;

        for (int i = 0; i < sample_size; i++) {
            int random_id = RandomUtil::uniform_int(0, data->num_of_training_triples);
            Triple<int> &trainTriple = data->training_triples[random_id];
            if (relation2tupleList_mapping.find(trainTriple.relation) == relation2tupleList_mapping.end()) {
                relation2tupleList_mapping[trainTriple.relation] = vector<Tuple<int> >();
            }
            relation2tupleList_mapping[trainTriple.relation].push_back(Tuple<int>(trainTriple.subject, trainTriple.object));
        }

        for(unordered_map<int, vector<Tuple<int> > >::iterator itr = relation2tupleList_mapping.begin(); itr!=relation2tupleList_mapping.end();itr++){

            int relation_id = itr->first;
            vector<Tuple<int> > &tuples = itr->second;

            int size4relation = tuples.size();
            int workload = size4relation / parameter->num_of_thread_eval + ((size4relation % parameter->num_of_thread_eval == 0) ? 0 : 1);

            vector<string> outputs(parameter->num_of_thread_eval, "");

            std::function<void(int)> compute_func = [&](int thread_index) -> void {

                int start = thread_index * workload;
                int end = std::min(start + workload, size4relation);

                vector<ull> &counts_s = rel_counts_s[thread_index];
                vector<ull> &counts_o = rel_counts_o[thread_index];
                vector<ull> &counts_s_ranking = rel_counts_s_ranking[thread_index];
                vector<ull> &counts_o_ranking = rel_counts_o_ranking[thread_index];

                // first: entity id, second: score
                vector<pair<int, value_type>> result(data->N);

                for (int n = start; n < end; n++) {

                    outputs[thread_index].append("-------------------------\ntesting tuple: ");

                    Tuple<int> &test_tuple = tuples[n];

                    outputs[thread_index].append(data->entity_decoder[test_tuple.subject] + " " + data->entity_decoder[test_tuple.object] + "\ntop" + to_string(parameter->hit_rate_topk) + " for replacing subject:\n");

                    // first replace subject with other entities
                    for (int i = 0; i < data->N; i++) {
                        result[i] = make_pair(i, cal_transe_score(i, test_tuple.object, relation_id, parameter->transe_D, parameter->L1_flag, transeA, transeR));
                    }

                    // sort
                    std::sort(result.begin(), result.end(), &CompareUtil::pairLessCompare<int>);

                    bool found = false;

                    for (int i = 0; i < parameter->hit_rate_topk; i++) {

                        outputs[thread_index].append(data->entity_decoder[result[i].first] + " " + to_string(result[i].second) + "\n");

                        if (result[i].first == test_tuple.subject) {
                            counts_s[relation_id]++;
                            counts_s_ranking[relation_id] += i + 1;
                            found = true;
                        }
                    }

                    if(!found) {
                        for (int i = parameter->hit_rate_topk; i < data->N; i++) {
                            if (result[i].first == test_tuple.subject) {
                                counts_s_ranking[relation_id] += i + 1;
                                found = true;
                                break;
                            }
                        }
                    }

                    if (!found) {
                        cerr << "logical error in eval_hit_rate! (1)" << endl;
                        exit(1);
                    }

                    // then replace object with other entities
                    for (int i = 0; i < data->N; i++) {
                        result[i] = make_pair(i, cal_transe_score(test_tuple.subject, i, relation_id, parameter->transe_D, parameter->L1_flag, transeA, transeR));
                    }

                    // sort
                    std::sort(result.begin(), result.end(), &CompareUtil::pairLessCompare<int>);

                    found = false;

                    for (int i = 0; i < parameter->hit_rate_topk; i++) {

                        outputs[thread_index].append(data->entity_decoder[result[i].first] + " " + to_string(result[i].second) + "\n");

                        if (result[i].first == test_tuple.object) {
                            counts_o[relation_id]++;
                            counts_o_ranking[relation_id] += i + 1;
                            found = true;
                        }
                    }

                    if(!found) {
                        for (int i = parameter->hit_rate_topk; i < data->N; i++) {
                            if (result[i].first == test_tuple.object) {
                                counts_o_ranking[relation_id] += i + 1;
                                found = true;
                                break;
                            }
                        }
                    }

                    if (!found) {
                        cerr << "logical error in eval_hit_rate! (2)" << endl;
                        exit(1);
                    }
                }
            };

            ThreadUtil::execute_threads(compute_func, parameter->num_of_thread_eval);

            ofstream output(output_folder + "/topk_results_relation_" + to_string(relation_id) + ".dat");
            for(auto s:outputs){
                output << s;
            }
            output.close();
        }

        hit_rate measure;
        measure.count_s = 0;
        measure.count_o = 0;
        measure.count_s_ranking = 0;
        measure.count_o_ranking = 0;

        vector<ull> rel_s(data->K, 0);
        vector<ull> rel_o(data->K, 0);
        vector<ull> rel_s_ranking(data->K, 0);
        vector<ull> rel_o_ranking(data->K, 0);

        for (int thread_id = 0; thread_id < parameter->num_of_thread_eval; thread_id++) {

            vector<ull> &counts_s = rel_counts_s[thread_id];
            vector<ull> &counts_o = rel_counts_o[thread_id];
            vector<ull> &counts_s_ranking = rel_counts_s_ranking[thread_id];
            vector<ull> &counts_o_ranking = rel_counts_o_ranking[thread_id];

            for (int relation_id = 0; relation_id < data->K; relation_id++) {
                measure.count_s += counts_s[relation_id];
                measure.count_o += counts_o[relation_id];
                measure.count_s_ranking += counts_s_ranking[relation_id];
                measure.count_o_ranking += counts_o_ranking[relation_id];

                rel_s[relation_id] += counts_s[relation_id];
                rel_o[relation_id] += counts_o[relation_id];
                rel_s_ranking[relation_id] += counts_s_ranking[relation_id];
                rel_o_ranking[relation_id] += counts_o_ranking[relation_id];
            }
        }

        measure.count_s /= sample_size;
        measure.count_o /= sample_size;
        measure.count_s_ranking /= sample_size;
        measure.count_o_ranking /= sample_size;

        ofstream output(output_folder + "/statistics.dat");

        output << "hit_rate_subject@" << parameter->hit_rate_topk << ": " << measure.count_s << endl;
        output << "hit_rate_object@" << parameter->hit_rate_topk << ": " << measure.count_o << endl;
        output << "subject_ranking" << parameter->hit_rate_topk << ": " << measure.count_s_ranking << endl;
        output << "object_ranking" << parameter->hit_rate_topk << ": " << measure.count_o_ranking << endl;

        output << "-------------------" << endl;
        for(int relation_id = 0; relation_id < data->K; relation_id++){
            value_type size = data->relation2tupleTestList_mapping[relation_id].size();
            output << "relation id: " << relation_id << endl;
            output << "hit_rate_subject@" << parameter->hit_rate_topk << ": " << rel_s[relation_id] / size << endl;
            output << "hit_rate_object@" << parameter->hit_rate_topk << ": " << rel_o[relation_id] / size << endl;
            output << "subject_ranking" << parameter->hit_rate_topk << ": " << rel_s_ranking[relation_id] / size << endl;
            output << "object_ranking" << parameter->hit_rate_topk << ": " << rel_o_ranking[relation_id] / size << endl;
            output << "-------------------" << endl;
        }

        output.close();

        return measure;
    }
}

#endif //EVALUATIONUTIL_H