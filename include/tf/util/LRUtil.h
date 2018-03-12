#ifndef LRUTIL_H
#define LRUTIL_H

#include "tf/util/Base.h"
#include "tf/struct/Weight.h"
#include "tf/struct/Min_Max.h"
#include "Parameter.h"
#include "Data.h"
#include "tf/liblinear/linear.cpp"

namespace LRUtil{

    // RESCAL + TransE
    inline void learn_weights_RT(Data *data, Parameter *_parameter, DenseMatrix &rescal_A, vector<DenseMatrix> &rescal_R,
                              DenseMatrix &transe_A, DenseMatrix &transe_R, vector<SimpleWeight> &weights, vector<min_max> &min_max_values) {

        weights.resize(data->K);

        if (_parameter->normalize) {
            min_max_values.resize(data->K);
        }

        parameter param;
        param.solver_type = L2R_LR;
        param.C = _parameter->c;
        param.eps = 1e-4;

        if(_parameter->num_of_thread==-1) {
            param.nr_thread = omp_get_num_procs();
        } else {
            param.nr_thread = _parameter->num_of_thread;
        }
        cout << "number of thread for logistic regression: " << param.nr_thread << endl;

        param.nr_weight = 0;
        param.weight_label = NULL;
        param.weight = NULL;
        param.init_sol = NULL;

        // if parameter->normalize, it is min and max. if parameter->znormalize, it is mean and std
        value_type rescal_min = 0;
        value_type transe_min = 0;
        value_type rescal_max = 0;
        value_type transe_max = 0;

        vector<value_type > rescal_scores;
        vector<value_type > transe_scores;

        Monitor timer;
        timer.start();

        for (int rel_id = 0; rel_id < data->K; rel_id++) {

            vector<Tuple<int> > &tuples = data->relation2tupleList_mapping[rel_id];
            int size = tuples.size() * 4 * _parameter->num_of_duplicated_true_triples;

            // `l' is the number of training data. If bias >= 0, we assume
            // that one additional feature is added to the end of each data
            // instance. `n' is the number of feature (including the bias feature
            // if bias >= 0).
            problem prob;
            prob.bias = 1;
            prob.l = size;
            prob.n = 3;
            prob.y = new double[prob.l];

            prob.x = new feature_node* [prob.l];

            int index1 = 0;
            int index2 = 0;

            if(_parameter->normalize) {
                rescal_min = std::numeric_limits<value_type>::max();
                transe_min = rescal_min;
                rescal_max = std::numeric_limits<value_type>::min();
                transe_max = rescal_max;
            } else if (_parameter->znormalize) {
                rescal_scores.resize(size);
                transe_scores.resize(size);
            }

            int score_index = 0;

            for (int tuple_index = 0; tuple_index < tuples.size(); tuple_index++) {

                value_type p_rescal_score = cal_rescal_score(rel_id, tuples[tuple_index].subject, tuples[tuple_index].object, _parameter->rescal_D, rescal_A, rescal_R);
                value_type p_transe_score = cal_transe_score(tuples[tuple_index].subject, tuples[tuple_index].object, rel_id, _parameter->transe_D, _parameter->L1_flag, transe_A, transe_R);

                if(_parameter->normalize) {
                    rescal_max = std::max(rescal_max, p_rescal_score);
                    rescal_min = std::min(rescal_min, p_rescal_score);
                    transe_max = std::max(transe_max, p_transe_score);
                    transe_min = std::min(transe_min, p_transe_score);
                } else if (_parameter->znormalize) {
                    rescal_scores[score_index] = p_rescal_score;
                    transe_scores[score_index] = p_transe_score;
                    score_index++;
                }

                for (int count = 0; count < 2 * _parameter->num_of_duplicated_true_triples; count++) {

                    prob.x[index1] = new feature_node[prob.n + 1];

                    prob.x[index1][0].index = 1;  // index should start from 1
                    prob.x[index1][0].value = p_rescal_score;

                    prob.x[index1][1].index = 2;
                    prob.x[index1][1].value = p_transe_score;

                    prob.x[index1][2].index = 3;
                    prob.x[index1][2].value = prob.bias;

                    prob.x[index1][3].index = -1; //Each row of properties should be terminated with a -1 according to the readme

                    index1++;

                    prob.y[index2] = 1;
                    index2++;
                }

                // first replace subjects
                for (int count = 0; count < _parameter->num_of_duplicated_true_triples; count++) {
                    int random_entity_id = RandomUtil::uniform_int(0, data->N);
                    while(data->faked_tuple_exist_train(rel_id, random_entity_id, tuples[tuple_index].object)){
                        random_entity_id = RandomUtil::uniform_int(0, data->N);
                    }

                    value_type n_rescal_score = cal_rescal_score(rel_id, random_entity_id, tuples[tuple_index].object, _parameter->rescal_D, rescal_A, rescal_R);
                    value_type n_transe_score = cal_transe_score(random_entity_id, tuples[tuple_index].object, rel_id, _parameter->transe_D, _parameter->L1_flag, transe_A, transe_R);

                    if(_parameter->normalize) {
                        rescal_max = std::max(rescal_max, n_rescal_score);
                        rescal_min = std::min(rescal_min, n_rescal_score);
                        transe_max = std::max(transe_max, n_transe_score);
                        transe_min = std::min(transe_min, n_transe_score);
                    } else if (_parameter->znormalize) {
                        rescal_scores[score_index] = n_rescal_score;
                        transe_scores[score_index] = n_transe_score;
                        score_index++;
                    }

                    prob.x[index1] = new feature_node[prob.n + 1];

                    prob.x[index1][0].index = 1;
                    prob.x[index1][0].value = n_rescal_score;

                    prob.x[index1][1].index = 2;
                    prob.x[index1][1].value = n_transe_score;

                    prob.x[index1][2].index = 3;
                    prob.x[index1][2].value = prob.bias;

                    prob.x[index1][3].index = -1; //Each row of properties should be terminated with a -1 according to the readme

                    index1++;

                    prob.y[index2] = 0;
                    index2++;
                }

                // then replace objects
                for (int count = 0; count < _parameter->num_of_duplicated_true_triples; count++) {
                    int random_entity_id = RandomUtil::uniform_int(0, data->N);
                    while(data->faked_tuple_exist_train(rel_id, tuples[tuple_index].subject, random_entity_id)){
                        random_entity_id = RandomUtil::uniform_int(0, data->N);
                    }

                    value_type n_rescal_score = cal_rescal_score(rel_id, tuples[tuple_index].subject, random_entity_id, _parameter->rescal_D, rescal_A, rescal_R);
                    value_type n_transe_score = cal_transe_score(tuples[tuple_index].subject, random_entity_id, rel_id, _parameter->transe_D, _parameter->L1_flag, transe_A, transe_R);

                    if(_parameter->normalize) {
                        rescal_max = std::max(rescal_max, n_rescal_score);
                        rescal_min = std::min(rescal_min, n_rescal_score);
                        transe_max = std::max(transe_max, n_transe_score);
                        transe_min = std::min(transe_min, n_transe_score);
                    } else if (_parameter->znormalize) {
                        rescal_scores[score_index] = n_rescal_score;
                        transe_scores[score_index] = n_transe_score;
                        score_index++;
                    }

                    prob.x[index1] = new feature_node[prob.n + 1];

                    prob.x[index1][0].index = 1;
                    prob.x[index1][0].value = n_rescal_score;

                    prob.x[index1][1].index = 2;
                    prob.x[index1][1].value = n_transe_score;

                    prob.x[index1][2].index = 3;
                    prob.x[index1][2].value = prob.bias;

                    prob.x[index1][3].index = -1; //Each row of properties should be terminated with a -1 according to the readme

                    index1++;

                    prob.y[index2] = 0;
                    index2++;
                }
            }

            if(_parameter->normalize) {
                value_type rescal_range = rescal_max - rescal_min;
                value_type transe_range = transe_max - transe_min;

                for (int i = 0; i < index1; i++) {
                    prob.x[i][0].value = (prob.x[i][0].value - rescal_min) / rescal_range;
                    prob.x[i][1].value = (prob.x[i][1].value - transe_min) / transe_range;
                }

                min_max_values[rel_id].max1 = rescal_max;
                min_max_values[rel_id].min1 = rescal_min;

                min_max_values[rel_id].max2 = transe_max;
                min_max_values[rel_id].min2 = transe_min;

            } else if (_parameter->znormalize) {

                cal_mean_std(rescal_scores, rescal_min, rescal_max);
                cal_mean_std(transe_scores, transe_min, transe_max);

                for (int i = 0; i < index1; i++) {
                    prob.x[i][0].value = (prob.x[i][0].value - rescal_min) / rescal_max;
                    prob.x[i][1].value = (prob.x[i][1].value - transe_min) / transe_max;
                }

                min_max_values[rel_id].max1 = rescal_max;
                min_max_values[rel_id].min1 = rescal_min;

                min_max_values[rel_id].max2 = transe_max;
                min_max_values[rel_id].min2 = transe_min;
            }

            model *model_ = train(&prob, &param);

            weights[rel_id] = SimpleWeight(model_->w[0], model_->w[1]);

            free_and_destroy_model(&model_);

            for (int i = 0; i < prob.l; i++) {
                delete[] prob.x[i];
            }

            delete[] prob.x;
            delete[] prob.y;


        }

        destroy_param(&param);

        timer.stop();
        cout << "time for learning weight: " << timer.getElapsedTime() << " secs" << endl;
    }

    // RESCAL + HOLE
    inline void learn_weights_RH(Data *data, Parameter *_parameter, DenseMatrix &rescal_A, vector<DenseMatrix> &rescal_R,
                                 DenseMatrix &hole_E, DenseMatrix &hole_P, vector<SimpleWeight> &weights, DFTI_DESCRIPTOR_HANDLE &descriptor, vector<min_max> &min_max_values) {

        weights.resize(data->K);

        if (_parameter->normalize) {
            min_max_values.resize(data->K);
        }

        parameter param;
        param.solver_type = L2R_LR;
        param.C = _parameter->c;
        param.eps = 1e-4;

        if(_parameter->num_of_thread==-1) {
            param.nr_thread = omp_get_num_procs();
        } else {
            param.nr_thread = _parameter->num_of_thread;
        }
        cout << "number of thread for logistic regression: " << param.nr_thread << endl;

        param.nr_weight = 0;
        param.weight_label = NULL;
        param.weight = NULL;
        param.init_sol = NULL;

        // if parameter->normalize, it is min and max. if parameter->znormalize, it is mean and std
        value_type rescal_min = 0;
        value_type hole_min = 0;
        value_type rescal_max = 0;
        value_type hole_max = 0;

        vector<value_type > rescal_scores;
        vector<value_type > hole_scores;

        Monitor timer;
        timer.start();

        for (int rel_id = 0; rel_id < data->K; rel_id++) {

            vector<Tuple<int> > &tuples = data->relation2tupleList_mapping[rel_id];
            int size = tuples.size() * 4 * _parameter->num_of_duplicated_true_triples;

            // `l' is the number of training data. If bias >= 0, we assume
            // that one additional feature is added to the end of each data
            // instance. `n' is the number of feature (including the bias feature
            // if bias >= 0).
            problem prob;
            prob.bias = 1;
            prob.l = size;
            prob.n = 3;
            prob.y = new double[prob.l];

            prob.x = new feature_node* [prob.l];

            int index1 = 0;
            int index2 = 0;

            if(_parameter->normalize) {
                rescal_min = std::numeric_limits<value_type>::max();
                hole_min = rescal_min;
                rescal_max = std::numeric_limits<value_type>::min();
                hole_max = rescal_max;
            } else if (_parameter->znormalize) {
                rescal_scores.resize(size);
                hole_scores.resize(size);
            }

            int score_index = 0;

            for (int tuple_index = 0; tuple_index < tuples.size(); tuple_index++) {

                value_type p_rescal_score = cal_rescal_score(rel_id, tuples[tuple_index].subject, tuples[tuple_index].object, _parameter->rescal_D, rescal_A, rescal_R);
                value_type p_hole_score = cal_hole_score(tuples[tuple_index].subject, tuples[tuple_index].object, rel_id, _parameter->hole_D, hole_E, hole_P, descriptor, false);

                if(_parameter->normalize) {
                    rescal_max = std::max(rescal_max, p_rescal_score);
                    rescal_min = std::min(rescal_min, p_rescal_score);
                    hole_max = std::max(hole_max, p_hole_score);
                    hole_min = std::min(hole_min, p_hole_score);
                } else if (_parameter->znormalize) {
                    rescal_scores[score_index] = p_rescal_score;
                    hole_scores[score_index] = p_hole_score;
                    score_index++;
                }

                for (int count = 0; count < 2 * _parameter->num_of_duplicated_true_triples; count++) {

                    prob.x[index1] = new feature_node[prob.n + 1];

                    prob.x[index1][0].index = 1;  // index should start from 1
                    prob.x[index1][0].value = p_rescal_score;

                    prob.x[index1][1].index = 2;
                    prob.x[index1][1].value = p_hole_score;

                    prob.x[index1][2].index = 3;
                    prob.x[index1][2].value = prob.bias;

                    prob.x[index1][3].index = -1; //Each row of properties should be terminated with a -1 according to the readme

                    index1++;

                    prob.y[index2] = 1;
                    index2++;
                }

                // first replace subjects
                for (int count = 0; count < _parameter->num_of_duplicated_true_triples; count++) {
                    int random_entity_id = RandomUtil::uniform_int(0, data->N);
                    while(data->faked_tuple_exist_train(rel_id, random_entity_id, tuples[tuple_index].object)){
                        random_entity_id = RandomUtil::uniform_int(0, data->N);
                    }

                    value_type n_rescal_score = cal_rescal_score(rel_id, random_entity_id, tuples[tuple_index].object, _parameter->rescal_D, rescal_A, rescal_R);
                    value_type n_hole_score = cal_hole_score(random_entity_id, tuples[tuple_index].object, rel_id, _parameter->hole_D, hole_E, hole_P, descriptor, false);

                    if(_parameter->normalize) {
                        rescal_max = std::max(rescal_max, n_rescal_score);
                        rescal_min = std::min(rescal_min, n_rescal_score);
                        hole_max = std::max(hole_max, n_hole_score);
                        hole_min = std::min(hole_min, n_hole_score);
                    } else if (_parameter->znormalize) {
                        rescal_scores[score_index] = n_rescal_score;
                        hole_scores[score_index] = n_hole_score;
                        score_index++;
                    }

                    prob.x[index1] = new feature_node[prob.n + 1];

                    prob.x[index1][0].index = 1;
                    prob.x[index1][0].value = n_rescal_score;

                    prob.x[index1][1].index = 2;
                    prob.x[index1][1].value = n_hole_score;

                    prob.x[index1][2].index = 3;
                    prob.x[index1][2].value = prob.bias;

                    prob.x[index1][3].index = -1; //Each row of properties should be terminated with a -1 according to the readme

                    index1++;

                    prob.y[index2] = 0;
                    index2++;
                }

                // then replace objects
                for (int count = 0; count < _parameter->num_of_duplicated_true_triples; count++) {
                    int random_entity_id = RandomUtil::uniform_int(0, data->N);
                    while(data->faked_tuple_exist_train(rel_id, tuples[tuple_index].subject, random_entity_id)){
                        random_entity_id = RandomUtil::uniform_int(0, data->N);
                    }

                    value_type n_rescal_score = cal_rescal_score(rel_id, tuples[tuple_index].subject, random_entity_id, _parameter->rescal_D, rescal_A, rescal_R);
                    value_type n_hole_score = cal_hole_score(tuples[tuple_index].subject, random_entity_id, rel_id, _parameter->hole_D, hole_E, hole_P, descriptor, false);

                    if(_parameter->normalize) {
                        rescal_max = std::max(rescal_max, n_rescal_score);
                        rescal_min = std::min(rescal_min, n_rescal_score);
                        hole_max = std::max(hole_max, n_hole_score);
                        hole_min = std::min(hole_min, n_hole_score);
                    } else if (_parameter->znormalize) {
                        rescal_scores[score_index] = n_rescal_score;
                        hole_scores[score_index] = n_hole_score;
                        score_index++;
                    }

                    prob.x[index1] = new feature_node[prob.n + 1];

                    prob.x[index1][0].index = 1;
                    prob.x[index1][0].value = n_rescal_score;

                    prob.x[index1][1].index = 2;
                    prob.x[index1][1].value = n_hole_score;

                    prob.x[index1][2].index = 3;
                    prob.x[index1][2].value = prob.bias;

                    prob.x[index1][3].index = -1; //Each row of properties should be terminated with a -1 according to the readme

                    index1++;

                    prob.y[index2] = 0;
                    index2++;
                }
            }

            if(_parameter->normalize) {
                value_type rescal_range = rescal_max - rescal_min;
                value_type hole_range = hole_max - hole_min;

                for (int i = 0; i < index1; i++) {
                    prob.x[i][0].value = (prob.x[i][0].value - rescal_min) / rescal_range;
                    prob.x[i][1].value = (prob.x[i][1].value - hole_min) / hole_range;
                }

                min_max_values[rel_id].max1 = rescal_max;
                min_max_values[rel_id].min1 = rescal_min;
                min_max_values[rel_id].max2 = hole_max;
                min_max_values[rel_id].min2 = hole_min;
            } else if (_parameter->znormalize) {

                cal_mean_std(rescal_scores, rescal_min, rescal_max);
                cal_mean_std(hole_scores, hole_min, hole_max);

                for (int i = 0; i < index1; i++) {
                    prob.x[i][0].value = (prob.x[i][0].value - rescal_min) / rescal_max;
                    prob.x[i][1].value = (prob.x[i][1].value - hole_min) / hole_max;
                }

                min_max_values[rel_id].max1 = rescal_max;
                min_max_values[rel_id].min1 = rescal_min;

                min_max_values[rel_id].max2 = hole_max;
                min_max_values[rel_id].min2 = hole_min;
            }

            model *model_ = train(&prob, &param);

            weights[rel_id] = SimpleWeight(model_->w[0], model_->w[1]);

            free_and_destroy_model(&model_);

            for (int i = 0; i < prob.l; i++) {
                delete[] prob.x[i];
            }

            delete[] prob.x;
            delete[] prob.y;
        }

        destroy_param(&param);

        cout << "time for learning weight: " << timer.getElapsedTime() << " secs" << endl;
    }


    // HOLE + TransE
    inline void learn_weights_HT(Data *data, Parameter *_parameter, DenseMatrix &hole_E, DenseMatrix &hole_P,
                                 DenseMatrix &transe_A, DenseMatrix &transe_R, vector<SimpleWeight> &weights, vector<min_max> &min_max_values) {

        weights.resize(data->K);
        if (_parameter->normalize) {
            min_max_values.resize(data->K);
        }

        parameter param;
        param.solver_type = L2R_LR;
        param.C = _parameter->c;
        param.eps = 1e-4;

        if(_parameter->num_of_thread==-1) {
            param.nr_thread = omp_get_num_procs();
        } else {
            param.nr_thread = _parameter->num_of_thread;
        }
        cout << "number of thread for logistic regression: " << param.nr_thread << endl;

        param.nr_weight = 0;
        param.weight_label = NULL;
        param.weight = NULL;
        param.init_sol = NULL;

        // if parameter->normalize, it is min and max. if parameter->znormalize, it is mean and std
        value_type hole_min = 0;
        value_type transe_min = 0;
        value_type hole_max = 0;
        value_type transe_max = 0;

        vector<value_type > hole_scores;
        vector<value_type > transe_scores;

        DFTI_DESCRIPTOR_HANDLE descriptor;

#ifdef use_double
        DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_REAL, 1, _parameter->hole_D);
#else
        DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_REAL, 1, _parameter->hole_D);
#endif

        value_type scale = 1.0 / _parameter->hole_D;
        DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, scale); //Scale down the output
        DftiSetValue(descriptor, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE); //Out of place FFT
        DftiCommitDescriptor(descriptor);

        Monitor timer;
        timer.start();

        for (int rel_id = 0; rel_id < data->K; rel_id++) {

            vector<Tuple<int> > &tuples = data->relation2tupleList_mapping[rel_id];
            int size = tuples.size() * 4 * _parameter->num_of_duplicated_true_triples;

            // `l' is the number of training data. If bias >= 0, we assume
            // that one additional feature is added to the end of each data
            // instance. `n' is the number of feature (including the bias feature
            // if bias >= 0).
            problem prob;
            prob.bias = 1;
            prob.l = size;
            prob.n = 3;
            prob.y = new double[prob.l];

            prob.x = new feature_node* [prob.l];

            int index1 = 0;
            int index2 = 0;

            if(_parameter->normalize) {
                transe_min = std::numeric_limits<value_type>::max();
                hole_min = transe_min;
                transe_max = std::numeric_limits<value_type>::min();
                hole_max = transe_max;
            } else if (_parameter->znormalize) {
                hole_scores.resize(size);
                transe_scores.resize(size);
            }

            int score_index = 0;

            for (int tuple_index = 0; tuple_index < tuples.size(); tuple_index++) {

                value_type p_hole_score = cal_hole_score(tuples[tuple_index].subject, tuples[tuple_index].object, rel_id, _parameter->hole_D, hole_E, hole_P, descriptor, false);
                value_type p_transe_score = cal_transe_score(tuples[tuple_index].subject, tuples[tuple_index].object, rel_id, _parameter->transe_D, _parameter->L1_flag, transe_A, transe_R);

                if(_parameter->normalize) {
                    transe_max = std::max(transe_max, p_transe_score);
                    transe_min = std::min(transe_min, p_transe_score);
                    hole_max = std::max(hole_max, p_hole_score);
                    hole_min = std::min(hole_min, p_hole_score);
                } else if (_parameter->znormalize) {
                    transe_scores[score_index] = p_transe_score;
                    hole_scores[score_index] = p_hole_score;
                    score_index++;
                }

                for (int count = 0; count < 2 * _parameter->num_of_duplicated_true_triples; count++) {

                    prob.x[index1] = new feature_node[prob.n + 1];

                    prob.x[index1][0].index = 1;  // index should start from 1
                    prob.x[index1][0].value = p_hole_score;

                    prob.x[index1][1].index = 2;
                    prob.x[index1][1].value = p_transe_score;

                    prob.x[index1][2].index = 3;
                    prob.x[index1][2].value = prob.bias;

                    prob.x[index1][3].index = -1; //Each row of properties should be terminated with a -1 according to the readme

                    index1++;

                    prob.y[index2] = 1;
                    index2++;
                }

                // first replace subjects
                for (int count = 0; count < _parameter->num_of_duplicated_true_triples; count++) {
                    int random_entity_id = RandomUtil::uniform_int(0, data->N);
                    while(data->faked_tuple_exist_train(rel_id, random_entity_id, tuples[tuple_index].object)){
                        random_entity_id = RandomUtil::uniform_int(0, data->N);
                    }

                    value_type n_hole_score = cal_hole_score(random_entity_id, tuples[tuple_index].object, rel_id, _parameter->hole_D, hole_E, hole_P, descriptor, false);
                    value_type n_transe_score = cal_transe_score(random_entity_id, tuples[tuple_index].object, rel_id, _parameter->transe_D, _parameter->L1_flag, transe_A, transe_R);

                    if(_parameter->normalize) {
                        transe_max = std::max(transe_max, n_transe_score);
                        transe_min = std::min(transe_min, n_transe_score);
                        hole_max = std::max(hole_max, n_hole_score);
                        hole_min = std::min(hole_min, n_hole_score);
                    } else if (_parameter->znormalize) {
                        transe_scores[score_index] = n_transe_score;
                        hole_scores[score_index] = n_hole_score;
                        score_index++;
                    }

                    prob.x[index1] = new feature_node[prob.n + 1];

                    prob.x[index1][0].index = 1;
                    prob.x[index1][0].value = n_hole_score;

                    prob.x[index1][1].index = 2;
                    prob.x[index1][1].value = n_transe_score;

                    prob.x[index1][2].index = 3;
                    prob.x[index1][2].value = prob.bias;

                    prob.x[index1][3].index = -1; //Each row of properties should be terminated with a -1 according to the readme

                    index1++;

                    prob.y[index2] = 0;
                    index2++;
                }

                // then replace objects
                for (int count = 0; count < _parameter->num_of_duplicated_true_triples; count++) {
                    int random_entity_id = RandomUtil::uniform_int(0, data->N);
                    while(data->faked_tuple_exist_train(rel_id, tuples[tuple_index].subject, random_entity_id)){
                        random_entity_id = RandomUtil::uniform_int(0, data->N);
                    }

                    value_type n_hole_score = cal_hole_score(tuples[tuple_index].subject, random_entity_id, rel_id, _parameter->hole_D, hole_E, hole_P, descriptor, false);
                    value_type n_transe_score = cal_transe_score(tuples[tuple_index].subject, random_entity_id, rel_id, _parameter->transe_D, _parameter->L1_flag, transe_A, transe_R);

                    if(_parameter->normalize) {
                        transe_max = std::max(transe_max, n_transe_score);
                        transe_min = std::min(transe_min, n_transe_score);
                        hole_max = std::max(hole_max, n_hole_score);
                        hole_min = std::min(hole_min, n_hole_score);
                    } else if (_parameter->znormalize) {
                        transe_scores[score_index] = n_transe_score;
                        hole_scores[score_index] = n_hole_score;
                        score_index++;
                    }

                    prob.x[index1] = new feature_node[prob.n + 1];

                    prob.x[index1][0].index = 1;
                    prob.x[index1][0].value = n_hole_score;

                    prob.x[index1][1].index = 2;
                    prob.x[index1][1].value = n_transe_score;

                    prob.x[index1][2].index = 3;
                    prob.x[index1][2].value = prob.bias;

                    prob.x[index1][3].index = -1; //Each row of properties should be terminated with a -1 according to the readme

                    index1++;

                    prob.y[index2] = 0;
                    index2++;
                }
            }

            if(_parameter->normalize) {
                value_type transe_range = transe_max - transe_min;
                value_type hole_range = hole_max - hole_min;

                for (int i = 0; i < index1; i++) {
                    prob.x[i][0].value = (prob.x[i][0].value - hole_min) / hole_range;
                    prob.x[i][1].value = (prob.x[i][1].value - transe_min) / transe_range;
                }

                min_max_values[rel_id].max1 = hole_max;
                min_max_values[rel_id].min1 = hole_min;
                min_max_values[rel_id].max2 = transe_max;
                min_max_values[rel_id].min2 = transe_min;
            } else if (_parameter->znormalize) {

                cal_mean_std(hole_scores, hole_min, hole_max);
                cal_mean_std(transe_scores, transe_min, transe_max);

                for (int i = 0; i < index1; i++) {
                    prob.x[i][0].value = (prob.x[i][0].value - hole_min) / hole_max;
                    prob.x[i][1].value = (prob.x[i][1].value - transe_min) / transe_max;
                }

                min_max_values[rel_id].max1 = hole_max;
                min_max_values[rel_id].min1 = hole_min;

                min_max_values[rel_id].max2 = transe_max;
                min_max_values[rel_id].min2 = transe_min;
            }

            model *model_ = train(&prob, &param);

            weights[rel_id] = SimpleWeight(model_->w[0], model_->w[1]);

            free_and_destroy_model(&model_);

            for (int i = 0; i < prob.l; i++) {
                delete[] prob.x[i];
            }

            delete[] prob.x;
            delete[] prob.y;

        }

        destroy_param(&param);
        DftiFreeDescriptor(&descriptor);

        cout << "time for learning weight: " << timer.getElapsedTime() << " secs" << endl;
    }

    // RESCAL + HOLE + TransE
    inline void learn_weights_RHT(Data *data, Parameter *_parameter, DenseMatrix &rescal_A, vector<DenseMatrix> &rescal_R,
                      DenseMatrix &hole_E, DenseMatrix &hole_P, DenseMatrix &transe_A, DenseMatrix &transe_R,
                      vector<SimpleWeight> &weights, DFTI_DESCRIPTOR_HANDLE &descriptor, vector<min_max> &min_max_values) {

        weights.resize(data->K);
        if (_parameter->normalize) {
            min_max_values.resize(data->K);
        }

        parameter param;
        param.solver_type = L2R_LR;
        param.C = _parameter->c;
        param.eps = 1e-4;

        if(_parameter->num_of_thread==-1) {
            param.nr_thread = omp_get_num_procs();
        } else {
            param.nr_thread = _parameter->num_of_thread;
        }
        cout << "number of thread for logistic regression: " << param.nr_thread << endl;

        param.nr_weight = 0;
        param.weight_label = NULL;
        param.weight = NULL;
        param.init_sol = NULL;

        // if parameter->normalize, it is min and max. if parameter->znormalize, it is mean and std
        value_type rescal_min = 0;
        value_type transe_min = 0;
        value_type hole_min = 0;

        value_type rescal_max = 0;
        value_type transe_max = 0;
        value_type hole_max = 0;

        vector<value_type > rescal_scores;
        vector<value_type > transe_scores;
        vector<value_type > hole_scores;

        Monitor timer;
        timer.start();

        for (int rel_id = 0; rel_id < data->K; rel_id++) {

            vector<Tuple<int> > &tuples = data->relation2tupleList_mapping[rel_id];
            int size = tuples.size() * 4 * _parameter->num_of_duplicated_true_triples;

            // `l' is the number of training data. If bias >= 0, we assume
            // that one additional feature is added to the end of each data
            // instance. `n' is the number of feature (including the bias feature
            // if bias >= 0).
            problem prob;
            prob.bias = 1;
            prob.l = size;
            prob.n = 4;
            prob.y = new double[prob.l];

            prob.x = new feature_node* [prob.l];

            int index1 = 0;
            int index2 = 0;

            if(_parameter->normalize) {
                transe_min = std::numeric_limits<value_type>::max();
                hole_min = transe_min;
                rescal_min = transe_min;
                transe_max = std::numeric_limits<value_type>::min();
                hole_max = transe_max;
                rescal_max = transe_max;
            } else if (_parameter->znormalize) {
                hole_scores.resize(size);
                rescal_scores.resize(size);
                transe_scores.resize(size);
            }

            int score_index = 0;

            for (int tuple_index = 0; tuple_index < tuples.size(); tuple_index++) {

                value_type p_rescal_score = cal_rescal_score(rel_id, tuples[tuple_index].subject, tuples[tuple_index].object, _parameter->rescal_D, rescal_A, rescal_R);
                value_type p_transe_score = cal_transe_score(tuples[tuple_index].subject, tuples[tuple_index].object, rel_id, _parameter->transe_D, _parameter->L1_flag, transe_A, transe_R);
                value_type p_hole_score = cal_hole_score(tuples[tuple_index].subject, tuples[tuple_index].object, rel_id, _parameter->hole_D, hole_E, hole_P, descriptor, false);

                if(_parameter->normalize) {
                    transe_max = std::max(transe_max, p_transe_score);
                    transe_min = std::min(transe_min, p_transe_score);

                    hole_max = std::max(hole_max, p_hole_score);
                    hole_min = std::min(hole_min, p_hole_score);

                    rescal_max = std::max(rescal_max, p_rescal_score);
                    rescal_min = std::min(rescal_min, p_rescal_score);
                } else if (_parameter->znormalize) {
                    transe_scores[score_index] = p_transe_score;
                    hole_scores[score_index] = p_hole_score;
                    rescal_scores[score_index] = p_rescal_score;
                    score_index++;
                }

                for (int count = 0; count < 2 * _parameter->num_of_duplicated_true_triples; count++) {

                    prob.x[index1] = new feature_node[prob.n + 1];

                    prob.x[index1][0].index = 1;  // index should start from 1
                    prob.x[index1][0].value = p_rescal_score;

                    prob.x[index1][1].index = 2;
                    prob.x[index1][1].value = p_hole_score;

                    prob.x[index1][2].index = 3;
                    prob.x[index1][2].value = p_transe_score;

                    prob.x[index1][3].index = 4;
                    prob.x[index1][3].value = prob.bias;

                    prob.x[index1][4].index = -1; //Each row of properties should be terminated with a -1 according to the readme

                    index1++;

                    prob.y[index2] = 1;
                    index2++;
                }

                // first replace subjects
                for (int count = 0; count < _parameter->num_of_duplicated_true_triples; count++) {
                    int random_entity_id = RandomUtil::uniform_int(0, data->N);
                    while(data->faked_tuple_exist_train(rel_id, random_entity_id, tuples[tuple_index].object)){
                        random_entity_id = RandomUtil::uniform_int(0, data->N);
                    }

                    value_type n_rescal_score = cal_rescal_score(rel_id, random_entity_id, tuples[tuple_index].object, _parameter->rescal_D, rescal_A, rescal_R);
                    value_type n_transe_score = cal_transe_score(random_entity_id, tuples[tuple_index].object, rel_id, _parameter->transe_D, _parameter->L1_flag, transe_A, transe_R);
                    value_type n_hole_score = cal_hole_score(random_entity_id, tuples[tuple_index].object, rel_id, _parameter->hole_D, hole_E, hole_P, descriptor, false);

                    if(_parameter->normalize) {
                        transe_max = std::max(transe_max, n_transe_score);
                        transe_min = std::min(transe_min, n_transe_score);

                        hole_max = std::max(hole_max, n_hole_score);
                        hole_min = std::min(hole_min, n_hole_score);

                        rescal_max = std::max(rescal_max, n_rescal_score);
                        rescal_min = std::min(rescal_min, n_rescal_score);
                    } else if (_parameter->znormalize) {
                        transe_scores[score_index] = n_transe_score;
                        hole_scores[score_index] = n_hole_score;
                        rescal_scores[score_index] = n_rescal_score;
                        score_index++;
                    }

                    prob.x[index1] = new feature_node[prob.n + 1];

                    prob.x[index1][0].index = 1;
                    prob.x[index1][0].value = n_rescal_score;

                    prob.x[index1][1].index = 2;
                    prob.x[index1][1].value = n_hole_score;

                    prob.x[index1][2].index = 3;
                    prob.x[index1][2].value = n_transe_score;

                    prob.x[index1][3].index = 4;
                    prob.x[index1][3].value = prob.bias;

                    prob.x[index1][4].index = -1; //Each row of properties should be terminated with a -1 according to the readme

                    index1++;

                    prob.y[index2] = 0;
                    index2++;
                }

                // then replace objects
                for (int count = 0; count < _parameter->num_of_duplicated_true_triples; count++) {
                    int random_entity_id = RandomUtil::uniform_int(0, data->N);
                    while(data->faked_tuple_exist_train(rel_id, tuples[tuple_index].subject, random_entity_id)){
                        random_entity_id = RandomUtil::uniform_int(0, data->N);
                    }

                    value_type n_rescal_score = cal_rescal_score(rel_id, tuples[tuple_index].subject, random_entity_id, _parameter->rescal_D, rescal_A, rescal_R);
                    value_type n_transe_score = cal_transe_score(tuples[tuple_index].subject, random_entity_id, rel_id, _parameter->transe_D, _parameter->L1_flag, transe_A, transe_R);
                    value_type n_hole_score = cal_hole_score(tuples[tuple_index].subject, random_entity_id, rel_id, _parameter->hole_D, hole_E, hole_P, descriptor, false);

                    if(_parameter->normalize) {
                        transe_max = std::max(transe_max, n_transe_score);
                        transe_min = std::min(transe_min, n_transe_score);

                        hole_max = std::max(hole_max, n_hole_score);
                        hole_min = std::min(hole_min, n_hole_score);

                        rescal_max = std::max(rescal_max, n_rescal_score);
                        rescal_min = std::min(rescal_min, n_rescal_score);
                    } else if (_parameter->znormalize) {
                        transe_scores[score_index] = n_transe_score;
                        hole_scores[score_index] = n_hole_score;
                        rescal_scores[score_index] = n_rescal_score;
                        score_index++;
                    }

                    prob.x[index1] = new feature_node[prob.n + 1];

                    prob.x[index1][0].index = 1;
                    prob.x[index1][0].value = n_rescal_score;

                    prob.x[index1][1].index = 2;
                    prob.x[index1][1].value = n_hole_score;

                    prob.x[index1][2].index = 3;
                    prob.x[index1][2].value = n_transe_score;

                    prob.x[index1][3].index = 4;
                    prob.x[index1][3].value = prob.bias;

                    prob.x[index1][4].index = -1; //Each row of properties should be terminated with a -1 according to the readme

                    index1++;

                    prob.y[index2] = 0;
                    index2++;
                }
            }

            if(_parameter->normalize) {
                value_type transe_range = transe_max - transe_min;
                value_type hole_range = hole_max - hole_min;
                value_type rescal_range = rescal_max - rescal_min;

                for (int i = 0; i < index1; i++) {
                    prob.x[i][0].value = (prob.x[i][0].value - rescal_min) / rescal_range;
                    prob.x[i][1].value = (prob.x[i][1].value - hole_min) / hole_range;
                    prob.x[i][2].value = (prob.x[i][2].value - transe_min) / transe_range;
                }

                min_max_values[rel_id].max1 = rescal_max;
                min_max_values[rel_id].min1 = rescal_min;
                min_max_values[rel_id].max2 = hole_max;
                min_max_values[rel_id].min2 = hole_min;
                min_max_values[rel_id].max3 = transe_max;
                min_max_values[rel_id].min3 = transe_min;

            } else if (_parameter->znormalize) {

                cal_mean_std(rescal_scores, rescal_min, rescal_max);
                cal_mean_std(hole_scores, hole_min, hole_max);
                cal_mean_std(transe_scores, transe_min, transe_max);

                for (int i = 0; i < index1; i++) {
                    prob.x[i][0].value = (prob.x[i][0].value - rescal_min) / rescal_max;
                    prob.x[i][1].value = (prob.x[i][1].value - hole_min) / hole_max;
                    prob.x[i][2].value = (prob.x[i][2].value - transe_min) / transe_max;
                }

                min_max_values[rel_id].max1 = rescal_max;
                min_max_values[rel_id].min1 = rescal_min;

                min_max_values[rel_id].max2 = hole_max;
                min_max_values[rel_id].min2 = hole_min;

                min_max_values[rel_id].max3 = transe_max;
                min_max_values[rel_id].min3 = transe_min;
            }

            model *model_ = train(&prob, &param);

            weights[rel_id] = SimpleWeight(model_->w[0], model_->w[1], model_->w[2]);

            free_and_destroy_model(&model_);

            for (int i = 0; i < prob.l; i++) {
                delete[] prob.x[i];
            }

            delete[] prob.x;
            delete[] prob.y;

        }

        destroy_param(&param);

        cout << "time for learning weight: " << timer.getElapsedTime() << " secs" << endl;
    }
}
#endif //LRUTIL_H
