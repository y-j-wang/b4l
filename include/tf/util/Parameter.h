#ifndef PARAMETER_H
#define PARAMETER_H

#include "tf/util/Base.h"

class Parameter {

public:

    int num_of_thread = 1;
    int num_of_thread_eval; // Threads for evaluation
    int mkl_num_of_thread;

    string train_data_path;
    string test_data_path;
    string valid_data_path;

    string weight_path;

    string output_path;

    int epoch;
    value_type step_size;

    int restore_epoch;  // set restore_epoch to the last epoch, if you want to load A and R from saved files
    string restore_path;
    bool restore_from_transe = false; // use results from transe as initialization
    bool restore_from_hole = false;
    bool init_check = false;

    int print_epoch;
    int output_epoch;
    bool eval_train;  // true: show evaluation of training data
    bool eval_rel;    // true: show evaluation of replacing relations
    bool eval_map;    // true: show evaluation of mean average precision
    bool compute_fit;
    int hit_rate_topk;
    value_type margin;
    value_type transe_margin;
    value_type hole_margin;
    value_type rescal_margin;
    value_type Rho;
    value_type lambdaA; // regularization weight
    value_type lambdaR; // regularization weight
    value_type lambdaE;
    value_type lambdaP;
    value_type train_sample_percentage;

    string optimization;

    string log_path;
    bool print_log_header;

    bool L1_flag;

    // for ensemble
    int rescal_D, transe_D, hole_D;
    int rescal_restore_epoch, transe_restore_epoch, hole_restore_epoch;
    string hole_restore_path;
    string rescal_restore_path;
    string transe_restore_path;

    // for evaluation of MAP
    int num_of_replaced_entities;

    // for ensemble
    int num_of_duplicated_true_triples;
    bool normalize;
    bool znormalize;
    value_type c;

    int method;
};
#endif //PARAMETER_H
