#include "tf/util/Base.h"
#include "tf/util/Data.h"
#include "tf/util/Parameter.h"
#include "tf/util/Calculator.h"
#include <boost/bind.hpp>

using namespace Calculator;

void cal_score(Parameter *_parameter, Data *data, DenseMatrix &rescal_A, vector<DenseMatrix> &rescal_R, DenseMatrix &hole_E, DenseMatrix &hole_P,
          DenseMatrix &transe_A, DenseMatrix &transe_R, vector<SimpleWeight> *ensembleWeights,
          vector<min_max> *min_max_values) {

    DFTI_DESCRIPTOR_HANDLE descriptor;

    if ((_parameter->method == Method::m_HOLE) || (_parameter->method == Method::m_HTLREnsemble) ||
        (_parameter->method == Method::m_RHLREnsemble) || (_parameter->method == Method::m_RHTLREnsemble)) {

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

    }

    string output_folder = _parameter->output_path + "/score_list";
    boost::filesystem::create_directories(output_folder);

    for (int rel_id = 0; rel_id < data->K; rel_id++) {

        vector<Tuple<int> > &valid_pos_tuples = (data->relation2tupleValidationPosList_mapping)[rel_id];
        vector<Tuple<int> > &test_pos_tuples = (data->relation2tupleTestPosList_mapping)[rel_id];
        vector<Tuple<int> > &valid_neg_tuples = (data->relation2tupleValidationNegList_mapping)[rel_id];
        vector<Tuple<int> > &test_neg_tuples = (data->relation2tupleTestNegList_mapping)[rel_id];

        vector<pair<int, value_type> > valid_scores(valid_pos_tuples.size() + valid_neg_tuples.size());

        Tuple<int> *tuple = nullptr;
        int label;

        for (int tuple_index = 0; tuple_index < valid_scores.size(); tuple_index++) {

            if (tuple_index >= valid_pos_tuples.size()) {
                tuple = &(valid_neg_tuples[tuple_index - valid_pos_tuples.size()]);
                label = 0;
            } else {
                tuple = &(valid_pos_tuples[tuple_index]);
                label = 1;
            }

            switch(_parameter->method) {
                case Method::m_RESCAL_RANK:{
                    value_type rescal_score = cal_rescal_score(rel_id, tuple->subject,
                                                               tuple->object, _parameter->rescal_D,
                                                               rescal_A, rescal_R);

                    valid_scores[tuple_index] = make_pair(label, rescal_score);

                    break;
                }
                case Method::m_TransE:{
                    value_type transe_score = - cal_transe_score(tuple->subject,
                                                                 tuple->object, rel_id,
                                                                 _parameter->transe_D, _parameter->L1_flag, transe_A,
                                                                 transe_R);

                    valid_scores[tuple_index] = make_pair(label, transe_score);

                    break;
                }
                case Method::m_HOLE: {
                    value_type hole_score = cal_hole_score(tuple->subject,
                                                           tuple->object, rel_id,
                                                           _parameter->hole_D, hole_E, hole_P, descriptor, false);

                    valid_scores[tuple_index] = make_pair(label, hole_score);

                    break;
                }
                case Method::m_RTLREnsemble: {
                    value_type rescal_score = cal_rescal_score(rel_id, tuple->subject,
                                                               tuple->object, _parameter->rescal_D,
                                                               rescal_A, rescal_R);

                    value_type transe_score = cal_transe_score(tuple->subject,
                                                               tuple->object, rel_id,
                                                               _parameter->transe_D, _parameter->L1_flag, transe_A,
                                                               transe_R);
                    value_type score;
                    if (_parameter->normalize) {
                        score = (*ensembleWeights)[rel_id].w1 * (rescal_score - (*min_max_values)[rel_id].min1) /
                                ((*min_max_values)[rel_id].max1 - (*min_max_values)[rel_id].min1) +
                                (*ensembleWeights)[rel_id].w2 * (transe_score - (*min_max_values)[rel_id].min2)  /
                                ((*min_max_values)[rel_id].max2 - (*min_max_values)[rel_id].min2);
                    } else if (_parameter->znormalize) {
                        score = (*ensembleWeights)[rel_id].w1 * (rescal_score - (*min_max_values)[rel_id].min1) /
                                ((*min_max_values)[rel_id].max1) +
                                (*ensembleWeights)[rel_id].w2 * (transe_score - (*min_max_values)[rel_id].min2)  /
                                ((*min_max_values)[rel_id].max2);
                    } else {
                        score = (*ensembleWeights)[rel_id].w1 * rescal_score +
                                (*ensembleWeights)[rel_id].w2 * transe_score;
                    }

                    valid_scores[tuple_index] = make_pair(label, score);
                    break;
                }
                case Method::m_HTLREnsemble: {

                    value_type hole_score = cal_hole_score(tuple->subject,
                                                           tuple->object, rel_id,
                                                           _parameter->hole_D, hole_E, hole_P, descriptor, false);

                    value_type transe_score = cal_transe_score(tuple->subject,
                                                               tuple->object, rel_id,
                                                               _parameter->transe_D, _parameter->L1_flag, transe_A,
                                                               transe_R);

                    value_type score;
                    if (_parameter->normalize) {
                        score = (*ensembleWeights)[rel_id].w1 * (hole_score - (*min_max_values)[rel_id].min1) /
                                                    ((*min_max_values)[rel_id].max1 - (*min_max_values)[rel_id].min1) +
                                                    (*ensembleWeights)[rel_id].w2 * (transe_score - (*min_max_values)[rel_id].min2)  /
                                                    ((*min_max_values)[rel_id].max2 - (*min_max_values)[rel_id].min2);
                    } else if (_parameter->znormalize) {
                        score = (*ensembleWeights)[rel_id].w1 * (hole_score - (*min_max_values)[rel_id].min1) /
                                                    ((*min_max_values)[rel_id].max1) +
                                                    (*ensembleWeights)[rel_id].w2 * (transe_score - (*min_max_values)[rel_id].min2)  /
                                                    ((*min_max_values)[rel_id].max2);
                    } else {
                        score = (*ensembleWeights)[rel_id].w1 * hole_score +
                                                    (*ensembleWeights)[rel_id].w2 * transe_score;
                    }

                    valid_scores[tuple_index] = make_pair(label, score);
                    break;
                }
                case Method::m_RHLREnsemble: {

                    value_type rescal_score = cal_rescal_score(rel_id, tuple->subject,
                                                               tuple->object, _parameter->rescal_D,
                                                               rescal_A, rescal_R);

                    value_type hole_score = cal_hole_score(tuple->subject,
                                                           tuple->object, rel_id,
                                                           _parameter->hole_D, hole_E, hole_P, descriptor, false);

                    value_type score;
                    if (_parameter->normalize) {
                        score = (*ensembleWeights)[rel_id].w1 * (rescal_score - (*min_max_values)[rel_id].min1) /
                                                    ((*min_max_values)[rel_id].max1 - (*min_max_values)[rel_id].min1) +
                                                    (*ensembleWeights)[rel_id].w2 * (hole_score - (*min_max_values)[rel_id].min2)  /
                                                    ((*min_max_values)[rel_id].max2 - (*min_max_values)[rel_id].min2);
                    } else if (_parameter->znormalize) {
                        score = (*ensembleWeights)[rel_id].w1 * (rescal_score - (*min_max_values)[rel_id].min1) /
                                                    ((*min_max_values)[rel_id].max1) +
                                                    (*ensembleWeights)[rel_id].w2 * (hole_score - (*min_max_values)[rel_id].min2)  /
                                                    ((*min_max_values)[rel_id].max2);
                    } else {
                        score = (*ensembleWeights)[rel_id].w1 * rescal_score +
                                                            (*ensembleWeights)[rel_id].w2 * hole_score;
                    }

                    valid_scores[tuple_index] = make_pair(label, score);
                    break;
                }
                case Method::m_RHTLREnsemble: {

                    value_type rescal_score = cal_rescal_score(rel_id, tuple->subject,
                                                               tuple->object, _parameter->rescal_D,
                                                               rescal_A, rescal_R);

                    value_type hole_score = cal_hole_score(tuple->subject,
                                                           tuple->object, rel_id,
                                                           _parameter->hole_D, hole_E, hole_P, descriptor, false);

                    value_type transe_score = cal_transe_score(tuple->subject,
                                                               tuple->object, rel_id,
                                                               _parameter->transe_D, _parameter->L1_flag, transe_A,
                                                               transe_R);
                    value_type score;
                    if (_parameter->normalize) {

                        score = (*ensembleWeights)[rel_id].w1 * (rescal_score - (*min_max_values)[rel_id].min1) /
                                ((*min_max_values)[rel_id].max1 - (*min_max_values)[rel_id].min1) +
                                (*ensembleWeights)[rel_id].w2 * (hole_score - (*min_max_values)[rel_id].min2)  /
                                ((*min_max_values)[rel_id].max2 - (*min_max_values)[rel_id].min2) +
                                (*ensembleWeights)[rel_id].w3 * (transe_score - (*min_max_values)[rel_id].min3)  /
                                ((*min_max_values)[rel_id].max3 - (*min_max_values)[rel_id].min3);

                    } else if (_parameter->znormalize) {
                        score = (*ensembleWeights)[rel_id].w1 * (rescal_score - (*min_max_values)[rel_id].min1) /
                                ((*min_max_values)[rel_id].max1) +
                                (*ensembleWeights)[rel_id].w2 * (hole_score - (*min_max_values)[rel_id].min2)  /
                                ((*min_max_values)[rel_id].max2) +
                                (*ensembleWeights)[rel_id].w3 * (transe_score - (*min_max_values)[rel_id].min3)  /
                                ((*min_max_values)[rel_id].max3);
                    } else {
                        score = (*ensembleWeights)[rel_id].w1 * rescal_score + (*ensembleWeights)[rel_id].w2 * hole_score +
                                (*ensembleWeights)[rel_id].w3 * transe_score;
                    }

                    valid_scores[tuple_index] = make_pair(label, score);
                    break;
                }
                default: {
                    cerr << "unrecognized method" << endl;
                    exit(1);
                }
            }


        }

        std::sort(valid_scores.begin(), valid_scores.end(),
                  boost::bind(&std::pair<int, value_type>::second, _1) >
                  boost::bind(&std::pair<int, value_type>::second, _2));

        string valid_file_name = output_folder + "/valid_" + to_string(rel_id) + ".dat";

        ofstream valid_output(valid_file_name.c_str());
        for (auto p:valid_scores){
            valid_output << p.first << "," << p.second << endl;
        }
        valid_output.close();

        vector<pair<int, value_type> > test_scores(test_pos_tuples.size() + test_neg_tuples.size());

        for (int tuple_index = 0; tuple_index < test_scores.size(); tuple_index++) {

            if (tuple_index >= test_pos_tuples.size()) {
                tuple = &(test_neg_tuples[tuple_index - test_pos_tuples.size()]);
                label = 0;
            } else {
                tuple = &(test_pos_tuples[tuple_index]);
                label = 1;
            }


            switch(_parameter->method) {
                case Method::m_RESCAL_RANK:{
                    value_type rescal_score = cal_rescal_score(rel_id, tuple->subject,
                                                               tuple->object, _parameter->rescal_D,
                                                               rescal_A, rescal_R);

                    test_scores[tuple_index] = make_pair(label, rescal_score);

                    break;
                }
                case Method::m_TransE:{
                    value_type transe_score = - cal_transe_score(tuple->subject,
                                                                 tuple->object, rel_id,
                                                               _parameter->transe_D, _parameter->L1_flag, transe_A,
                                                               transe_R);

                    test_scores[tuple_index] = make_pair(label, transe_score);

                    break;
                }
                case Method::m_HOLE: {
                    value_type hole_score = cal_hole_score(tuple->subject,
                                                           tuple->object, rel_id,
                                                           _parameter->hole_D, hole_E, hole_P, descriptor, false);

                    test_scores[tuple_index] = make_pair(label, hole_score);

                    break;
                }
                case Method::m_RTLREnsemble: {
                    value_type rescal_score = cal_rescal_score(rel_id, tuple->subject,
                                                               tuple->object, _parameter->rescal_D,
                                                               rescal_A, rescal_R);

                    value_type transe_score = cal_transe_score(tuple->subject,
                                                               tuple->object, rel_id,
                                                               _parameter->transe_D, _parameter->L1_flag, transe_A,
                                                               transe_R);
                    value_type score;
                    if (_parameter->normalize) {
                        score = (*ensembleWeights)[rel_id].w1 * (rescal_score - (*min_max_values)[rel_id].min1) /
                                ((*min_max_values)[rel_id].max1 - (*min_max_values)[rel_id].min1) +
                                (*ensembleWeights)[rel_id].w2 * (transe_score - (*min_max_values)[rel_id].min2)  /
                                ((*min_max_values)[rel_id].max2 - (*min_max_values)[rel_id].min2);
                    } else if (_parameter->znormalize) {
                        score = (*ensembleWeights)[rel_id].w1 * (rescal_score - (*min_max_values)[rel_id].min1) /
                                ((*min_max_values)[rel_id].max1) +
                                (*ensembleWeights)[rel_id].w2 * (transe_score - (*min_max_values)[rel_id].min2)  /
                                ((*min_max_values)[rel_id].max2);
                    } else {
                        score = (*ensembleWeights)[rel_id].w1 * rescal_score +
                                (*ensembleWeights)[rel_id].w2 * transe_score;
                    }

                    test_scores[tuple_index] = make_pair(label, score);
                    break;
                }
                case Method::m_HTLREnsemble: {

                    value_type hole_score = cal_hole_score(tuple->subject,
                                                           tuple->object, rel_id,
                                                           _parameter->hole_D, hole_E, hole_P, descriptor, false);

                    value_type transe_score = cal_transe_score(tuple->subject,
                                                               tuple->object, rel_id,
                                                               _parameter->transe_D, _parameter->L1_flag, transe_A,
                                                               transe_R);

                    value_type score;
                    if (_parameter->normalize) {
                        score = (*ensembleWeights)[rel_id].w1 * (hole_score - (*min_max_values)[rel_id].min1) /
                                ((*min_max_values)[rel_id].max1 - (*min_max_values)[rel_id].min1) +
                                (*ensembleWeights)[rel_id].w2 * (transe_score - (*min_max_values)[rel_id].min2)  /
                                ((*min_max_values)[rel_id].max2 - (*min_max_values)[rel_id].min2);
                    } else if (_parameter->znormalize) {
                        score = (*ensembleWeights)[rel_id].w1 * (hole_score - (*min_max_values)[rel_id].min1) /
                                ((*min_max_values)[rel_id].max1) +
                                (*ensembleWeights)[rel_id].w2 * (transe_score - (*min_max_values)[rel_id].min2)  /
                                ((*min_max_values)[rel_id].max2);
                    } else {
                        score = (*ensembleWeights)[rel_id].w1 * hole_score +
                                (*ensembleWeights)[rel_id].w2 * transe_score;
                    }

                    test_scores[tuple_index] = make_pair(label, score);
                    break;
                }
                case Method::m_RHLREnsemble: {

                    value_type rescal_score = cal_rescal_score(rel_id, tuple->subject,
                                                               tuple->object, _parameter->rescal_D,
                                                               rescal_A, rescal_R);

                    value_type hole_score = cal_hole_score(tuple->subject,
                                                           tuple->object, rel_id,
                                                           _parameter->hole_D, hole_E, hole_P, descriptor, false);

                    value_type score;
                    if (_parameter->normalize) {
                        score = (*ensembleWeights)[rel_id].w1 * (rescal_score - (*min_max_values)[rel_id].min1) /
                                ((*min_max_values)[rel_id].max1 - (*min_max_values)[rel_id].min1) +
                                (*ensembleWeights)[rel_id].w2 * (hole_score - (*min_max_values)[rel_id].min2)  /
                                ((*min_max_values)[rel_id].max2 - (*min_max_values)[rel_id].min2);
                    } else if (_parameter->znormalize) {
                        score = (*ensembleWeights)[rel_id].w1 * (rescal_score - (*min_max_values)[rel_id].min1) /
                                ((*min_max_values)[rel_id].max1) +
                                (*ensembleWeights)[rel_id].w2 * (hole_score - (*min_max_values)[rel_id].min2)  /
                                ((*min_max_values)[rel_id].max2);
                    } else {
                        score = (*ensembleWeights)[rel_id].w1 * rescal_score +
                                (*ensembleWeights)[rel_id].w2 * hole_score;
                    }

                    test_scores[tuple_index] = make_pair(label, score);
                    break;
                }
                case Method::m_RHTLREnsemble: {

                    value_type rescal_score = cal_rescal_score(rel_id, tuple->subject,
                                                               tuple->object, _parameter->rescal_D,
                                                               rescal_A, rescal_R);

                    value_type hole_score = cal_hole_score(tuple->subject,
                                                           tuple->object, rel_id,
                                                           _parameter->hole_D, hole_E, hole_P, descriptor, false);

                    value_type transe_score = cal_transe_score(tuple->subject,
                                                               tuple->object, rel_id,
                                                               _parameter->transe_D, _parameter->L1_flag, transe_A,
                                                               transe_R);
                    value_type score;
                    if (_parameter->normalize) {

                        score = (*ensembleWeights)[rel_id].w1 * (rescal_score - (*min_max_values)[rel_id].min1) /
                                ((*min_max_values)[rel_id].max1 - (*min_max_values)[rel_id].min1) +
                                (*ensembleWeights)[rel_id].w2 * (hole_score - (*min_max_values)[rel_id].min2)  /
                                ((*min_max_values)[rel_id].max2 - (*min_max_values)[rel_id].min2) +
                                (*ensembleWeights)[rel_id].w3 * (transe_score - (*min_max_values)[rel_id].min3)  /
                                ((*min_max_values)[rel_id].max3 - (*min_max_values)[rel_id].min3);

                    } else if (_parameter->znormalize) {
                        score = (*ensembleWeights)[rel_id].w1 * (rescal_score - (*min_max_values)[rel_id].min1) /
                                ((*min_max_values)[rel_id].max1) +
                                (*ensembleWeights)[rel_id].w2 * (hole_score - (*min_max_values)[rel_id].min2)  /
                                ((*min_max_values)[rel_id].max2) +
                                (*ensembleWeights)[rel_id].w3 * (transe_score - (*min_max_values)[rel_id].min3)  /
                                ((*min_max_values)[rel_id].max3);
                    } else {
                        score = (*ensembleWeights)[rel_id].w1 * rescal_score + (*ensembleWeights)[rel_id].w2 * hole_score +
                                (*ensembleWeights)[rel_id].w3 * transe_score;
                    }

                    test_scores[tuple_index] = make_pair(label, score);
                    break;
                }
                default: {
                    cerr << "unrecognized method" << endl;
                    exit(1);
                }
            }


        }

        std::sort(test_scores.begin(), test_scores.end(),
                  boost::bind(&std::pair<int, value_type>::second, _1) >
                  boost::bind(&std::pair<int, value_type>::second, _2));

        string test_file_name = output_folder + "/test_" + to_string(rel_id) + ".dat";

        ofstream test_output(test_file_name.c_str());
        for (auto p:test_scores){
            test_output << p.first << "," << p.second << endl;
        }
        test_output.close();
    }

    if ((_parameter->method == Method::m_HOLE) || (_parameter->method == Method::m_HTLREnsemble) ||
        (_parameter->method == Method::m_RHLREnsemble) || (_parameter->method == Method::m_RHTLREnsemble)) {
        DftiFreeDescriptor(&descriptor);
    }
}

/**
 * Output sorted score list for
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv) {

    string train_file, positive_valid_file, negative_valid_file, positive_test_file, negative_test_file;

    Parameter parameter;

    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("rescal_d", po::value<int>(&(parameter.rescal_D))->default_value(100), "number of dimensions for RESCAL")
            ("hole_d", po::value<int>(&(parameter.hole_D))->default_value(100), "number of dimensions for HOLE")
            ("transe_d", po::value<int>(&(parameter.transe_D))->default_value(100), "number of dimensions for TransE")
            ("n_mkl", po::value<int>(&(parameter.mkl_num_of_thread))->default_value(-1), "number of MKL threads. -1: automatically set by mkl")
            ("dist", po::value<bool>(&(parameter.L1_flag))->default_value(1), "1: L1 distance, 0: L2 distance")
            ("normalize", po::value<bool>(&(parameter.normalize))->default_value(1), "1: normalize the score, 0: original score")
            ("znormalize", po::value<bool>(&(parameter.znormalize))->default_value(0), "1: normalize the score using z score, 0: original score")
            ("rescal_r_epoch", po::value<int>(&(parameter.rescal_restore_epoch))->default_value(-1),
             "set restore_epoch to the last epoch, if you want to load A and R from saved files (RESCAL)")
            ("hole_r_epoch", po::value<int>(&(parameter.hole_restore_epoch))->default_value(-1),
             "set restore_epoch to the last epoch, if you want to load A and R from saved files (HOLE)")
            ("transe_r_epoch", po::value<int>(&(parameter.transe_restore_epoch))->default_value(-1),
             "set restore_epoch to the last epoch, if you want to load A and R from saved files (TransE)")
            ("rescal_path", po::value<string>(&(parameter.rescal_restore_path)), "restore folder for Rescal")
            ("hole_path", po::value<string>(&(parameter.hole_restore_path)), "restore folder for HOLE")
            ("transe_path", po::value<string>(&(parameter.transe_restore_path)), "restore folder for transE")
            ("t_path", po::value<string>(&(train_file)), "path to training file")
            ("v_p_path", po::value<string>(&(positive_valid_file)), "path to positive validation file")
            ("e_p_path", po::value<string>(&(positive_test_file)), "path to positive testing file")
            ("v_n_path", po::value<string>(&(negative_valid_file)), "path to negative validation file")
            ("e_n_path", po::value<string>(&(negative_test_file)), "path to negative testing file")
            ("w_path", po::value<string>(&(parameter.weight_path))->default_value(""), "path to weight file")
            ("o_path", po::value<string>(&(parameter.output_path))->default_value("./output"), "path to output file")
            ("method", po::value<int>(&(parameter.method)), "m_RESCAL_RANK=1, m_TransE=2, m_HOLE=3, m_RTLREnsemble=4, m_HTLREnsemble=5, m_RHLREnsemble=6, m_RHTLREnsemble=7");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << endl;
        return 0;
    } else if (parameter.normalize && parameter.znormalize){
        cout << "normalize and znormalize cannot be both 1" << endl << endl;
        cout << desc << endl;
        return 0;
    }

    if (parameter.mkl_num_of_thread > 0) {
        mkl_set_num_threads(parameter.mkl_num_of_thread);
    }

    int num_of_training_triples;
    int num_of_testing_triples;
    int num_of_validation_triples;

    Data data;
//    cout << positive_valid_file << endl;

    data.read_triples_separate(train_file, positive_valid_file, negative_valid_file, positive_test_file, negative_test_file);
    data.encode_triples_triple_classification();

    cout << "maximum number of threads for MKL: " << mkl_get_max_threads() << endl;
    cout << "num of training triples: " << data.num_of_training_triples << endl;
    cout << "num of testing triples: " << data.num_of_testing_triples_positive << endl;
    cout << "num of validation triples: " << data.num_of_validation_triples_positive << endl;
    cout << "Entity: " << data.N << endl;
    cout << "Relation: " << data.K << endl;
    cout << "RESCAL_Dimension: " << parameter.rescal_D << endl;
    cout << "HOLE_Dimension: " << parameter.hole_D << endl;
    cout << "TransE_Dimension: " << parameter.transe_D << endl;
    cout << "------------------------" << endl;

    DenseMatrix rescal_A;
    vector<DenseMatrix> rescal_R;

    DenseMatrix hole_E;
    DenseMatrix hole_P;

    DenseMatrix transe_A;
    DenseMatrix transe_R;

    vector<SimpleWeight> weights(data.K);
    vector<min_max> min_max_values(data.K);

    switch(parameter.method) {
        case Method::m_RESCAL_RANK:{
            rescal_A.resize(data.N, parameter.rescal_D);
            rescal_R.resize(data.K, DenseMatrix(parameter.rescal_D, parameter.rescal_D));
            read_rescal_matrices(rescal_A, rescal_R, parameter.rescal_restore_epoch, parameter.rescal_restore_path);
            break;
        }
        case Method::m_TransE:{
            transe_A.resize(data.N, parameter.transe_D);
            transe_R.resize(data.K, parameter.transe_D);
            read_transe_matrices(transe_A, transe_R, parameter.transe_restore_epoch, parameter.transe_restore_path);
            break;
        }
        case Method::m_HOLE: {
            hole_E.resize(data.N, parameter.hole_D);
            hole_P.resize(data.K, parameter.hole_D);
            read_transe_matrices(hole_E, hole_P, parameter.hole_restore_epoch, parameter.hole_restore_path);
            break;
        }
        case Method::m_RTLREnsemble: {
            rescal_A.resize(data.N, parameter.rescal_D);
            rescal_R.resize(data.K, DenseMatrix(parameter.rescal_D, parameter.rescal_D));
            transe_A.resize(data.N, parameter.transe_D);
            transe_R.resize(data.K, parameter.transe_D);

            read_rescal_matrices(rescal_A, rescal_R, parameter.rescal_restore_epoch, parameter.rescal_restore_path);
            read_transe_matrices(transe_A, transe_R, parameter.transe_restore_epoch, parameter.transe_restore_path);

            cout << "read weight from file: " << parameter.weight_path << endl;
            FileUtil::read_weights(weights, min_max_values, parameter.weight_path, false);

            break;
        }
        case Method::m_HTLREnsemble: {
            hole_E.resize(data.N, parameter.hole_D);
            hole_P.resize(data.K, parameter.hole_D);
            transe_A.resize(data.N, parameter.transe_D);
            transe_R.resize(data.K, parameter.transe_D);

            read_transe_matrices(hole_E, hole_P, parameter.hole_restore_epoch, parameter.hole_restore_path);
            read_transe_matrices(transe_A, transe_R, parameter.transe_restore_epoch, parameter.transe_restore_path);

            cout << "read weight from file: " << parameter.weight_path << endl;
            FileUtil::read_weights(weights, min_max_values, parameter.weight_path, false);

            break;
        }
        case Method::m_RHLREnsemble: {
            rescal_A.resize(data.N, parameter.rescal_D);
            rescal_R.resize(data.K, DenseMatrix(parameter.rescal_D, parameter.rescal_D));
            hole_E.resize(data.N, parameter.hole_D);
            hole_P.resize(data.K, parameter.hole_D);

            read_rescal_matrices(rescal_A, rescal_R, parameter.rescal_restore_epoch, parameter.rescal_restore_path);
            read_transe_matrices(hole_E, hole_P, parameter.hole_restore_epoch, parameter.hole_restore_path);

            cout << "read weight from file: " << parameter.weight_path << endl;
            FileUtil::read_weights(weights, min_max_values, parameter.weight_path, false);

            break;
        }
        case Method::m_RHTLREnsemble: {
            rescal_A.resize(data.N, parameter.rescal_D);
            rescal_R.resize(data.K, DenseMatrix(parameter.rescal_D, parameter.rescal_D));
            hole_E.resize(data.N, parameter.hole_D);
            hole_P.resize(data.K, parameter.hole_D);
            transe_A.resize(data.N, parameter.transe_D);
            transe_R.resize(data.K, parameter.transe_D);

            read_rescal_matrices(rescal_A, rescal_R, parameter.rescal_restore_epoch, parameter.rescal_restore_path);
            read_transe_matrices(hole_E, hole_P, parameter.hole_restore_epoch, parameter.hole_restore_path);
            read_transe_matrices(transe_A, transe_R, parameter.transe_restore_epoch, parameter.transe_restore_path);

            cout << "read weight from file: " << parameter.weight_path << endl;
            FileUtil::read_weights(weights, min_max_values, parameter.weight_path, true);

            break;
        }
        default: {
            cerr << "unrecognized method" << endl;
            exit(1);
        }
    }

    cal_score(&parameter, &data, rescal_A, rescal_R, hole_E, hole_P,
              transe_A, transe_R, &weights,
              &min_max_values);

    cout << "finish" << endl;

    return 0;
}