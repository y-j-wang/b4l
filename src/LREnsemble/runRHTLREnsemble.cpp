#include "tf/util/Base.h"
#include "tf/util/Data.h"
#include "tf/util/FileUtil.h"
#include "tf/util/Monitor.h"
#include "tf/util/EvaluationUtil.h"
#include "tf/util/Parameter.h"
#include "tf/struct/Weight.h"
#include "tf/util/LRUtil.h"

using namespace FileUtil;
using namespace EvaluationUtil;

/**
 * Logistic Regression Ensemble for RESCAL_RANK and HOLE
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv) {

    Parameter parameter;

    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("rescal_d", po::value<int>(&(parameter.rescal_D))->default_value(100), "number of dimensions for RESCAL")
            ("hole_d", po::value<int>(&(parameter.hole_D))->default_value(100), "number of dimensions for HOLE")
            ("transe_d", po::value<int>(&(parameter.transe_D))->default_value(100), "number of dimensions for TransE")
            ("n", po::value<int>(&(parameter.num_of_thread))->default_value(-1), "number of threads for logistic regression. -1: automatically set")
            ("n_mkl", po::value<int>(&(parameter.mkl_num_of_thread))->default_value(-1), "number of MKL threads. -1: automatically set by mkl")
            ("n_e", po::value<int>(&(parameter.num_of_thread_eval))->default_value(-1), "number of threads for evaluation. -1: automatically set")
            ("dist", po::value<bool>(&(parameter.L1_flag))->default_value(1), "1: L1 distance, 0: L2 distance")
            ("normalize", po::value<bool>(&(parameter.normalize))->default_value(0), "1: normalize the score, 0: original score")
            ("znormalize", po::value<bool>(&(parameter.znormalize))->default_value(1), "1: normalize the score using z score, 0: original score")
            ("c", po::value<value_type>(&(parameter.c))->default_value(1), "search range in liblinear")
            ("hit_rate_topk", po::value<int>(&(parameter.hit_rate_topk))->default_value(10), "hit rate@k")
            ("r_map", po::value<int>(&(parameter.num_of_replaced_entities))->default_value(100),
             "number of replaced entities for MAP evaluation")
            ("r_t", po::value<int>(&(parameter.num_of_duplicated_true_triples))->default_value(100),
             "duplicate 2 * r_t true triples, r_t faked triples for replacing subjects and r_t faked triples for replacing objects")
            ("rescal_r_epoch", po::value<int>(&(parameter.rescal_restore_epoch))->default_value(-1),
             "set restore_epoch to the last epoch, if you want to load A and R from saved files (RESCAL)")
            ("hole_r_epoch", po::value<int>(&(parameter.hole_restore_epoch))->default_value(-1),
             "set restore_epoch to the last epoch, if you want to load A and R from saved files (HOLE)")
            ("transe_r_epoch", po::value<int>(&(parameter.transe_restore_epoch))->default_value(-1),
             "set restore_epoch to the last epoch, if you want to load A and R from saved files (TransE)")
            ("rescal_path", po::value<string>(&(parameter.rescal_restore_path)), "restore folder for Rescal")
            ("hole_path", po::value<string>(&(parameter.hole_restore_path)), "restore folder for HOLE")
            ("transe_path", po::value<string>(&(parameter.transe_restore_path)), "restore folder for transE")
            ("t_path", po::value<string>(&(parameter.train_data_path)), "path to training file")
            ("v_path", po::value<string>(&(parameter.valid_data_path)), "path to validation file")
            ("e_path", po::value<string>(&(parameter.test_data_path)), "path to testing file")
            ("w_path", po::value<string>(&(parameter.weight_path))->default_value(""), "path to weight file")
            ("l_path", po::value<string>(&(parameter.log_path))->default_value("./log.csv"), "path to log file")
            ("header", po::value<bool>(&(parameter.print_log_header))->default_value(1),
             "whether to print the header of csv log")
            ("o_path", po::value<string>(&(parameter.output_path))->default_value("./output"), "path to output file");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << endl;
        return 0;
    } else if (parameter.train_data_path == "" || parameter.test_data_path == "" || parameter.valid_data_path == "") {
        cout << "Please specify path to training, validation and testing files" << endl << endl;
        cout << desc << endl;
        return 0;
    } else if (parameter.rescal_restore_path == "" || parameter.transe_restore_path == "" || parameter.hole_restore_path == "") {
        cout << "Please specify path to RESCAL, TransE and HOLE files" << endl << endl;
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

    if (parameter.num_of_thread_eval == -1) {
        int num_of_thread = std::thread::hardware_concurrency();
        parameter.num_of_thread_eval = (num_of_thread == 0) ? 1 : num_of_thread;
    }

    int num_of_training_triples;
    int num_of_testing_triples;
    int num_of_validation_triples;

    Data data;
    data.read_triples(parameter.train_data_path, parameter.valid_data_path, parameter.test_data_path);
    data.encode_triples_ensemble();

    cout << "number of threads logistic regression: " << parameter.num_of_thread << endl;
    cout << "maximum number of threads for MKL: " << mkl_get_max_threads() << endl;
    cout << "number of threads for evaluation: " << parameter.num_of_thread_eval << endl;
    cout << "num of training triples: " << data.num_of_training_triples << endl;
    cout << "num of testing triples: " << data.num_of_testing_triples << endl;
    cout << "num of validation triples: " << data.num_of_validation_triples << endl;
    cout << "Entity: " << data.N << endl;
    cout << "Relation: " << data.K << endl;
    cout << "RESCAL_Dimension: " << parameter.rescal_D << endl;
    cout << "HOLE_Dimension: " << parameter.hole_D << endl;
    cout << "TransE_Dimension: " << parameter.transe_D << endl;
    cout << "------------------------" << endl;

    DFTI_DESCRIPTOR_HANDLE descriptor;

#ifdef use_double
    DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_REAL, 1, parameter.hole_D);
#else
    DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_REAL, 1, _parameter->hole_D);
#endif

    value_type scale = 1.0 / parameter.hole_D;
    DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, scale); //Scale down the output
    DftiSetValue(descriptor, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE); //Out of place FFT
    DftiCommitDescriptor(descriptor);

    DenseMatrix rescal_A(data.N, parameter.rescal_D);
    vector<DenseMatrix> rescal_R;
    rescal_R.resize(data.K, DenseMatrix(parameter.rescal_D, parameter.rescal_D));

    DenseMatrix hole_E(data.N, parameter.hole_D);
    DenseMatrix hole_P(data.K, parameter.hole_D);

    DenseMatrix transe_A(data.N, parameter.transe_D);
    DenseMatrix transe_R(data.K, parameter.transe_D);

    read_rescal_matrices(rescal_A, rescal_R, parameter.rescal_restore_epoch, parameter.rescal_restore_path);
    read_transe_matrices(hole_E, hole_P, parameter.hole_restore_epoch, parameter.hole_restore_path);
    read_transe_matrices(transe_A, transe_R, parameter.transe_restore_epoch, parameter.transe_restore_path);

    vector<SimpleWeight> weights(data.K);
    vector<min_max> min_max_values(data.K);

    if(parameter.weight_path=="") {
        LRUtil::learn_weights_RHT(&data, &parameter, rescal_A, rescal_R, hole_E, hole_P, transe_A, transe_R, weights, descriptor, min_max_values);
        FileUtil::output_weights(weights, min_max_values, parameter.output_path, true);
    } else {
        cout << "read weight from file: " << parameter.weight_path << endl;
        FileUtil::read_weights(weights, min_max_values, parameter.weight_path, true);
    }

    Monitor timer;
    cout << "start evaluation" << endl;

    timer.start();

    hit_rate measure = eval_hit_rate(Method::m_RHTLREnsemble, &parameter, &data, &rescal_A, &rescal_R,
                                     &transe_A, &transe_R, &hole_E, &hole_P, &descriptor, &weights,
                                     parameter.output_path + "/rht_lr_ensemble", &min_max_values);

    pair <value_type, value_type> map = eval_MAP(m_RHTLREnsemble, &parameter, &data, &rescal_A, &rescal_R, &transe_A, &transe_R, &hole_E,
                                                 &hole_P, &descriptor, &weights, &min_max_values);

    timer.stop();

    cout << "restore to epoch (rescal) " << parameter.rescal_restore_epoch << endl;
    cout << "restore to epoch (hole) " << parameter.hole_restore_epoch << endl;
    cout << "restore to epoch (transe) " << parameter.transe_restore_epoch << endl;

    if(measure.count_s != -1) {
        cout << "hit_rate_subject@" << parameter.hit_rate_topk
             << ": " << measure.count_s
             << ", hit_rate_object@" << parameter.hit_rate_topk << ": " << measure.count_o << ", subject_ranking: "
             << measure.count_s_ranking << ", object_ranking: " << measure.count_o_ranking << endl;
    }

    cout << "hit_rate_subject_filter@" << parameter.hit_rate_topk << ": " << measure.count_s_filtering
         << ", hit_rate_object_filter@" << parameter.hit_rate_topk << ": " << measure.count_o_filtering
         << ", subject_ranking_filter: "
         << measure.count_s_ranking_filtering << ", object_ranking_filter: " << measure.count_o_ranking_filtering
         << endl;

    if(measure.inv_count_s_ranking!=-1){
        cout << "subject_MRR: " << measure.inv_count_s_ranking << ", object_MRR: " << measure.inv_count_o_ranking << endl;
    }

    cout << "subject_MRR_filter: " << measure.inv_count_s_ranking_filtering << ", object_MRR_filter: " << measure.inv_count_o_ranking_filtering << endl;

    string prefix = "MAP evalution >>> ";
    print_map(prefix, parameter.num_of_replaced_entities, map);

    cout << "evaluation time: " << timer.getElapsedTime() << " secs" << endl;

    return 0;
}