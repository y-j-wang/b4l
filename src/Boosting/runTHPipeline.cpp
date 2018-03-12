#include <tf/alg/THPipeline.h>
#include "tf/util/Base.h"
#include "tf/util/Data.h"
#include "tf/util/FileUtil.h"
#include "tf/util/Monitor.h"
#include "tf/util/EvaluationUtil.h"
#include "tf/util/Parameter.h"
#include "tf/util/LRUtil.h"
#include "tf/alg/Sampler.h"

using namespace FileUtil;
using namespace EvaluationUtil;

/**
 * Pipeline Ensemble for TransE and HOLE
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv) {

    Parameter parameter;

    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("opt", po::value<string>(&(parameter.optimization))->default_value("SGD"), "optimization method, i.e., SGD, AdaGrad or AdaDelta")
            ("transe_d", po::value<int>(&(parameter.transe_D))->default_value(100), "number of dimensions for TransE")
            ("hole_d", po::value<int>(&(parameter.hole_D))->default_value(100), "number of dimensions for HOLE")
            ("transe_margin", po::value<value_type>(&(parameter.transe_margin))->default_value(4), "parameter for margin when training TransE")
            ("hole_margin", po::value<value_type>(&(parameter.hole_margin))->default_value(4), "parameter for margin when training HOLE")
            ("step_size", po::value<value_type>(&(parameter.step_size))->default_value(0.001), "step size")
            ("dist", po::value<bool>(&(parameter.L1_flag))->default_value(1), "1: L1 distance, 0: L2 distance")
            ("epoch", po::value<int>(&(parameter.epoch))->default_value(1000), "maximum training epoch")
            ("rho", po::value<value_type>(&(parameter.Rho))->default_value(0.9), "parameter for AdaDelta")
            ("n", po::value<int>(&(parameter.num_of_thread))->default_value(1), "number of threads for logistic regression. -1: automatically set")
            ("n_mkl", po::value<int>(&(parameter.mkl_num_of_thread))->default_value(-1), "number of MKL threads. -1: automatically set by mkl")
            ("n_e", po::value<int>(&(parameter.num_of_thread_eval))->default_value(-1), "number of threads for evaluation. -1: automatically set")
            ("hit_rate_topk", po::value<int>(&(parameter.hit_rate_topk))->default_value(10), "hit rate@k")
            ("lambdaE", po::value<value_type>(&(parameter.lambdaE))->default_value(0), "regularization weight for entity")
            ("lambdaP", po::value<value_type>(&(parameter.lambdaP))->default_value(0), "regularization weight for relation")
            ("r_map", po::value<int>(&(parameter.num_of_replaced_entities))->default_value(100), "number of replaced entities for MAP evaluation")
            ("r_transe", po::value<bool>(&(parameter.restore_from_transe))->default_value(0), "use results from transE as initialization (d should be set to dimensionality of TransE model)")
            ("r_path", po::value<string>(&(parameter.transe_restore_path)), "restore folder for TransE")
            ("r_epoch", po::value<int>(&(parameter.restore_epoch))->default_value(0), "set restore_epoch to the last epoch, if you want to load A and R from saved files")
            ("r_path", po::value<string>(&(parameter.restore_path)), "restore folder")
            ("t_path", po::value<string>(&(parameter.train_data_path)), "path to training file")
            ("v_path", po::value<string>(&(parameter.valid_data_path)), "path to validation file")
            ("e_path", po::value<string>(&(parameter.test_data_path)), "path to testing file")
            ("l_path", po::value<string>(&(parameter.log_path))->default_value("./log.csv"), "path to log file")
            ("header", po::value<bool>(&(parameter.print_log_header))->default_value(1), "whether to print the header of csv log")
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
    } else {

        std::transform(parameter.optimization.begin(), parameter.optimization.end(), parameter.optimization.begin(),
                       ::tolower);

        if (parameter.optimization != "sgd" && parameter.optimization != "adagrad" &&
            parameter.optimization != "adadelta") {
            cout << "Unrecognized optimization method. opt should be SGD, AdaGrad or AdaDelta" << endl << endl;
            cout << desc << endl;
            return 0;
        }
    }

    if (parameter.mkl_num_of_thread > 0) {
        mkl_set_num_threads(parameter.mkl_num_of_thread);
    }

    if (parameter.num_of_thread_eval == -1) {
        int num_of_thread = std::thread::hardware_concurrency();
        parameter.num_of_thread_eval = (num_of_thread == 0) ? 1 : num_of_thread;
    }

    Data data;
    data.read_triples(parameter.train_data_path, parameter.valid_data_path, parameter.test_data_path);
    data.encode_triples_transe();
//    data.output_decoder(parameter.output_path);

    cout << "number of threads: " << parameter.num_of_thread << endl;
    cout << "maximum number of threads for MKL: " << mkl_get_max_threads() << endl;
    cout << "number of threads for evaluation: " << parameter.num_of_thread_eval << endl;
    cout << "num of training triples: " << data.num_of_training_triples << endl;
    cout << "num of testing triples: " << data.num_of_testing_triples << endl;
    cout << "num of validation triples: " << data.num_of_validation_triples << endl;
    cout << "Entity: " << data.N << endl;
    cout << "Relation: " << data.K << endl;
    cout << "------------------------" << endl;

    THPipeline thPipeline(&parameter, &data);
    thPipeline.train();

    return 0;
}