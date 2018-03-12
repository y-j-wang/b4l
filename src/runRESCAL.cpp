#include "tf/util/Base.h"
#include "tf/alg/RESCAL.h"
#include "tf/util/Parameter.h"
#include "tf/util/Data.h"


int main(int argc, char **argv) {

    Parameter parameter;
    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("d", po::value<int>(&(parameter.rescal_D))->default_value(100), "number of dimensions")
            ("eval_train", po::value<bool>(&(parameter.eval_train))->default_value(0), "whether the program shows evaluation of training data")
            ("eval_rel", po::value<bool>(&(parameter.eval_rel))->default_value(0), "whether the program shows evaluation of replacing relations")
            ("eval_map", po::value<bool>(&(parameter.eval_map))->default_value(1), "whether the program shows evaluation of mean average precision")
            ("compute_fit", po::value<bool>(&(parameter.compute_fit))->default_value(0), "whether the program computes loss values for each epoch")
            ("lambdaA", po::value<value_type>(&(parameter.lambdaA))->default_value(0), "regularization weight for entity")
            ("lambdaR", po::value<value_type>(&(parameter.lambdaR))->default_value(0), "regularization weight for relation")
            ("t_sample", po::value<value_type>(&(parameter.train_sample_percentage))->default_value(0.01), "sample percentage for evaluating training data")
            ("epoch", po::value<int>(&(parameter.epoch))->default_value(50), "maximum training epoch")
            ("n", po::value<int>(&(parameter.num_of_thread))->default_value(1), "number of threads")
            ("n_mkl", po::value<int>(&(parameter.mkl_num_of_thread))->default_value(-1), "number of MKL threads. -1: automatically set by mkl")
            ("n_e", po::value<int>(&(parameter.num_of_thread_eval))->default_value(-1), "number of threads for evaluation. -1: automatically set")
            ("p_epoch", po::value<int>(&(parameter.print_epoch))->default_value(1), "print statistics every p_epoch")
            ("o_epoch", po::value<int>(&(parameter.output_epoch))->default_value(50), "output A and R every o_epoch")
            ("hit_rate_topk", po::value<int>(&(parameter.hit_rate_topk))->default_value(10), "hit rate@k")
            ("r_map", po::value<int>(&(parameter.num_of_replaced_entities))->default_value(100), "number of replaced entities for MAP evaluation")
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
    }

    srand(time(NULL));

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
    data.encode_triples_rescal();
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

    RESCAL rescal(&parameter,  &data);
    rescal.als();

    return 0;
}
