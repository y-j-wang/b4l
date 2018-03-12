#include <tf/util/Data.h>
#include "tf/util/Base.h"

int main(int argc, char **argv) {

    string train_file, positive_valid_file, negative_valid_file, positive_test_file, negative_test_file;
    // results from outputScore
    string data_folder;

    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("t_path", po::value<string>(&(train_file)), "path to training file")
            ("v_p_path", po::value<string>(&(positive_valid_file)), "path to positive validation file")
            ("e_p_path", po::value<string>(&(positive_test_file)), "path to positive testing file")
            ("v_n_path", po::value<string>(&(negative_valid_file)), "path to negative validation file")
            ("e_n_path", po::value<string>(&(negative_test_file)), "path to negative testing file")
            ("s_path", po::value<string>(&(data_folder)), "path to score list from outputScore");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << endl;
        return 0;
    }

    Data data;
    data.read_triples_separate(train_file, positive_valid_file, negative_valid_file, positive_test_file, negative_test_file);
    data.encode_triples_triple_classification();

    cout << "num of training triples: " << data.num_of_training_triples << endl;
    cout << "num of positive testing triples: " << data.num_of_testing_triples_positive << endl;
    cout << "num of negative testing triples: " << data.num_of_testing_triples_negative << endl;
    cout << "num of positive validation triples: " << data.num_of_validation_triples_positive << endl;
    cout << "num of negative validation triples: " << data.num_of_validation_triples_negative << endl;
    cout << "Entity: " << data.N << endl;
    cout << "Relation: " << data.K << endl;

    int global_current_valid = 0;
    int global_current_test = 0;

    for (int rel_id = 0; rel_id < data.K; rel_id++) {

        cout << "------------------------" << endl;
        cout << "Relation " << rel_id << endl;
        vector<pair<int, value_type > > valid_scores;
        vector<pair<int, value_type > > test_scores;

        string valid_file_name = data_folder + "/valid_" + to_string(rel_id) + ".dat";
        string line;

        ifstream valid_data_file(valid_file_name.c_str());

        while (getline(valid_data_file, line)){
            if (line == ""){
                continue;
            }

            vector<string> par;
            boost::split(par, line, boost::is_any_of(","));

            valid_scores.push_back(make_pair(stoi(par[0]), stod(par[1])));
        }
        valid_data_file.close();

        if(valid_scores.size()==0){
            cout << "Validation Set: Relation " << rel_id << " is empty" << endl;
            continue;
        }

        string test_file_name = data_folder + "/test_" + to_string(rel_id) + ".dat";

        ifstream test_data_file(test_file_name.c_str());

        while (getline(test_data_file, line)){
            if (line == ""){
                continue;
            }

            vector<string> par;
            boost::split(par, line, boost::is_any_of(","));

            test_scores.push_back(make_pair(stoi(par[0]), stod(par[1])));
        }
        test_data_file.close();

        if(test_scores.size()==0){
            cout << "Test Set: Relation " << rel_id << " is empty" << endl;
            continue;
        }

        value_type best_threshold = -1;
        int best_correct_valid;
        value_type best_result = 0;

        for (int index = 0; index < valid_scores.size(); index++) {

            pair<int, value_type> &score = valid_scores[index];

            int correct_valid = 0;
            for (int i = 0; i < valid_scores.size(); i++) {
                if(valid_scores[i].second >= score.second) {
                    // predict as positive
                    if (valid_scores[i].first == 1){
                        correct_valid++;
                    }
                } else {
                    // predict as negative
                    if (valid_scores[i].first == 0){
                        correct_valid++;
                    }
                }
            }

            value_type accuracy = (correct_valid + 0.0) / valid_scores.size();

            if (accuracy >= best_result) {
                best_threshold = score.second;
                best_correct_valid = correct_valid;
                best_result = accuracy;
            }
        }

        cout << "best threshold on validation data: " << best_threshold << endl;
        cout << "validation accuracy: " << best_result << endl;

        int correct_test = 0;
        for (int i = 0; i < test_scores.size(); i++) {
            if (test_scores[i].second >= best_threshold) {
                // predict as positive
                if (test_scores[i].first == 1) {
                    correct_test++;
                }
            } else {
                // predict as negative
                if (test_scores[i].first == 0) {
                    correct_test++;
                }
            }
        }

        value_type t_accuracy = (correct_test + 0.0) / test_scores.size();
        cout << "testing accuracy: " << t_accuracy << endl;

        global_current_valid += best_correct_valid;
        global_current_test += correct_test;
    }

    cout << "------------------------" << endl;

    cout << "accuracy on validation data: " << global_current_valid / (data.num_of_validation_triples_positive + data.num_of_validation_triples_negative + 0.0) << endl;
    cout << "accuracy on test data: " << global_current_test / (data.num_of_testing_triples_positive + data.num_of_testing_triples_negative+0.0) << endl;
}

