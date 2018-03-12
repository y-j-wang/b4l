#ifndef FILEUTIL_H
#define FILEUTIL_H

#include "tf/util/Base.h"
#include "tf/struct/Triple.h"
#include "tf/struct/Weight.h"
#include "tf/struct/Min_Max.h"
#include <fstream>

using std::fstream;
using std::ifstream;
using std::ofstream;


namespace FileUtil {

    void read_triple_data(string data_path, vector<Triple<string> > &triples, int &num_of_triple, const bool init=true) {

        ifstream data_file(data_path.c_str());
        if(!data_file.good()){
            cerr << "cannot find file: " << data_path << endl;
            exit(1);
        }

        string line;

        if(init) {
            num_of_triple = 0;
        }

        while (getline(data_file, line)) {
            boost::trim(line);
            if (line.length() == 0) {
                continue;
            }
            vector<string> par;
            boost::split(par, line, boost::is_any_of("\t"));

            boost::trim(par[0]);
            boost::trim(par[1]);
            boost::trim(par[2]);

            triples.push_back(Triple<string>(par[0], par[1], par[2]));

            num_of_triple++;
        }

        data_file.close();

    }

    void output_matrices(DenseMatrix &A, vector<DenseMatrix> &R, const int epoch, string output_folder) {

        string epoch_folder = output_folder + "/" + to_string(epoch);
        boost::filesystem::create_directories(epoch_folder);

        string A_file_name = epoch_folder + "/A_" + to_string(epoch) + ".dat";

        ofstream A_output(A_file_name.c_str());
        for (int row = 0; row < A.size1(); row++) {
            for (int col = 0; col < A.size2(); col++) {
                A_output << A(row, col) << " ";
            }
            A_output << endl;
        }
        A_output.close();

        for (int k = 0; k < R.size(); k++) {
            string R_path = epoch_folder + "/R_" + to_string(k) + "_" + to_string(epoch) + ".dat";
            DenseMatrix &subR = R[k];
            ofstream R_output(R_path.c_str());
            for (int row = 0; row < subR.size1(); row++) {
                for (int col = 0; col < subR.size2(); col++) {
                    R_output << subR(row, col) << " ";
                }
                R_output << endl;
            }
            R_output.close();
        }

    }

    void output_matrices(DenseMatrix &A, vector<DenseMatrix> &R, DenseMatrix &A_G, vector<DenseMatrix> &R_G, const int epoch, string output_folder) {

        string epoch_folder = output_folder + "/" + to_string(epoch);
        boost::filesystem::create_directories(epoch_folder);

        string A_file_name = epoch_folder + "/A_" + to_string(epoch) + ".dat";

        ofstream A_output(A_file_name.c_str());
        for (int row = 0; row < A.size1(); row++) {
            for (int col = 0; col < A.size2(); col++) {
                A_output << A(row, col) << " ";
            }
            A_output << endl;
        }
        A_output.close();

        string A_G_file_name = epoch_folder + "/A_G_" + to_string(epoch) + ".dat";

        ofstream A_G_output(A_G_file_name.c_str());
        for (int row = 0; row < A_G.size1(); row++) {
            for (int col = 0; col < A_G.size2(); col++) {
                A_G_output << A_G(row, col) << " ";
            }
            A_G_output << endl;
        }
        A_G_output.close();

        for (int k = 0; k < R.size(); k++) {
            string R_path = epoch_folder + "/R_" + to_string(k) + "_" + to_string(epoch) + ".dat";
            DenseMatrix &subR = R[k];
            ofstream R_output(R_path.c_str());
            for (int row = 0; row < subR.size1(); row++) {
                for (int col = 0; col < subR.size2(); col++) {
                    R_output << subR(row, col) << " ";
                }
                R_output << endl;
            }
            R_output.close();

            string R_G_path = epoch_folder + "/R_G_" + to_string(k) + "_" + to_string(epoch) + ".dat";
            DenseMatrix &subR_G = R_G[k];
            ofstream R_G_output(R_G_path.c_str());
            for (int row = 0; row < subR_G.size1(); row++) {
                for (int col = 0; col < subR_G.size2(); col++) {
                    R_G_output << subR_G(row, col) << " ";
                }
                R_G_output << endl;
            }
            R_G_output.close();
        }

    }

    void output_matrices(DenseMatrix &A, DenseMatrix &R, const int N, const int D, const int K,
                         const int epoch, string output_folder) {

        value_type *entity_vec = A.data().begin();
        value_type *relation_vec = R.data().begin();

        string epoch_folder = output_folder + "/" + to_string(epoch);
        boost::filesystem::create_directories(epoch_folder);

        string A_file_name = epoch_folder + "/A_" + to_string(epoch) + ".dat";

        ofstream A_output(A_file_name.c_str());
        for (int row = 0; row < N; row++) {
            for (int col = 0; col < D; col++) {
                A_output << entity_vec[row * D + col] << " ";
            }
            A_output << endl;
        }
        A_output.close();

        string R_file_name = epoch_folder + "/R_" + to_string(epoch) + ".dat";

        ofstream R_output(R_file_name.c_str());
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < D; col++) {
                R_output << relation_vec[row * D + col] << " ";
            }
            R_output << endl;
        }
        A_output.close();

    }

    void output_matrices(DenseMatrix &A, DenseMatrix &R, DenseMatrix &A_G, DenseMatrix &R_G, const int N, const int D, const int K,
                         const int epoch, string output_folder) {

        value_type *entity_vec = A.data().begin();
        value_type *relation_vec = R.data().begin();

        string epoch_folder = output_folder + "/" + to_string(epoch);
        boost::filesystem::create_directories(epoch_folder);

        string A_file_name = epoch_folder + "/A_" + to_string(epoch) + ".dat";

        ofstream A_output(A_file_name.c_str());
        for (int row = 0; row < N; row++) {
            for (int col = 0; col < D; col++) {
                A_output << entity_vec[row * D + col] << " ";
            }
            A_output << endl;
        }
        A_output.close();

        string A_G_file_name = epoch_folder + "/A_G_" + to_string(epoch) + ".dat";

        ofstream A_G_output(A_G_file_name.c_str());
        for (int row = 0; row < A_G.size1(); row++) {
            for (int col = 0; col < A_G.size2(); col++) {
                A_G_output << A_G(row, col) << " ";
            }
            A_G_output << endl;
        }
        A_G_output.close();

        string R_file_name = epoch_folder + "/R_" + to_string(epoch) + ".dat";

        ofstream R_output(R_file_name.c_str());
        for (int row = 0; row < K; row++) {
            for (int col = 0; col < D; col++) {
                R_output << relation_vec[row * D + col] << " ";
            }
            R_output << endl;
        }
        A_output.close();

        string R_G_file_name = epoch_folder + "/R_G_" + to_string(epoch) + ".dat";

        ofstream R_G_output(R_G_file_name.c_str());
        for (int row = 0; row < R_G.size1(); row++) {
            for (int col = 0; col < R_G.size2(); col++) {
                R_G_output << R_G(row, col) << " ";
            }
            R_G_output << endl;
        }
        R_G_output.close();
    }

    void output_weights(vector<SimpleWeight> &weights, const int epoch, string output_folder) {

        string epoch_folder = output_folder + "/" + to_string(epoch);
        boost::filesystem::create_directories(epoch_folder);

        string file_name = epoch_folder + "/W_" + to_string(epoch) + ".dat";

        ofstream output(file_name.c_str());
        for(SimpleWeight weight:weights){
            output << weight.w1 << " " << weight.w2 << endl;
        }
        output.close();

    }

    void output_weights(vector<SimpleWeight> &weights, vector<min_max> &min_max_values, string output_folder, const int three=false) {

        boost::filesystem::create_directories(output_folder);

        string file_name = output_folder + "/W.dat";

        ofstream output(file_name.c_str());
        int rel_index = 0;
        for (SimpleWeight weight:weights) {
            output << weight.w1 << " " << weight.w2;
            if(three){
                output << " " << weight.w3;
            }

            output << " " << min_max_values[rel_index].max1 << " " << min_max_values[rel_index].min1 << " "
                   << min_max_values[rel_index].max2 << " " << min_max_values[rel_index].min2;
            if (three) {
                output << " " << min_max_values[rel_index].max3 << " " << min_max_values[rel_index].min3;
            }

            output << endl;
            rel_index++;
        }
        output.close();

    }

    void output_weights(vector<SimpleWeight> &weights, string output_folder, const int three=false) {

        boost::filesystem::create_directories(output_folder);

        string file_name = output_folder + "/W.dat";

        ofstream output(file_name.c_str());
        for (SimpleWeight weight:weights) {
            output << weight.w1 << " " << weight.w2;
            if(three){
                output << " " << weight.w3;
            }
            output << endl;
        }
        output.close();

    }

    void read_transe_matrices(DenseMatrix &transeA, DenseMatrix &transeR, const int epoch, string input_folder = "./output") {

        int D = transeA.size2();
        value_type *A = transeA.data().begin();
        value_type *R = transeR.data().begin();

        string A_file_name = input_folder + "/A_" + to_string(epoch) + ".dat";

        string line;

        ifstream A_input(A_file_name.c_str());

        if(!A_input.good()){
            cerr << "cannot find file: " << A_file_name << endl;
            exit(1);
        }

        int row_index = 0;

        while (getline(A_input, line)){
            if (line == ""){
                continue;
            }

            vector<string> par;
            boost::split(par, line, boost::is_any_of(" "));

            for (int col_index = 0; col_index < D; col_index++) {
                A[row_index * D + col_index] = stod(par[col_index]);
            }
            row_index++;
        }
        A_input.close();

        string R_path = input_folder + "/R_" + to_string(epoch) + ".dat";

        ifstream R_input(R_path.c_str());

        if(!R_input.good()){
            cerr << "cannot find file: " << R_path << endl;
            exit(1);
        }

        row_index = 0;
        while (getline(R_input, line)){
            if (line == ""){
                continue;
            }

            vector<string> par;
            boost::split(par, line, boost::is_any_of(" "));
            for (int col_index = 0; col_index < D; col_index++) {
                R[row_index * D + col_index] = stod(par[col_index]);
            }
            row_index++;
        }

        R_input.close();

    }

    void read_transe_matrices(DenseMatrix &transeA, DenseMatrix &transeR, const int D, const int epoch, string input_folder = "./output") {

        value_type *A = transeA.data().begin();
        value_type *R = transeR.data().begin();

        string A_file_name = input_folder + "/A_" + to_string(epoch) + ".dat";

        string line;

        ifstream A_input(A_file_name.c_str());

        if(!A_input.good()){
            cerr << "cannot find file: " << A_file_name << endl;
            exit(1);
        }

        int row_index = 0;

        while (getline(A_input, line)){
            if (line == ""){
                continue;
            }

            vector<string> par;
            boost::split(par, line, boost::is_any_of(" "));

            for (int col_index = 0; col_index < D; col_index++) {
                A[row_index * D + col_index] = stod(par[col_index]);
            }
            row_index++;
        }
        A_input.close();

        string R_path = input_folder + "/R_" + to_string(epoch) + ".dat";

        ifstream R_input(R_path.c_str());

        if(!R_input.good()){
            cerr << "cannot find file: " << R_path << endl;
            exit(1);
        }

        row_index = 0;
        while (getline(R_input, line)){
            if (line == ""){
                continue;
            }

            vector<string> par;
            boost::split(par, line, boost::is_any_of(" "));
            for (int col_index = 0; col_index < D; col_index++) {
                R[row_index * D + col_index] = stod(par[col_index]);
            }
            row_index++;
        }

        R_input.close();

    }

    void read_transe_matrices(DenseMatrix &transeA, DenseMatrix &transeR, DenseMatrix &transeA_G, DenseMatrix &transeR_G, const int D, const int epoch, string input_folder = "./output") {

        value_type *A = transeA.data().begin();
        value_type *A_G = transeA_G.data().begin();
        value_type *R = transeR.data().begin();
        value_type *R_G = transeR_G.data().begin();

        string A_file_name = input_folder + "/A_" + to_string(epoch) + ".dat";

        string line;

        ifstream A_input(A_file_name.c_str());

        if(!A_input.good()){
            cerr << "cannot find file: " << A_file_name << endl;
            exit(1);
        }

        int row_index = 0;

        while (getline(A_input, line)){
            if (line == ""){
                continue;
            }

            vector<string> par;
            boost::split(par, line, boost::is_any_of(" "));

            for (int col_index = 0; col_index < D; col_index++) {
                A[row_index * D + col_index] = stod(par[col_index]);
            }
            row_index++;
        }
        A_input.close();

        string A_G_file_name = input_folder + "/A_G_" + to_string(epoch) + ".dat";

        ifstream A_G_input(A_G_file_name.c_str());

        if(!A_G_input.good()){
            cerr << "cannot find file: " << A_G_file_name << endl;
            exit(1);
        }

        row_index = 0;

        while (getline(A_G_input, line)){
            if (line == ""){
                continue;
            }

            vector<string> par;
            boost::split(par, line, boost::is_any_of(" "));

            for (int col_index = 0; col_index < D; col_index++) {
                A_G[row_index * D + col_index] = stod(par[col_index]);
            }
            row_index++;
        }
        A_G_input.close();

        string R_path = input_folder + "/R_" + to_string(epoch) + ".dat";

        ifstream R_input(R_path.c_str());

        if(!R_input.good()){
            cerr << "cannot find file: " << R_path << endl;
            exit(1);
        }

        row_index = 0;
        while (getline(R_input, line)){
            if (line == ""){
                continue;
            }

            vector<string> par;
            boost::split(par, line, boost::is_any_of(" "));
            for (int col_index = 0; col_index < D; col_index++) {
                R[row_index * D + col_index] = stod(par[col_index]);
            }
            row_index++;
        }

        R_input.close();

        string R_G_file_name = input_folder + "/R_G_" + to_string(epoch) + ".dat";

        ifstream R_G_input(R_G_file_name.c_str());

        if(!R_G_input.good()){
            cerr << "cannot find file: " << R_G_file_name << endl;
            exit(1);
        }

        row_index = 0;

        while (getline(R_G_input, line)){
            if (line == ""){
                continue;
            }

            vector<string> par;
            boost::split(par, line, boost::is_any_of(" "));

            for (int col_index = 0; col_index < D; col_index++) {
                R_G[row_index * D + col_index] = stod(par[col_index]);
            }
            row_index++;
        }
        R_G_input.close();

    }

    void read_rescal_matrices(DenseMatrix &A, vector<DenseMatrix> &R, const int epoch, string input_folder = "./output") {

        string A_file_name = input_folder + "/A_" + to_string(epoch) + ".dat";

        string line;

        ifstream A_input(A_file_name.c_str());

        if(!A_input.good()){
            cerr << "cannot find file: " << A_file_name << endl;
            exit(1);
        }

        int row_index = 0;

        while (getline(A_input, line)){
            if (line == ""){
                continue;
            }

            vector<string> par;
            boost::split(par, line, boost::is_any_of(" "));

            for (int col_index = 0; col_index < A.size2(); col_index++) {
                A(row_index, col_index) = stod(par[col_index]);
            }
            row_index++;
        }
        A_input.close();

        for (int k = 0; k < R.size(); k++) {
            string R_path = input_folder + "/R_" + to_string(k) + "_" + to_string(epoch) + ".dat";
            DenseMatrix &subR = R[k];
            ifstream R_input(R_path.c_str());

            if(!R_input.good()){
                cerr << "cannot find file: " << R_path << endl;
                exit(1);
            }

            row_index = 0;
            while (getline(R_input, line)){
                if (line == ""){
                    continue;
                }

                vector<string> par;
                boost::split(par, line, boost::is_any_of(" "));
                for (int col_index = 0; col_index < subR.size2(); col_index++) {
                    subR(row_index, col_index) = stod(par[col_index]);
                }
                row_index++;
            }

            R_input.close();
        }

    }

    void read_rescal_rank_matrices(DenseMatrix &A, vector<DenseMatrix> &R, DenseMatrix &A_G, vector<DenseMatrix> &R_G, const int epoch, string input_folder = "./output") {

        string A_file_name = input_folder + "/A_" + to_string(epoch) + ".dat";

        string line;

        ifstream A_input(A_file_name.c_str());

        if(!A_input.good()){
            cerr << "cannot find file: " << A_file_name << endl;
            exit(1);
        }

        int row_index = 0;

        while (getline(A_input, line)){
            if (line == ""){
                continue;
            }

            vector<string> par;
            boost::split(par, line, boost::is_any_of(" "));

            for (int col_index = 0; col_index < A.size2(); col_index++) {
                A(row_index, col_index) = stod(par[col_index]);
            }
            row_index++;
        }
        A_input.close();

        string A_G_file_name = input_folder + "/A_G_" + to_string(epoch) + ".dat";

        ifstream A_G_input(A_G_file_name.c_str());

        if(!A_G_input.good()){
            cerr << "cannot find file: " << A_G_file_name << endl;
            exit(1);
        }

        row_index = 0;
        while (getline(A_G_input, line)){
            if (line == ""){
                continue;
            }

            vector<string> par;
            boost::split(par, line, boost::is_any_of(" "));

            for (int col_index = 0; col_index < A_G.size2(); col_index++) {
                A_G(row_index, col_index) = stod(par[col_index]);
            }
            row_index++;
        }
        A_G_input.close();

        for (int k = 0; k < R.size(); k++) {
            string R_path = input_folder + "/R_" + to_string(k) + "_" + to_string(epoch) + ".dat";
            DenseMatrix &subR = R[k];
            ifstream R_input(R_path.c_str());

            if(!R_input.good()){
                cerr << "cannot find file: " << R_path << endl;
                exit(1);
            }

            row_index = 0;
            while (getline(R_input, line)){
                if (line == ""){
                    continue;
                }

                vector<string> par;
                boost::split(par, line, boost::is_any_of(" "));
                for (int col_index = 0; col_index < subR.size2(); col_index++) {
                    subR(row_index, col_index) = stod(par[col_index]);
                }
                row_index++;
            }

            R_input.close();

            string R_G_path = input_folder + "/R_G_" + to_string(k) + "_" + to_string(epoch) + ".dat";
            DenseMatrix &subR_G = R_G[k];
            ifstream R_G_input(R_G_path.c_str());

            if(!R_G_input.good()){
                cerr << "cannot find file: " << R_G_path << endl;
                exit(1);
            }

            row_index = 0;
            while (getline(R_G_input, line)){
                if (line == ""){
                    continue;
                }

                vector<string> par;
                boost::split(par, line, boost::is_any_of(" "));
                for (int col_index = 0; col_index < subR_G.size2(); col_index++) {
                    subR_G(row_index, col_index) = stod(par[col_index]);
                }
                row_index++;
            }

            R_G_input.close();
        }

    }

    void read_weights(vector<SimpleWeight> &weights, string weights_file_path, const bool three=false) {

        string line;

        ifstream W_input(weights_file_path.c_str());

        if(!W_input.good()){
            cerr << "cannot find file: " << weights_file_path << endl;
            exit(1);
        }

        int row_index = 0;

        while (getline(W_input, line)){
            if (line == ""){
                continue;
            }

            vector<string> par;
            boost::split(par, line, boost::is_any_of(" "));

            weights[row_index].w1 = stod(par[0]);
            weights[row_index].w2 = stod(par[1]);
            if (three) {
                weights[row_index].w3 = stod(par[2]);
            }
            row_index++;
        }
        W_input.close();
    }

    void read_weights(vector<SimpleWeight> &weights, vector<min_max> &min_max_values, string weights_file_path, const bool three=false) {

        string line;

        ifstream W_input(weights_file_path.c_str());

        if(!W_input.good()){
            cerr << "cannot find file: " << weights_file_path << endl;
            exit(1);
        }

        int row_index = 0;

        while (getline(W_input, line)){
            if (line == ""){
                continue;
            }

            vector<string> par;
            boost::split(par, line, boost::is_any_of(" "));

            weights[row_index].w1 = stod(par[0]);
            weights[row_index].w2 = stod(par[1]);

            if (three) {
                weights[row_index].w3 = stod(par[2]);

                min_max_values[row_index].max1 = stod(par[3]);
                min_max_values[row_index].min1 = stod(par[4]);

                min_max_values[row_index].max2 = stod(par[5]);
                min_max_values[row_index].min2 = stod(par[6]);

                min_max_values[row_index].max3 = stod(par[7]);
                min_max_values[row_index].min3 = stod(par[8]);
            } else {

                min_max_values[row_index].max1 = stod(par[2]);
                min_max_values[row_index].min1 = stod(par[3]);

                min_max_values[row_index].max2 = stod(par[4]);
                min_max_values[row_index].min2 = stod(par[5]);
            }

            row_index++;
        }
        W_input.close();
    }


    void output_statistic_list(vector<value_type> &objs, const int print_epoch, const int current_epoch, const string obj_file_name, const string output_folder = "./output") {
        boost::filesystem::create_directories(output_folder);

        ofstream output(obj_file_name.c_str(), std::ofstream::out | std::ofstream::app);
        for (int i = 0; i < objs.size(); i++) {
            output << current_epoch + i * print_epoch << " " << objs[i] << endl;
        }
        output.close();

    }
}

#endif //FILEUTIL_H
