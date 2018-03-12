#ifndef ENSEMBLE_H
#define ENSEMBLE_H

#include "tf/util/Base.h"
#include "tf/alg/RESCAL_RANK.h"
#include "tf/alg/TransE.h"
#include <mlpack/core.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include <tf/util/LRUtil.h>

using namespace EvaluationUtil;
using namespace Calculator;
using namespace FileUtil;
using namespace mlpack;

class Ensemble: public TransE, public RESCAL_RANK {

private:

    vector<SimpleWeight> weights;
    vector<arma::mat> predictors;
    vector<arma::Row<size_t> > responses;

    void initialize(){
        current_epoch = 1;
        TransE::initialize();
        RESCAL_RANK::initialize();

        weights.resize(data->K);

        for (int i = 0; i < data->K; i++) {
            weights[i].w1 = 0.5;
            weights[i].w2 = 0.5;
        }

        predictors.resize(data->K);
        responses.resize(data->K);
    }

    void sample_LR_data(vector<arma::mat> &predictors, vector<arma::Row<size_t> > &responses){

        int sample_size = data->num_of_training_triples * parameter->train_sample_percentage;
        vector<vector<pair<Triple<int>, int> > > sampled_training_triples(data->K, vector<pair<Triple<int>, int> >());

        int inner_rel_index = 0;
        for (int i = 0; i < sample_size; i++) {

            int random_triple_id = RandomUtil::uniform_int(0, data->relation2tupleList_mapping[inner_rel_index].size());
            Tuple<int> &trueTuple = data->relation2tupleList_mapping[inner_rel_index][random_triple_id];

            vector<pair<Triple<int>, int> > &sub_sampled_training_triples = sampled_training_triples[inner_rel_index];

            for (int count = 0; count < parameter->num_of_duplicated_true_triples; count++) {
                sub_sampled_training_triples.push_back(make_pair(Triple<int>(trueTuple.subject, inner_rel_index, trueTuple.object), 1));
            }

            // first replace subjects
            for (int count = 0; count < parameter->num_of_replaced_entities; count++) {
                int random_entity_id = RandomUtil::uniform_int(0, data->N);
                while(data->faked_tuple_exist_train(inner_rel_index, random_entity_id, trueTuple.object)){
                    random_entity_id = RandomUtil::uniform_int(0, data->N);
                }
                sub_sampled_training_triples.push_back(make_pair(Triple<int>(random_entity_id, inner_rel_index, trueTuple.object), 0));
            }

            // then replace objects
            for (int count = 0; count < parameter->num_of_replaced_entities; count++) {
                int random_entity_id = RandomUtil::uniform_int(0, data->N);
                while(data->faked_tuple_exist_train(inner_rel_index, trueTuple.subject, random_entity_id)){
                    random_entity_id = RandomUtil::uniform_int(0, data->N);
                }
                sub_sampled_training_triples.push_back(make_pair(Triple<int>(trueTuple.subject, inner_rel_index, random_entity_id), 0));
            }

            inner_rel_index++;

            if(inner_rel_index == data->K){
                inner_rel_index = 0;
            }
        }

        for (int rel_id = 0; rel_id < data->K; rel_id++) {
            vector<pair<Triple<int>, int> > &sub_sampled_training_triples = sampled_training_triples[rel_id];
            value_type *predictor = new value_type[sub_sampled_training_triples.size() * 2];
            size_t *response = new size_t[sub_sampled_training_triples.size()];
            int index1 = 0;
            int index2 = 0;

            for(auto p:sub_sampled_training_triples){
                predictor[index1] = cal_rescal_score(rel_id, p.first.subject, p.first.object, parameter->rescal_D, RESCAL_RANK::rescalA, RESCAL_RANK::rescalR);
                index1++;
                predictor[index1] = - cal_transe_score(p.first.subject, p.first.object, rel_id, parameter->transe_D, parameter->L1_flag, TransE::transeA, TransE::transeR);
                index1++;
                response[index2] = p.second;
                index2++;
            }

            predictors[rel_id] = arma::mat(predictor, 2, sub_sampled_training_triples.size(), true, true);
            responses[rel_id] = arma::Row<size_t>(response, sub_sampled_training_triples.size(), true, true);

            delete[] predictor;
            delete[] response;
        }

    }

    void update(Sample &sample, const value_type weight = 1.0) {
        // violation will be the sum of two
        RESCAL_RANK::update(sample, weights[sample.relation_id].w1);
        TransE::update(sample, weights[sample.relation_id].w2);
    }
    virtual string eval(const int epoch) {
        hit_rate measure = eval_hit_rate(Method::m_Ensemble, parameter, data, &(RESCAL_RANK::rescalA), &(RESCAL_RANK::rescalR), &(TransE::transeA),
                                         &(TransE::transeR), nullptr, nullptr, nullptr, &weights, parameter->output_path + "/ensemble");

        string prefix = "Ensemble >>> ";

        print_hit_rate(prefix, parameter->hit_rate_topk, measure);

        if (parameter->eval_map) {
            pair<value_type, value_type> map = eval_MAP(m_Ensemble, parameter, data, &(RESCAL_RANK::rescalA), &(RESCAL_RANK::rescalR), &(TransE::transeA), &(TransE::transeR), nullptr, nullptr, nullptr, &weights);

            string prefix = "MAP evalution >>> ";
            print_map(prefix, parameter->num_of_replaced_entities, map);
        }

        // ToDo
        return "";
    };

    virtual void output(const int epoch) {
        output_matrices(RESCAL_RANK::rescalA, RESCAL_RANK::rescalR, RESCAL_RANK::rescalA_G, RESCAL_RANK::rescalR_G,
                        epoch,
                        parameter->output_path + "/Ensemble/RESCAL_RANK");
        output_matrices(TransE::transeA, TransE::transeR, data->N, parameter->transe_D, data->K, epoch,
                        parameter->output_path + "/Ensemble/TransE");
        output_weights(weights, epoch, parameter->output_path + "/Weight");
    };

    virtual string get_log_header() {
        return "";
    };

public:
    Ensemble(Parameter *parameter, Data *data):Optimizer(parameter, data), TransE(parameter, data),
                                               RESCAL_RANK(parameter, data){}

    void train() {

        initialize();

        Sample sample;

        Monitor timer;

        std::vector<int> indices(data->num_of_training_triples);
        std::iota(std::begin(indices), std::end(indices), 0);

        if(parameter->print_log_header) {
            print_log(get_log_header());
        }

        for (int epoch = current_epoch; epoch <= parameter->epoch; epoch++) {

            violations = 0;

            timer.start();

            std::random_shuffle(indices.begin(), indices.end());

            for (int n = 0; n < data->num_of_training_triples; n++) {
                Sampler::random_sample(*data, sample, indices[n]);
                update(sample);
            }

            sample_LR_data(predictors, responses);
            LRUtil::learn_weights(data, parameter, predictors, responses, weights);

            timer.stop();

            cout << "epoch " << epoch << ", time " << timer.getElapsedTime() << " secs" << endl;

            cout << "violations: " << violations << endl;

            if (epoch % parameter->print_epoch == 0) {

                timer.start();

                string log = eval(epoch);

                timer.stop();

                cout << "evaluation time: " << timer.getElapsedTime() << " secs" << endl;

                print_log(log);
            }

            if (epoch % parameter->output_epoch == 0) {
                output(epoch);
            }
        }

    }
};

#endif //ENSEMBLE_H
