#ifndef THPIPELINE_H
#define THPIPELINE_H

#include "HOLE.h"
#include "TransE.h"

class THPipeline: public TransE, public HOLE {

private:

    vector<SimpleWeight> weights;

protected:

    void initialize(){
        current_epoch = 1;

        if (parameter->restore_from_transe && parameter->restore_epoch > 0 && parameter->transe_restore_path != "") {

            if (parameter->optimization == "sgd") {
                read_transe_matrices(TransE::transeA, TransE::transeR, parameter->transe_D, parameter->restore_epoch,
                                     parameter->restore_path);
            } else {
                read_transe_matrices(TransE::transeA, TransE::transeR, TransE::transeA_G, TransE::transeR_G,
                                     parameter->transe_D, parameter->restore_epoch,
                                     parameter->restore_path);
            }

            Monitor timer;
            timer.start();

            hit_rate measure = eval_hit_rate(Method::m_TransE, parameter, data, nullptr, nullptr,
                                             &transeA, &transeR, nullptr, nullptr, nullptr, nullptr,
                                             parameter->output_path + "/TransE_test_epoch_" +
                                             to_string(parameter->restore_epoch));
            timer.stop();

            string prefix = "[TransE] restore to epoch " + to_string(parameter->restore_epoch) + " ";

            print_hit_rate(prefix, parameter->hit_rate_topk, measure);

            cout << "evaluation time: " << timer.getElapsedTime() << " secs" << endl;

            cout << "------------------------" << endl;

        } else {
            TransE::initialize();
        }

        HOLE::initialize();

        weights.resize(data->K);

        for (int i = 0; i < data->K; i++) {
            weights[i].w1 = 1;
            weights[i].w2 = 1;  // never change
        }
    }

    void update(Sample &sample, const value_type weight = 1.0) {

        bool subject_replace = (sample.n_sub != sample.p_sub); // true: subject is replaced, false: object is replaced.

        value_type positive_transe_score = cal_transe_score(sample.p_sub, sample.p_obj, sample.relation_id,
                                                            parameter->transe_D, parameter->L1_flag, transeA, transeR);

        value_type negative_transe_score = cal_transe_score(sample.n_sub, sample.n_obj, sample.relation_id,
                                                            parameter->transe_D, parameter->L1_flag, transeA, transeR);

        // these two are already the results from sigmoid function
        value_type positive_hole_score = cal_hole_score(sample.p_sub, sample.p_obj, sample.relation_id,
                                                        parameter->hole_D, HoleE,
                                                        HoleP, descriptor);

        value_type negative_hole_score = cal_hole_score(sample.n_sub, sample.n_obj, sample.relation_id,
                                                        parameter->hole_D, HoleE,
                                                        HoleP, descriptor);

        value_type margin = weights[sample.relation_id].w1 * (sigmoid(positive_transe_score) - sigmoid(negative_transe_score)) + positive_hole_score - negative_hole_score - parameter->hole_margin;

        // compute the gradient then update the embeddings and weights
        if (margin < 0) {

            violations++;

            value_type *e_ps = HoleE.data().begin() + sample.p_sub * parameter->hole_D;
            value_type *e_po = HoleE.data().begin() + sample.p_obj * parameter->hole_D;
            value_type *r_k = HoleP.data().begin() + sample.relation_id * parameter->hole_D;

            Vec grad_r_p;
            correlation(descriptor, e_ps, e_po, grad_r_p, parameter->hole_D);
            //calculate gradient of sub
            Vec grad_s_p;
            correlation(descriptor, r_k, e_po, grad_s_p, parameter->hole_D);
            //calculate gradient of obj
            Vec grad_o_p;
            convolution(descriptor, r_k, e_ps, grad_o_p, parameter->hole_D);

            value_type weight_p = g_sigmoid(positive_hole_score);
            value_type weight_n = g_sigmoid(negative_hole_score);

            weights[sample.relation_id].w1 += parameter->step_size * (weight_p - weight_n);

            if (subject_replace) {

                value_type *e_ns = HoleE.data().begin() + sample.n_sub * parameter->hole_D;

                //calculate gradient of r
                Vec grad_r_n;
                correlation(descriptor, e_ns, e_po, grad_r_n, parameter->hole_D);
                //calculate gradient of sub
                Vec grad_s_n;
                correlation(descriptor, r_k, e_po, grad_s_n, parameter->hole_D);
                //calculate gradient of obj
                Vec grad_o_n;
                convolution(descriptor, r_k, e_ns, grad_o_n, parameter->hole_D);

                Vec grad_r_k = weight_p * grad_r_p - weight_n * grad_r_n - parameter->lambdaP * row(HoleP, sample.relation_id);
                Vec grad_e_ps = weight_p * grad_s_p - parameter->lambdaE * row(HoleE, sample.p_sub);
                Vec grad_e_po = weight_p * grad_o_p - weight_n * grad_o_n - parameter->lambdaE * row(HoleE, sample.p_obj);
                Vec grad_e_ns = - weight_n * grad_s_n - parameter->lambdaE * row(HoleE, sample.n_sub);

                (this->*update_grad)(r_k, grad_r_k.data().begin(), HoleP_G.data().begin() + sample.relation_id * parameter->hole_D, parameter->hole_D, 1);
                (this->*update_grad)(e_ps, grad_e_ps.data().begin(), HoleE_G.data().begin() + sample.p_sub * parameter->hole_D, parameter->hole_D, 1);
                (this->*update_grad)(e_po, grad_e_po.data().begin(), HoleE_G.data().begin() + sample.p_obj * parameter->hole_D, parameter->hole_D, 1);
                (this->*update_grad)(e_ns, grad_e_ns.data().begin(), HoleE_G.data().begin() + sample.n_sub * parameter->hole_D, parameter->hole_D, 1);

            } else {

                value_type *e_no = HoleE.data().begin() + sample.n_obj * parameter->hole_D;

                //calculate gradient of r
                Vec grad_r_n;
                correlation(descriptor, e_ps, e_no, grad_r_n, parameter->hole_D);
                //calculate gradient of sub
                Vec grad_s_n;
                correlation(descriptor, r_k, e_no, grad_s_n, parameter->hole_D);
                //calculate gradient of obj
                Vec grad_o_n;
                convolution(descriptor, r_k, e_ps, grad_o_n, parameter->hole_D);

                Vec grad_r_k = weight_p * grad_r_p - weight_n * grad_r_n - parameter->lambdaP * row(HoleP, sample.relation_id);
                Vec grad_e_ps = weight_p * grad_s_p - weight_n * grad_s_n - parameter->lambdaE * row(HoleE, sample.p_sub);
                Vec grad_e_po = weight_p * grad_o_p - parameter->lambdaE * row(HoleE, sample.p_obj);
                Vec grad_e_no = - weight_n * grad_o_n - parameter->lambdaE * row(HoleE, sample.n_obj);

                (this->*update_grad)(r_k, grad_r_k.data().begin(), HoleP_G.data().begin() + sample.relation_id * parameter->hole_D, parameter->hole_D, 1);
                (this->*update_grad)(e_ps, grad_e_ps.data().begin(), HoleE_G.data().begin() + sample.p_sub * parameter->hole_D, parameter->hole_D, 1);
                (this->*update_grad)(e_po, grad_e_po.data().begin(), HoleE_G.data().begin() + sample.p_obj * parameter->hole_D, parameter->hole_D, 1);
                (this->*update_grad)(e_no, grad_e_no.data().begin(), HoleE_G.data().begin() + sample.n_obj * parameter->hole_D, parameter->hole_D, 1);
            }
        }
    }

    string get_log_header() {
        return "Method,Optimization,epoch,TransE dimension,HOLE dimension,step size,TransE margin,HOLE margin,L1 flag,lambdaE,lambdaP,Rho,hit_rate_subject@" +
               to_string(parameter->hit_rate_topk) + ",hit_rate_object@" +
               to_string(parameter->hit_rate_topk) +
               ",subject_ranking,object_ranking,hit_rate_subject_filter@" +
               to_string(parameter->hit_rate_topk) + ",hit_rate_object_filter@" +
               to_string(parameter->hit_rate_topk) +
               ",subject_ranking_filter,object_ranking_filter,MAP_subject@" +
               to_string(parameter->num_of_replaced_entities) + ",MAP_object@" + to_string(parameter->num_of_replaced_entities);
    }

    string eval(const int epoch) {

        // transform HOLE to RESCAL for faster evaluation
        parameter->rescal_D = parameter->hole_D;
        vector<DenseMatrix> rescal_R;
        transform_hole2rescal(HOLE::HoleP, rescal_R);

        Monitor timer;
        timer.start();

        hit_rate testing_measure = eval_hit_rate(Method::m_RTLREnsemble, parameter, data, &(HOLE::HoleE), &rescal_R,
                                         &(TransE::transeA), &(TransE::transeR), nullptr, nullptr, nullptr, &weights,
                                         parameter->output_path + "/htpipeline_ensemble");

        string prefix = "testing data >>> ";

        print_hit_rate(prefix, parameter->hit_rate_topk, testing_measure);

        pair<value_type, value_type> map;
        map.first = -1;
        map.second = -1;
        if(parameter->eval_map) {
            map = eval_MAP(m_RTLREnsemble, parameter, data, &(HOLE::HoleE),
                                                        &rescal_R,
                                                        &(TransE::transeA), &(TransE::transeR), nullptr, nullptr,
                                                        nullptr,
                                                        &weights);

            string prefix = "testing data MAP evalution >>> ";
            print_map(prefix, parameter->num_of_replaced_entities, map);
        }

        timer.stop();

        cout << "evaluation time: " << timer.getElapsedTime() << " secs" << endl;

        string log = "";
        log.append("THPipeline,");
        log.append(parameter->optimization + ",");
        log.append(to_string(epoch) + ",");
        log.append(to_string(parameter->transe_D) + ",");
        log.append(to_string(parameter->hole_D) + ",");
        log.append(to_string(parameter->step_size) + ",");
        log.append(to_string(parameter->margin) + ",");
        log.append(to_string(parameter->hole_margin) + ",");
        log.append(to_string(parameter->L1_flag) + ",");
        log.append(to_string(parameter->lambdaE) + ",");
        log.append(to_string(parameter->lambdaP) + ",");
        log.append(to_string(parameter->Rho) + ",");

        string count_s = (testing_measure.count_s == -1? "Not Computed" : to_string(testing_measure.count_s));
        log.append(count_s + ",");

        string count_o = (testing_measure.count_o == -1? "Not Computed" : to_string(testing_measure.count_o));
        log.append(count_o + ",");

        string count_s_ranking = (testing_measure.count_s_ranking == -1? "Not Computed" : to_string(testing_measure.count_s_ranking));
        log.append(count_s_ranking + ",");

        string count_o_ranking = (testing_measure.count_o_ranking == -1? "Not Computed" : to_string(testing_measure.count_o_ranking));
        log.append(count_o_ranking + ",");

        log.append(to_string(testing_measure.count_s_filtering) + ",");
        log.append(to_string(testing_measure.count_o_filtering) + ",");
        log.append(to_string(testing_measure.count_s_ranking_filtering) + ",");
        log.append(to_string(testing_measure.count_o_ranking_filtering) + ",");

        string map1 = (map.first == -1 ? "Not Computed" : to_string(map.first));
        string map2 = (map.second == -1 ? "Not Computed" : to_string(map.second));

        log.append(map1 + ",");
        log.append(map2);

        return log;
    }

    void output(const int epoch) {
        if(parameter->optimization=="sgd"){

            output_matrices(TransE::transeA, TransE::transeR, data->N, parameter->transe_D, data->K, epoch,
                            parameter->output_path + "/TransE");

            output_matrices(HOLE::HoleE, HOLE::HoleP, data->N, parameter->hole_D, data->K, epoch,
                            parameter->output_path + "/HOLE");

        } else {

            output_matrices(TransE::transeA, TransE::transeR, TransE::transeA_G, TransE::transeR_G, data->N, parameter->transe_D, data->K, epoch,
                            parameter->output_path + "/HOLE");

            output_matrices(HOLE::HoleE, HOLE::HoleP, HOLE::HoleE_G, HOLE::HoleP_G, data->N, parameter->hole_D, data->K, epoch,
                            parameter->output_path + "/HOLE");
        }
    }

public:

    THPipeline(Parameter *parameter, Data *data) : Optimizer(parameter, data), TransE(parameter, data),
                                                   HOLE(parameter, data) {}

    void train() {

        initialize();

        Sample sample;

        Monitor timer;

        std::vector<int> indices(data->num_of_training_triples);
        std::iota(std::begin(indices), std::end(indices), 0);

        if (parameter->print_log_header) {
            print_log(get_log_header());
        }

        int workload = data->num_of_training_triples / parameter->num_of_thread;

        // first train TransE independently
        if (!(parameter->restore_from_transe && parameter->restore_epoch > 0 && parameter->transe_restore_path != "")) {

            for (int epoch = 1; epoch <= parameter->epoch; epoch++) {

                violations = 0;

                timer.start();

                std::random_shuffle(indices.begin(), indices.end());

                std::function<void(int)> compute_func = [&](int thread_index) -> void {
                    int start = thread_index * workload;
                    int end = std::min(start + workload, data->num_of_training_triples);
                    for (int n = start; n < end; n++) {
                        if (parameter->num_of_thread == 1) {
                            Sampler::random_sample(*data, sample, indices[n]);
                        } else {
                            Sampler::random_sample_multithreaded(*data, sample, indices[n]);
                        }

                        TransE::update(sample);
                    }
                };

                ThreadUtil::execute_threads(compute_func, parameter->num_of_thread);

                timer.stop();

                cout << "epoch " << epoch << ", time " << timer.getElapsedTime() << " secs" << endl;

                cout << "[1] violations: " << violations << endl;

            }

            timer.start();

            string log = TransE::eval(parameter->epoch);

            timer.stop();

            cout << "evaluation time: " << timer.getElapsedTime() << " secs" << endl;

            print_log(log);
        }

        // then train HOLE and learn weight
        for (int epoch = 1; epoch <= parameter->epoch; epoch++) {

            violations = 0;

            timer.start();

            std::random_shuffle(indices.begin(), indices.end());

            std::function<void(int)> compute_func = [&](int thread_index) -> void {
                int start = thread_index * workload;
                int end = std::min(start + workload, data->num_of_training_triples);
                for (int n = start; n < end; n++) {
                    if(parameter->num_of_thread == 1){
                        Sampler::random_sample(*data, sample, indices[n]);
                    } else {
                        Sampler::random_sample_multithreaded(*data, sample, indices[n]);
                    }

                    update(sample);
                }
            };

            ThreadUtil::execute_threads(compute_func, parameter->num_of_thread);

            timer.stop();

            cout << "epoch " << epoch << ", time " << timer.getElapsedTime() << " secs" << endl;

            cout << "[2] violations: " << violations << endl;
        }

        eval(parameter->epoch);



    }
};
#endif //THPIPELINE_H
