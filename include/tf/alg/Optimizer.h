#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tf/util/ThreadUtil.h"
#include "tf/alg/Sampler.h"
#include "tf/util/Base.h"
#include "tf/util/Parameter.h"
#include "tf/util/Data.h"
#include "tf/util/Monitor.h"

class Optimizer {

private:

    void init_update_fun() {
        if(parameter->optimization=="sgd"){
            update_grad = &Optimizer::sgd;
        } else if (parameter->optimization=="adagrad"){
            update_grad = &Optimizer::adagrad;
        } else if (parameter->optimization=="adadelta") {
            update_grad = &Optimizer::adadelta;
        } else {
            cerr << "recognized method: " << parameter->optimization << endl;
            exit(1);
        }
    }

    /**
     * Stochastic Gradient Descent
     * @param factors
     * @param gradient
     * @param pre_gradient_square: Square of previous gradient (not used)
     * @param d: length of factors/gradient/pre_gradient_square
     * @param weight: scaling value
     */
    void sgd(value_type *factors, value_type *gradient, value_type *pre_gradient_square,
             const int d, const value_type weight = 1.0) {

        value_type scale = parameter->step_size * weight;

#ifdef use_mkl
        cblas_xaxpy(d, scale, gradient, 1, factors, 1);
#else
        for (int i = 0; i < d; i++) {
            factors[i] += scale * gradient[i];
        }
#endif
    }

    /**
     * AdaGrad: Adaptive Gradient Algorithm
     * @param factors
     * @param gradient
     * @param pre_gradient_square: Square of previous gradient
     * @param d: length of factors/gradient/pre_gradient_square
     * @param weight: scaling value
     */
    void adagrad(value_type *factors, value_type *gradient, value_type *pre_gradient_square,
                 const int d, const value_type weight = 1.0) {

        value_type scale = parameter->step_size * weight;

#ifdef use_mkl
        Vec gg(d);
        vxMul(d, gradient, gradient, gg.data().begin());
        cblas_xaxpy(d, 1.0, gg.data().begin(), 1, pre_gradient_square, 1);
        vxAdd(d, pre_gradient_square, min_not_zeros.data().begin(), gg.data().begin());

        Vec sqrt(d);
        vxSqrt(d, gg.data().begin(), sqrt.data().begin());
        vxDiv(d, gradient, sqrt.data().begin(), gg.data().begin());
        cblas_xaxpy(d, scale, gg.data().begin(), 1, factors, 1);
#else
        for (int i = 0; i < d; i++) {
            pre_gradient_square[i] += gradient[i] * gradient[i];
            factors[i] += scale * gradient[i] / sqrt(pre_gradient_square[i] + min_not_zero_value);
        }
#endif
    }

    /**
     * AdaDelta: An Adaptive Learning Rate Method
     * @param factors
     * @param gradient
     * @param pre_gradient_square: Square of previous gradient
     * @param d: length of factors/gradient/pre_gradient_square
     * @param weight: scaling value
     */
    void adadelta(value_type *factors, value_type *gradient, value_type *pre_gradient_square,
                  const int d, const value_type weight = 1.0) {

        value_type scale = parameter->step_size * weight;

#ifdef use_mkl
        Vec gg(d);
        vxMul(d, gradient, gradient, gg.data().begin());
        cblas_xscal(d, 1 - parameter->Rho, gg.data().begin(), 1);
        cblas_xaxpy(d, parameter->Rho, pre_gradient_square, 1, gg.data().begin(), 1);
        cblas_xcopy(d, gg.data().begin(), 1, pre_gradient_square, 1);
        vxAdd(d, pre_gradient_square, min_not_zeros.data().begin(), gg.data().begin());

        Vec sqrt(d);
        vxSqrt(d, gg.data().begin(), sqrt.data().begin());
        vxDiv(d, gradient, sqrt.data().begin(), gg.data().begin());
        cblas_xaxpy(d, scale, gg.data().begin(), 1, factors, 1);

#else
        for (int i = 0; i < d; i++) {
            pre_gradient_square[i] = parameter->Rho * pre_gradient_square[i] + (1 - parameter->Rho) * gradient[i] * gradient[i];
            factors[i] += scale * gradient[i] / sqrt(pre_gradient_square[i] + min_not_zero_value);
        }
#endif
    }

protected:

    // Dynamic pointer to update function
    void (Optimizer::*update_grad)(value_type *factors, value_type *gradient, value_type *pre_gradient_square,
                                   const int d, const value_type weight) = nullptr;

    value_type min_not_zero_value = 1e-7;
    Vec min_not_zeros; // initialize its size and values in inherited class

    Parameter *parameter = nullptr;
    Data *data = nullptr;
    int current_epoch;
    int violations;
    value_type loss;

    /**
     * Update function
     * @param sample
     * @param weight
     */
    virtual void update(Sample &sample, const value_type weight = 1.0) = 0;

    /**
     * Initialization function
     * Initialize factors, pre_gradient_square, min_none_zeros...
     */
    virtual void initialize() = 0;

    /**
     * Evaluation function
     * @param epoch
     * @return
     */
    virtual string eval(const int epoch) = 0;

    /**
     * Output function after training
     * @param epoch
     */
    virtual void output(const int epoch) = 0;

    /**
     * Header for CSV file
     * @return
     */
    virtual string get_log_header() = 0;

    /**
     * Compute loss value
     * @return
     */
    virtual value_type cal_loss() {
        return 0;
    }

    void print_log(const string log_content) {
        ofstream log(parameter->log_path.c_str(), std::ofstream::out | std::ofstream::app);
        log << log_content << endl;
        log.close();
    }

public:

    Optimizer() {
        init_update_fun();
    }

    Optimizer(Parameter *parameter, Data *data) : parameter(parameter), data(data) {
        init_update_fun();;
    }

    /**
     * Start to train
     */
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

        value_type total_time = 0.0;

        for (int epoch = current_epoch; epoch <= parameter->epoch; epoch++) {

            violations = 0;
            loss = 0;

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

            total_time += timer.getElapsedTime();

            cout << "epoch " << epoch << ", time " << timer.getElapsedTime() << " secs" << endl;

            cout << "violations: " << violations << endl;

            if(parameter->eval_train) {
                loss = cal_loss();

                if (loss != 0) {
                    cout << "loss: " << loss << endl;
                }
            }

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

        cout << "Total Training Time: " << total_time << " secs" << endl;
    }

};
#endif //OPTIMIZER_H
