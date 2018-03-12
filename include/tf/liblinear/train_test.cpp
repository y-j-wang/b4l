#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include "linear.h"
#include "linear.cpp"
#include "tron.h"
#include "tron.cpp"

void exit_with_help() {
    printf(
            "Usage: train [options] training_set_file [model_file]\n"
                    "options:\n"
                    "-s type : set type of solver (default 1)\n"
                    "  for multi-class classification\n"
                    "	 0 -- L2-regularized logistic regression (primal)\n"
                    "	 1 -- L2-regularized L2-loss support vector classification (dual)\n"
                    "	 2 -- L2-regularized L2-loss support vector classification (primal)\n"
                    "	 3 -- L2-regularized L1-loss support vector classification (dual)\n"
                    "	 4 -- support vector classification by Crammer and Singer\n"
                    "	 5 -- L1-regularized L2-loss support vector classification\n"
                    "	 6 -- L1-regularized logistic regression\n"
                    "	 7 -- L2-regularized logistic regression (dual)\n"
                    "  for regression\n"
                    "	11 -- L2-regularized L2-loss support vector regression (primal)\n"
                    "	12 -- L2-regularized L2-loss support vector regression (dual)\n"
                    "	13 -- L2-regularized L1-loss support vector regression (dual)\n"
                    "-c cost : set the parameter C (default 1)\n"
                    "-p epsilon : set the epsilon in loss function of SVR (default 0.1)\n"
                    "-e epsilon : set tolerance of termination criterion\n"
                    "	-s 0 and 2\n"
                    "		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n"
                    "		where f is the primal function and pos/neg are # of\n"
                    "		positive/negative data (default 0.01)\n"
                    "	-s 11\n"
                    "		|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)\n"
                    "	-s 1, 3, 4, and 7\n"
                    "		Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
                    "	-s 5 and 6\n"
                    "		|f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,\n"
                    "		where f is the primal function (default 0.01)\n"
                    "	-s 12 and 13\n"
                    "		|f'(alpha)|_1 <= eps |f'(alpha0)|,\n"
                    "		where f is the dual function (default 0.1)\n"
                    "-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
                    "-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
                    "-v n: n-fold cross validation mode\n"
                    "-C : find parameter C (only for -s 0 and 2)\n"
                    "-q : quiet mode (no outputs)\n"
    );
    exit(1);
}

int main(int argc, char **argv) {
    // `l' is the number of training data. If bias >= 0, we assume
    // that one additional feature is added to the end of each data
    // instance. `n' is the number of feature (including the bias feature
    // if bias >= 0).

    problem prob;

    prob.l = 5;
    prob.n = 4;
    prob.bias = 1;

    feature_node **x = new feature_node *[prob.l];

    for (int row = 0; row < prob.l; row++) {

        x[row] = new feature_node[prob.n + 1];

        for (int col = 0; col < prob.n - 1; col++) {
            x[row][col].index = col + 1;  // index should start from 1
            x[row][col].value = row * col;
        }
        x[row][prob.n - 1].index = prob.n;
        x[row][prob.n - 1].value = prob.bias;

        x[row][prob.n].index = -1;
    }

    prob.x = x;

    prob.y = new double[prob.l];
    prob.y[0] = -1;
    prob.y[1] = 1;
    prob.y[2] = 1;
    prob.y[3] = -1;
    prob.y[4] = -1;

    // default values
    parameter param;
    param.solver_type = L2R_LR;
    param.C = 1;
    param.eps = 1e-4;
    param.nr_thread = 4;
    param.p = 0.1;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
    param.init_sol = NULL;

#define verbose_lr

    model *model_ = train(&prob, &param);

    std::cout << "---------" << std::endl;
    std::cout << model_->nr_class << std::endl;
    std::cout << model_->nr_feature << std::endl;

    for (int i = 0; i < model_->nr_feature; i++) {
        std::cout << model_->w[i] << std::endl;
    }

    free_and_destroy_model(&model_);

    destroy_param(&param);

    for (int row = 0; row < prob.l; row++) {
        delete[] prob.x[row];
    }

    delete[] prob.x;
    delete[] prob.y;

    return 0;
}
