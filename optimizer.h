#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include<algorithm>
#include <functional>
#include <utility>
#include"layer_backprop.h"

namespace nn {
    // The UpdateRule lambda signature: 
    // (param, gradient, state_vectors, index)
class Optimizer {
private:
    using UpdateRule = std::function<
        void(
            std::vector<std::vector<double>>&,
            std::vector<std::vector<double>>&,
            std::vector<double>&,
            std::vector<double>&,
            std::vector<std::vector<std::vector<std::vector<double>>>>&, // W states
            std::vector<std::vector<std::vector<double>>>&,              // b states
            int,
            double,
            int&   // timestep
        )
    >;

    UpdateRule rule;
    int t = 0;
    
    // state buffers
    // state_W[state_id][layer][i][j]
    std::vector<std::vector<std::vector<std::vector<double>>>> state_W;
    
    // state_b[state_id][layer][i]
    std::vector<std::vector<std::vector<double>>> state_b;
    
    bool initialized = false;
    std::vector<double> params_;
    
public:
    std::string name;
    double lr;
    Optimizer(UpdateRule r,
              double lr,
              int num_states,
              std::string name,
              std::vector<double> params = {})
        : rule(std::move(r)),
          lr(lr),
          params_(std::move(params)),
          name(std::move(name)) {
        state_W.resize(num_states);
        state_b.resize(num_states);
    }

    const std::vector<double>& params() const {
        return params_;
    }

        void init(std::vector<layer>& layers) {
        for (int s = 0; s < state_W.size(); s++) {
            for (auto& l : layers) {
                state_W[s].push_back(l.get_weights());
                state_b[s].push_back(l.get_bias());

                // zero init
                for (auto& row : state_W[s].back())
                    for (auto& x : row) x = 0.0;

                std::fill(state_b[s].back().begin(), state_b[s].back().end(), 0.0);
            }
        }
        initialized = true;
    }

    void update(nn::layer& l, int idx) {
        if (!initialized) return;

        auto& W = const_cast<std::vector<std::vector<double>>&>(l.get_weights());
        auto& b = const_cast<std::vector<double>&>(l.get_bias());
        auto& dW = l.get_dw();
        auto& db = l.get_db();

        rule(W, dW, b, db, state_W, state_b, idx, lr,t);
    }
    void step(std::vector<layer>& layers) {
        if (!initialized) init(layers);

        t++;

        for (int i = 0; i < layers.size(); i++) {
            auto& W = layers[i].get_weights();
            auto& dW = layers[i].get_dw();
            auto& b = layers[i].get_bias();
            auto& db = layers[i].get_db();

            rule(W, dW, b, db, state_W, state_b, i, lr, t);
        }
    }

    // Methods to access and restore optimizer state
    const std::vector<std::vector<std::vector<std::vector<double>>>>& get_state_W() const {
        return state_W;
    }

    const std::vector<std::vector<std::vector<double>>>& get_state_b() const {
        return state_b;
    }

    int get_timestep() const {
        return t;
    }

    void set_state_W(const std::vector<std::vector<std::vector<std::vector<double>>>>& new_state_W) {
        state_W = new_state_W;
    }

    void set_state_b(const std::vector<std::vector<std::vector<double>>>& new_state_b) {
        state_b = new_state_b;
    }

    void set_timestep(int new_t) {
        t = new_t;
    }

    void set_initialized(bool init) {
        initialized = init;
    }

    bool is_initialized() const {
        return initialized;
    }
};

inline Optimizer SGD(double lr = 0.01) {
    return Optimizer(
        [](auto& W, auto& dW, auto& b, auto& db,
           auto&, auto&, int, double lr, int&) {

            for (int i = 0; i < W.size(); i++)
                for (int j = 0; j < W[i].size(); j++)
                    W[i][j] -= lr * dW[i][j];

            for (int i = 0; i < b.size(); i++)
                b[i] -= lr * db[i];
        },
        lr, 0, "SGD", {lr}
    );
}

inline Optimizer Momentum(double lr = 0.01, double beta = 0.9) {
    return Optimizer(
        [beta](auto& W, auto& dW, auto& b, auto& db,
               auto& state_W, auto& state_b, int idx, double lr, int&) {

            auto& vW = state_W[0][idx];
            auto& vb = state_b[0][idx];

            for (int i = 0; i < W.size(); i++) {
                for (int j = 0; j < W[i].size(); j++) {
                    vW[i][j] = beta * vW[i][j] + (1 - beta) * dW[i][j];
                    W[i][j] -= lr * vW[i][j];
                }
            }

            for (int i = 0; i < b.size(); i++) {
                vb[i] = beta * vb[i] + (1 - beta) * db[i];
                b[i] -= lr * vb[i];
            }
        },
        lr, 1, "Momentum", {lr, beta}
    );
}

inline Optimizer RMSProp(double lr = 0.001, double beta = 0.9, double eps = 1e-8) {
    return Optimizer(
        [beta, eps](auto& W, auto& dW, auto& b, auto& db,
                    auto& state_W, auto& state_b, int idx, double lr, int&) {

            auto& sW = state_W[0][idx];
            auto& sb = state_b[0][idx];

            for (int i = 0; i < W.size(); i++) {
                for (int j = 0; j < W[i].size(); j++) {
                    sW[i][j] = beta * sW[i][j] + (1 - beta) * dW[i][j] * dW[i][j];
                    W[i][j] -= lr * dW[i][j] / (sqrt(sW[i][j]) + eps);
                }
            }

            for (int i = 0; i < b.size(); i++) {
                sb[i] = beta * sb[i] + (1 - beta) * db[i] * db[i];
                b[i] -= lr * db[i] / (sqrt(sb[i]) + eps);
            }
        },
        lr, 1, "RMSProp", {lr, beta, eps}
    );
}

inline Optimizer Adam(double lr = 0.001,
                      double b1 = 0.9,
                      double b2 = 0.999,
                      double eps = 1e-8) {
    return Optimizer(
        [b1, b2, eps](auto& W, auto& dW, auto& b, auto& db,
                      auto& state_W, auto& state_b, int idx, double lr, int& t) {

            auto& mW = state_W[0][idx];
            auto& vW = state_W[1][idx];

            auto& mb = state_b[0][idx];
            auto& vb = state_b[1][idx];

            for (int i = 0; i < W.size(); i++) {
                for (int j = 0; j < W[i].size(); j++) {
                    mW[i][j] = b1 * mW[i][j] + (1 - b1) * dW[i][j];
                    vW[i][j] = b2 * vW[i][j] + (1 - b2) * dW[i][j] * dW[i][j];

                    double m_hat = mW[i][j] / (1 - pow(b1, t));
                    double v_hat = vW[i][j] / (1 - pow(b2, t));

                    W[i][j] -= lr * m_hat / (sqrt(v_hat) + eps);
                }
            }

            for (int i = 0; i < b.size(); i++) {
                mb[i] = b1 * mb[i] + (1 - b1) * db[i];
                vb[i] = b2 * vb[i] + (1 - b2) * db[i] * db[i];

                double m_hat = mb[i] / (1 - pow(b1, t));
                double v_hat = vb[i] / (1 - pow(b2, t));

                b[i] -= lr * m_hat / (sqrt(v_hat) + eps);
            }
        },
        lr, 2, "Adam", {lr, b1, b2, eps}
    );
}


}
#endif