#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <functional>
#include <string>
#include <cmath>
#include<memory>
#include <Eigen/Dense>

#include "Layers.h"

namespace nn {

// =====================================================================
// CLEAN READABILITY ALIASES (Hides the ugly 4D vectors!)
// =====================================================================
using SavedMatrix = std::vector<std::vector<double>>;
using SavedVector = std::vector<double>;
using SavedStateW = std::vector<std::vector<SavedMatrix>>; // Replaces the 4D Vector
using SavedStateB = std::vector<std::vector<SavedVector>>; // Replaces the 3D Vector

class Optimizer {
private:
    using UpdateRule = std::function<void(
        Eigen::MatrixXd&, Eigen::MatrixXd&, Eigen::VectorXd&, Eigen::VectorXd&,
        std::vector<std::vector<Eigen::MatrixXd>>&, std::vector<std::vector<Eigen::VectorXd>>&,
        int, double, int&
    )>;

    UpdateRule rule;
    int t = 0;
    
    // Internal Eigen memory arrays
    std::vector<std::vector<Eigen::MatrixXd>> state_W;
    std::vector<std::vector<Eigen::VectorXd>> state_b;
    
    bool initialized = false;
    std::vector<double> params_;

public:
    std::string name;
    double lr;
    
    Optimizer(UpdateRule r, double lr, int num_states, std::string name, std::vector<double> params = {})
        : rule(std::move(r)), lr(lr), params_(std::move(params)), name(std::move(name)) {
        state_W.resize(num_states);
        state_b.resize(num_states);
    }

    const std::vector<double>& params() const { return params_; }

void init(const std::vector<std::unique_ptr<BaseLayer>>& layers) {
        for (int s = 0; s < state_W.size(); s++) {
            state_W[s].clear();
            state_b[s].clear();
            for (const auto& l : layers) {
                if (l->has_parameters()) {
                    state_W[s].push_back(Eigen::MatrixXd::Zero(l->get_output_dim(), l->get_input_dim()));
                    state_b[s].push_back(Eigen::VectorXd::Zero(l->get_output_dim()));
                } else {
                    // Push empty matrices for parameterless layers
                    state_W[s].push_back(Eigen::MatrixXd(0, 0));
                    state_b[s].push_back(Eigen::VectorXd(0));
                }
            }
        }
        initialized = true;
    }
void step(std::vector<std::unique_ptr<BaseLayer>>& layers) {
        if (!initialized) init(layers);
        t++; 
        for (int i = 0; i < layers.size(); i++) {
            // Skip layers without weights
            if (!layers[i]->has_parameters()|| !layers[i]->is_trainable()) continue; 
            
            rule(layers[i]->get_weights(), layers[i]->get_dw(), 
                 layers[i]->get_bias(), layers[i]->get_db(), 
                 state_W, state_b, i, lr, t);
        }
    }

    // =====================================================================
    // STATE TRANSLATORS (Now highly readable using aliases)
    // =====================================================================
    
    SavedStateW get_state_W() const {
        SavedStateW res(state_W.size());
        for (size_t s = 0; s < state_W.size(); ++s) {
            res[s].resize(state_W[s].size());
            for (size_t l = 0; l < state_W[s].size(); ++l) {
                res[s][l].assign(state_W[s][l].rows(), std::vector<double>(state_W[s][l].cols(), 0.0));
                for (int r = 0; r < state_W[s][l].rows(); ++r) {
                    for (int c = 0; c < state_W[s][l].cols(); ++c) {
                        res[s][l][r][c] = state_W[s][l](r, c);
                    }
                }
            }
        }
        return res;
    }

    SavedStateB get_state_b() const {
        SavedStateB res(state_b.size());
        for (size_t s = 0; s < state_b.size(); ++s) {
            res[s].resize(state_b[s].size());
            for (size_t l = 0; l < state_b[s].size(); ++l) {
                res[s][l].assign(state_b[s][l].size(), 0.0);
                for (int i = 0; i < state_b[s][l].size(); ++i) {
                    res[s][l][i] = state_b[s][l](i);
                }
            }
        }
        return res;
    }

    void set_state_W(const SavedStateW& new_state_W) {
        state_W.resize(new_state_W.size());
        for (size_t s = 0; s < new_state_W.size(); ++s) {
            state_W[s].clear();
            for (size_t l = 0; l < new_state_W[s].size(); ++l) {
                int rows = new_state_W[s][l].size();
                int cols = rows > 0 ? new_state_W[s][l][0].size() : 0;
                Eigen::MatrixXd mat(rows, cols);
                for (int r = 0; r < rows; ++r) {
                    for (int c = 0; c < cols; ++c) mat(r, c) = new_state_W[s][l][r][c];
                }
                state_W[s].push_back(mat);
            }
        }
    }

    void set_state_b(const SavedStateB& new_state_b) {
        state_b.resize(new_state_b.size());
        for (size_t s = 0; s < new_state_b.size(); ++s) {
            state_b[s].clear();
            for (size_t l = 0; l < new_state_b[s].size(); ++l) {
                int size = new_state_b[s][l].size();
                Eigen::VectorXd vec(size);
                for (int i = 0; i < size; ++i) vec(i) = new_state_b[s][l][i];
                state_b[s].push_back(vec);
            }
        }
    }

    // Standard getters/setters expected by model.h
    int get_timestep() const { return t; }
    void set_timestep(int new_t) { t = new_t; }
    void set_initialized(bool init) { initialized = init; }
    bool is_initialized() const { return initialized; }
};

// =====================================================================
// ALGORITHMS (Fully Functional Eigen implementations)
// =====================================================================

inline Optimizer SGD(double lr = 0.01) {
    return Optimizer(
        [](auto& W, auto& dW, auto& b, auto& db, auto&, auto&, int, double lr, int&) {
            W -= lr * dW; 
            b -= lr * db;
        },
        lr, 0, "SGD", {lr}
    );
}

inline Optimizer Momentum(double lr = 0.01, double beta = 0.9) {
    return Optimizer(
        [beta](auto& W, auto& dW, auto& b, auto& db, auto& state_W, auto& state_b, int idx, double lr, int&) {
            auto& vW = state_W[0][idx]; 
            auto& vb = state_b[0][idx];
            
            vW = beta * vW + (1 - beta) * dW;
            W -= lr * vW; 
            
            vb = beta * vb + (1 - beta) * db;
            b -= lr * vb;
        },
        lr, 1, "Momentum", {lr, beta}
    );
}

inline Optimizer RMSProp(double lr = 0.001, double beta = 0.9, double eps = 1e-8) {
    return Optimizer(
        [beta, eps](auto& W, auto& dW, auto& b, auto& db, auto& state_W, auto& state_b, int idx, double lr, int&) {
            auto& sW = state_W[0][idx];
            auto& sb = state_b[0][idx];
            
            sW.array() = beta * sW.array() + (1 - beta) * dW.array().square();
            W.array() -= lr * dW.array() / (sW.array().sqrt() + eps);
            
            sb.array() = beta * sb.array() + (1 - beta) * db.array().square();
            b.array() -= lr * db.array() / (sb.array().sqrt() + eps);
        },
        lr, 1, "RMSProp", {lr, beta, eps}
    );
}

inline Optimizer Adam(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8) {
    return Optimizer(
        [b1, b2, eps](auto& W, auto& dW, auto& b, auto& db, auto& state_W, auto& state_b, int idx, double lr, int& t) {
            auto& mW = state_W[0][idx];
            auto& vW = state_W[1][idx];
            auto& mb = state_b[0][idx];
            auto& vb = state_b[1][idx];

            mW = b1 * mW + (1 - b1) * dW;
            vW.array() = b2 * vW.array() + (1 - b2) * dW.array().square();

            Eigen::MatrixXd m_hat_W = mW / (1 - std::pow(b1, t));
            Eigen::MatrixXd v_hat_W = vW / (1 - std::pow(b2, t));
            W.array() -= lr * m_hat_W.array() / (v_hat_W.array().sqrt() + eps);

            mb = b1 * mb + (1 - b1) * db;
            vb.array() = b2 * vb.array() + (1 - b2) * db.array().square();

            Eigen::VectorXd m_hat_b = mb / (1 - std::pow(b1, t));
            Eigen::VectorXd v_hat_b = vb / (1 - std::pow(b2, t));
            b.array() -= lr * m_hat_b.array() / (v_hat_b.array().sqrt() + eps);
        },
        lr, 2, "Adam", {lr, b1, b2, eps}
    );
}

} // namespace nn
#endif