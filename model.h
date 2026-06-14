#ifndef MODEL_H
#define MODEL_H

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include "layer_backprop.h"
#include "loss_func.h"
#include "optimizer.h"
#include "normalization.h" // Needed to recognize Normalization classes

namespace nn {

// 1. Data Structure to hold all model parameters safely in memory
struct ModelParams {
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
    std::vector<std::string> activations;
    
    std::string loss;
    std::string optimizer;
    std::vector<double> optimizer_params;
    int input_dim;
    int epochs;
    int batch_size;
    int optimizer_timestep;
    std::vector<std::vector<std::vector<std::vector<double>>>> optimizer_state_W;
    std::vector<std::vector<std::vector<double>>> optimizer_state_b;
    bool optimizer_initialized;
};

class SequentialNetwork {
private:
    std::vector<layer> layers;
    LossFunction loss;
    Optimizer optimizer;
    int epochs;
    int batch_size;
    int input_dim; 

    static std::string canonical_key(const std::string& raw) {
        std::string key;
        for (char c : raw) {
            if (std::isalnum(static_cast<unsigned char>(c))) {
                key.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
        }
        return key;
    }

    ActivationFunction get_activation(const std::string& name, const std::vector<double>& p = {}) const {
        const std::string key = canonical_key(name);
        if (key == "identity") return activations::Identity();
        if (key == "binarystep") return activations::BinaryStep();
        if (key == "sigmoid") return activations::Sigmoid();
        if (key == "tanh") return activations::Tanh();
        if (key == "relu") return activations::ReLU();
        if (key == "relu6") return activations::ReLU6();
        if (key == "leakyrelu") return activations::LeakyReLU(p.empty() ? 0.01 : p[0]);
        if (key == "prelu") return activations::PReLU(p.empty() ? 0.25 : p[0]);
        if (key == "elu") return activations::ELU(p.empty() ? 1.0 : p[0]);
        if (key == "selu") return activations::SELU((p.size() > 0) ? p[0] : 1.0507009873554805, (p.size() > 1) ? p[1] : 1.6732632423543772);
        if (key == "softplus") return activations::Softplus();
        if (key == "softsign") return activations::Softsign();
        if (key == "swish") return activations::Swish(p.empty() ? 1.0 : p[0]);
        if (key == "silu") return activations::SiLU();
        if (key == "gelu") return activations::GELU();
        if (key == "mish") return activations::Mish();
        if (key == "hardsigmoid") return activations::HardSigmoid();
        if (key == "hardswish") return activations::HardSwish();
        return activations::Identity();
    }

    LossFunction get_loss(const std::string& name, const std::vector<double>& p = {}) const {
        const std::string key = canonical_key(name);
        if (key == "mse") return losses::MSE();
        if (key == "mae") return losses::MAE();
        if (key == "binarycrossentropy" || key == "bce") return losses::BinaryCrossEntropy();
        if (key == "hinge") return losses::Hinge();
        if (key == "squaredhinge") return losses::SquaredHinge();
        if (key == "huber") return losses::Huber(p.empty() ? 1.0 : p[0]);
        if (key == "logcosh") return losses::LogCosh();
        if (key == "msle") return losses::MSLE();
        if (key == "poisson") return losses::Poisson();
        if (key == "kldivergence" || key == "kl") return losses::KLDivergence();
        return losses::MSE();
    }

    Optimizer get_optimizer(const std::string& name, const std::vector<double>& p = {}) const {
        const std::string key = canonical_key(name);
        auto param = [&p](size_t idx, double f) { return (p.size() > idx) ? p[idx] : f; };
        if (key == "sgd") return nn::SGD(param(0, 0.01));
        if (key == "momentum") return nn::Momentum(param(0, 0.01), param(1, 0.9));
        if (key == "rmsprop") return nn::RMSProp(param(0, 0.001), param(1, 0.9), param(2, 1e-8));
        if (key == "adam") return nn::Adam(param(0, 0.001), param(1, 0.9), param(2, 0.999), param(3, 1e-8));
        return nn::SGD(0.01);
    }

public:
    SequentialNetwork(LossFunction loss_fn = losses::MSE(), Optimizer opt = nn::SGD(), int epochs = 100)
        : loss(std::move(loss_fn)), optimizer(std::move(opt)), epochs(epochs), batch_size(1), input_dim(-1) {}

    void add_layer(layer l) { layers.push_back(std::move(l)); }
    void add_layer(int in_dim, int out_dim, ActivationFunction act) { layers.push_back(layer(in_dim, out_dim, std::move(act))); }

    std::vector<double> forward(const std::vector<double>& input) {
        std::vector<double> out = input;
        for (auto& l : layers) out = l.forward(out);
        return out;
    }

    void backward(const std::vector<double>& y_pred, const std::vector<double>& y_true) {
        std::vector<double> grad = loss.gradvec(y_pred, y_true);
        for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
            grad = layers[i].backward(grad);
        }
    }

    void fit(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, int custom_batch_size = -1) {
        int eff_batch = (custom_batch_size > 0) ? custom_batch_size : batch_size;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_loss = 0.0;
            for (size_t b_start = 0; b_start < X.size(); b_start += eff_batch) {
                size_t b_end = std::min(X.size(), b_start + eff_batch);
                size_t c_batch_size = b_end - b_start;

                std::vector<std::vector<std::vector<double>>> batch_dW(layers.size());
                std::vector<std::vector<double>> batch_db(layers.size());

                for (size_t l = 0; l < layers.size(); ++l) {
                    batch_dW[l].assign(layers[l].get_weights().size(), std::vector<double>(layers[l].get_weights()[0].size(), 0.0));
                    batch_db[l].assign(layers[l].get_bias().size(), 0.0);
                }

                for (size_t i = b_start; i < b_end; ++i) {
                    std::vector<double> y_pred = forward(X[i]);
                    total_loss += loss.lossvec(y_pred, Y[i]);
                    backward(y_pred, Y[i]);

                    for (size_t l = 0; l < layers.size(); ++l) {
                        for (size_t r = 0; r < layers[l].get_dw().size(); ++r)
                            for (size_t c = 0; c < layers[l].get_dw()[r].size(); ++c)
                                batch_dW[l][r][c] += layers[l].get_dw()[r][c];
                        for (size_t j = 0; j < layers[l].get_db().size(); ++j)
                            batch_db[l][j] += layers[l].get_db()[j];
                    }
                }

                double inv_batch = 1.0 / c_batch_size;
                for (size_t l = 0; l < layers.size(); ++l) {
                    for (size_t r = 0; r < layers[l].get_dw().size(); ++r)
                        for (size_t c = 0; c < layers[l].get_dw()[r].size(); ++c)
                            layers[l].get_dw()[r][c] = batch_dW[l][r][c] * inv_batch;
                    for (size_t j = 0; j < layers[l].get_db().size(); ++j)
                        layers[l].get_db()[j] = batch_db[l][j] * inv_batch;
                }
                optimizer.step(layers);
            }
            if (epoch % (epochs / 10 + 1) == 0) {
                std::cout << "Epoch " << epoch << " | Loss: " << total_loss / X.size() << std::endl;
            }
        }
    }

    std::vector<std::vector<double>> predict(const std::vector<std::vector<double>>& X) {
        std::vector<std::vector<double>> preds;
        for (const auto& x : X) preds.push_back(forward(x));
        return preds;
    }

    // Load Model using the ModelParams Struct
    void load_model(const ModelParams& p) {
        layers.clear();
        input_dim = p.input_dim;
        epochs = p.epochs;
        batch_size = p.batch_size;
        loss = get_loss(p.loss);
        optimizer = get_optimizer(p.optimizer, p.optimizer_params);

        for (size_t i = 0; i < p.weights.size(); ++i) {
            int out_d = p.weights[i].size();
            int in_d = p.weights[i].empty() ? 0 : p.weights[i][0].size();
            add_layer(layer(in_d, out_d, get_activation(p.activations[i])));
            layers.back().get_weights() = p.weights[i];
            layers.back().get_bias() = p.biases[i];
        }

        optimizer.set_state_W(p.optimizer_state_W);
        optimizer.set_state_b(p.optimizer_state_b);
        optimizer.set_timestep(p.optimizer_timestep);
        optimizer.set_initialized(p.optimizer_initialized);
    }

    // Save Model directly to the ModelParams static initialization format
    void save_model(const std::string& filename = "output.h") {
        std::ofstream out(filename);
        int mod_in_dim = (input_dim > 0) ? input_dim : (layers.empty() ? -1 : layers.front().get_input_dim());

        out << "#ifndef MODEL_IMPORT\n#define MODEL_IMPORT\n\n#include \"model.h\"\n\nstatic nn::ModelParams model_param = {\n";

        // 1. Weights
        out << "  {\n";
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto& W = layers[i].get_weights();
            out << "    {";
            for (size_t r = 0; r < W.size(); ++r) {
                out << "{";
                for (size_t c = 0; c < W[r].size(); ++c) { out << W[r][c]; if (c + 1 < W[r].size()) out << ","; }
                out << "}"; if (r + 1 < W.size()) out << ",";
            }
            out << "}"; if (i + 1 < layers.size()) out << ",\n";
        }
        out << "  },\n";

        // 2. Biases
        out << "  {\n";
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto& b = layers[i].get_bias();
            out << "    {";
            for (size_t j = 0; j < b.size(); ++j) { out << b[j]; if (j + 1 < b.size()) out << ","; }
            out << "}"; if (i + 1 < layers.size()) out << ",\n";
        }
        out << "  },\n";

        // 3. Activations
        out << "  {";
        for (size_t i = 0; i < layers.size(); ++i) { out << "\"" << layers[i].get_activation().name() << "\""; if (i + 1 < layers.size()) out << ","; }
        out << "},\n";


        // 5. Loss, Optimizer, and Params
        out << "  \"" << loss.name() << "\",\n  \"" << optimizer.name << "\",\n  {";
        auto opt_params = optimizer.params();
        if (opt_params.empty()) opt_params.push_back(optimizer.lr);
        for (size_t i = 0; i < opt_params.size(); ++i) { out << opt_params[i]; if (i + 1 < opt_params.size()) out << ","; }
        out << "},\n";

        // 6. Metadata
        out << "  " << mod_in_dim << ",\n  " << epochs << ",\n  " << batch_size << ",\n  " << optimizer.get_timestep() << ",\n";

        // 7. Optimizer State W
        out << "  {\n";
        for (const auto& s : optimizer.get_state_W()) {
            out << "    {\n";
            for (size_t i = 0; i < s.size(); ++i) {
                out << "      {";
                for (size_t r = 0; r < s[i].size(); ++r) {
                    out << "{";
                    for (size_t c = 0; c < s[i][r].size(); ++c) { out << s[i][r][c]; if (c + 1 < s[i][r].size()) out << ","; }
                    out << "}"; if (r + 1 < s[i].size()) out << ",";
                }
                out << "}"; if (i + 1 < s.size()) out << ",\n";
            }
            out << "    },\n";
        }
        out << "  },\n";

        // 8. Optimizer State B
        out << "  {\n";
        for (const auto& s : optimizer.get_state_b()) {
            out << "    {\n";
            for (size_t i = 0; i < s.size(); ++i) {
                out << "      {";
                for (size_t j = 0; j < s[i].size(); ++j) { out << s[i][j]; if (j + 1 < s[i].size()) out << ","; }
                out << "}"; if (i + 1 < s.size()) out << ",\n";
            }
            out << "    },\n";
        }
        out << "  },\n";

        // 9. Initialized Boolean
        out << "  " << (optimizer.is_initialized() ? "true" : "false") << "\n};\n\n#endif\n";
    }

    std::vector<layer>& get_layers() { return layers; }
    void set_epochs(int e) { epochs = e; }
    int get_epochs() const { return epochs; }
    void set_batch_size(int b) { batch_size = b; }
    int get_batch_size() const { return batch_size; }
    int get_input_dim() const { return input_dim; }
};

}  // namespace nn

#endif