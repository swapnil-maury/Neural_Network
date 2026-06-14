#ifndef ACTIVATION_FUNC_H
#define ACTIVATION_FUNC_H

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <utility>
#include <vector>

class ActivationFunction {
private:
    std::function<double(double)> activation_;
    std::function<double(double)> derivative_;
    std::string name_;

public:
    ActivationFunction(std::function<double(double)> activation,
                       std::function<double(double)> derivative,
                       std::string name = "custom")
        : activation_(std::move(activation)),
          derivative_(std::move(derivative)),
          name_(std::move(name)) {}

    double activate(double z) const {
        return activation_(z);
    }

    double grad(double z) const {
        return derivative_(z);
    }

    const std::string& name() const {
        return name_;
    }
};

namespace activations {

inline ActivationFunction Identity() {
    return ActivationFunction(
        [](double z) { return z; },
        [](double) { return 1.0; },
        "identity"
    );
}

inline ActivationFunction BinaryStep() {
    return ActivationFunction(
        [](double z) { return z >= 0.0 ? 1.0 : 0.0; },
        [](double) { return 0.0; },
        "binary_step"
    );
}

inline ActivationFunction Sigmoid() {
    return ActivationFunction(
        [](double z) { return 1.0 / (1.0 + std::exp(-z)); },
        [](double z) {
            const double s = 1.0 / (1.0 + std::exp(-z));
            return s * (1.0 - s);
        },
        "sigmoid"
    );
}

inline ActivationFunction Tanh() {
    return ActivationFunction(
        [](double z) { return std::tanh(z); },
        [](double z) {
            const double t = std::tanh(z);
            return 1.0 - t * t;
        },
        "tanh"
    );
}

inline ActivationFunction ReLU() {
    return ActivationFunction(
        [](double z) { return std::max(0.0, z); },
        [](double z) { return z > 0.0 ? 1.0 : 0.0; },
        "relu"
    );
}

inline ActivationFunction ReLU6() {
    return ActivationFunction(
        [](double z) { return std::min(6.0, std::max(0.0, z)); },
        [](double z) { return (z > 0.0 && z < 6.0) ? 1.0 : 0.0; },
        "relu6"
    );
}

inline ActivationFunction LeakyReLU(double alpha = 0.01) {
    return ActivationFunction(
        [alpha](double z) { return z > 0.0 ? z : alpha * z; },
        [alpha](double z) { return z > 0.0 ? 1.0 : alpha; },
        "leaky_relu"
    );
}

inline ActivationFunction PReLU(double alpha = 0.25) {
    return ActivationFunction(
        [alpha](double z) { return z > 0.0 ? z : alpha * z; },
        [alpha](double z) { return z > 0.0 ? 1.0 : alpha; },
        "prelu"
    );
}

inline ActivationFunction ELU(double alpha = 1.0) {
    return ActivationFunction(
        [alpha](double z) { return z >= 0.0 ? z : alpha * (std::exp(z) - 1.0); },
        [alpha](double z) { return z >= 0.0 ? 1.0 : alpha * std::exp(z); },
        "elu"
    );
}

inline ActivationFunction SELU(double lambda = 1.0507009873554805,double alpha = 1.6732632423543772) {
    return ActivationFunction(
        [lambda, alpha](double z) {
            return lambda * (z >= 0.0 ? z : alpha * (std::exp(z) - 1.0));
        },
        [lambda, alpha](double z) {
            return lambda * (z >= 0.0 ? 1.0 : alpha * std::exp(z));
        },
        "selu"
    );
}

inline ActivationFunction Softplus() {
    return ActivationFunction(
        [](double z) {
            if (z > 20.0) {
                return z;
            }
            if (z < -20.0) {
                return std::exp(z);
            }
            return std::log1p(std::exp(z));
        },
        [](double z) {
            return 1.0 / (1.0 + std::exp(-z));
        },
        "softplus"
    );
}

inline ActivationFunction Softsign() {
    return ActivationFunction(
        [](double z) {
            return z / (1.0 + std::abs(z));
        },
        [](double z) {
            const double d = 1.0 + std::abs(z);
            return 1.0 / (d * d);
        },
        "softsign"
    );
}

inline ActivationFunction Swish(double beta = 1.0) {
    return ActivationFunction(
        [beta](double z) {
            const double s = 1.0 / (1.0 + std::exp(-beta * z));
            return z * s;
        },
        [beta](double z) {
            const double s = 1.0 / (1.0 + std::exp(-beta * z));
            return s + beta * z * s * (1.0 - s);
        },
        "swish"
    );
}

inline ActivationFunction SiLU() {
    return Swish(1.0);
}

inline ActivationFunction GELU() {
    constexpr double kInvSqrt2 = 0.7071067811865475;
    constexpr double kInvSqrt2Pi = 0.3989422804014327;

    return ActivationFunction(
        [](double z) {
            constexpr double invSqrt2 = 0.7071067811865475;
            return 0.5 * z * (1.0 + std::erf(z * invSqrt2));
        },
        [kInvSqrt2, kInvSqrt2Pi](double z) {
            return 0.5 * (1.0 + std::erf(z * kInvSqrt2))
                   + z * kInvSqrt2Pi * std::exp(-0.5 * z * z);
        },
        "gelu"
    );
}

inline ActivationFunction Mish() {
    return ActivationFunction(
        [](double z) {
            const double sp = z > 20.0 ? z : std::log1p(std::exp(z));
            return z * std::tanh(sp);
        },
        [](double z) {
            const double sp = z > 20.0 ? z : std::log1p(std::exp(z));
            const double t = std::tanh(sp);
            const double s = 1.0 / (1.0 + std::exp(-z));
            return t + z * s * (1.0 - t * t);
        },
        "mish"
    );
}

inline ActivationFunction HardSigmoid() {
    return ActivationFunction(
        [](double z) {
            return std::min(1.0, std::max(0.0, (z + 3.0) / 6.0));
        },
        [](double z) {
            return (z > -3.0 && z < 3.0) ? (1.0 / 6.0) : 0.0;
        },
        "hard_sigmoid"
    );
}

inline ActivationFunction HardSwish() {
    return ActivationFunction(
        [](double z) {
            const double hs = std::min(1.0, std::max(0.0, (z + 3.0) / 6.0));
            return z * hs;
        },
        [](double z) {
            if (z <= -3.0) {
                return 0.0;
            }
            if (z >= 3.0) {
                return 1.0;
            }
            return z / 3.0 + 0.5;
        },
        "hard_swish"
    );
}

inline std::vector<double> Softmax(const std::vector<double>& logits) {
    if (logits.empty()) {
        return {};
    }

    const double max_logit = *std::max_element(logits.begin(), logits.end());

    std::vector<double> exp_vals(logits.size());
    double sum = 0.0;

    for (size_t i = 0; i < logits.size(); ++i) {
        exp_vals[i] = std::exp(logits[i] - max_logit);
        sum += exp_vals[i];
    }

    for (double& v : exp_vals) {
        v /= sum;
    }

    return exp_vals;
}

}

#endif
