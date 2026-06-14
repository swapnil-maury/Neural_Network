#ifndef LOSS_FUNC_H
#define LOSS_FUNC_H

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class LossFunction {
private:
    std::function<double(double, double)> loss_fn_;
    std::function<double(double, double)> deriv_fn_;
    std::string name_;

public:
    LossFunction(std::function<double(double, double)> loss_fn,
                 std::function<double(double, double)> deriv_fn,
                 std::string name = "custom")
        : loss_fn_(std::move(loss_fn)),
          deriv_fn_(std::move(deriv_fn)),
          name_(std::move(name)) {}

    // Computes the raw loss for a single sample: L(y_pred, y_true)
    double compute_loss(double y_pred, double y_true) const {
        return loss_fn_(y_pred, y_true);
    }

    // Computes the raw derivative: dL/dy_pred
    double derivative(double y_pred, double y_true) const {
        return deriv_fn_(y_pred, y_true);
    }

    double lossvec(const std::vector<double>& y_pred,const std::vector<double>& y_true) const {

        double loss = 0.0;
        for (size_t i = 0; i < y_pred.size(); ++i) {
            loss += loss_fn_(y_pred[i], y_true[i]);
        }
        return loss / y_pred.size();
    }

    std::vector<double> gradvec(const std::vector<double>& y_pred,
                                const std::vector<double>& y_true) const {

        std::vector<double> grad(y_pred.size());

        for (size_t i = 0; i < y_pred.size(); ++i) {
            grad[i] = deriv_fn_(y_pred[i], y_true[i]);
        }
        return grad;
    }


    const std::string& name() const {
        return name_;
    }
};

namespace losses {

// Small epsilon for numerical stability
inline constexpr double kEps = 1e-12;

// Helper to keep probabilities in (0, 1) range
inline double clamp_prob(double p) {
    return std::min(1.0 - kEps, std::max(kEps, p));
}

// --- Scalar Loss Functions ---

inline LossFunction MSE() {
    return LossFunction(
        [](double y_pred, double y_true) {
            double e = y_pred - y_true;
            return e * e;
        },
        [](double y_pred, double y_true) {
            return 2.0 * (y_pred - y_true);
        },
        "MSE"
    );
}

inline LossFunction MAE() {
    return LossFunction(
        [](double y_pred, double y_true) {
            return std::abs(y_pred - y_true);
        },
        [](double y_pred, double y_true) {
            double e = y_pred - y_true;
            if (e > 0.0) return 1.0;
            if (e < 0.0) return -1.0;
            return 0.0;
        },
        "MAE"
    );
}

inline LossFunction BinaryCrossEntropy() {
    return LossFunction(
        [](double y_pred, double y_true) {
            double p = clamp_prob(y_pred);
            return -(y_true * std::log(p) + (1.0 - y_true) * std::log(1.0 - p));
        },
        [](double y_pred, double y_true) {
            double p = clamp_prob(y_pred);
            return (p - y_true) / (p * (1.0 - p));
        },
        "BinaryCrossEntropy"
    );
}

inline LossFunction Hinge() {
    return LossFunction(
        [](double y_pred, double y_true) {
            return std::max(0.0, 1.0 - y_true * y_pred);
        },
        [](double y_pred, double y_true) {
            return (1.0 - y_true * y_pred > 0.0) ? -y_true : 0.0;
        },
        "Hinge"
    );
}

inline LossFunction SquaredHinge() {
    return LossFunction(
        [](double y_pred, double y_true) {
            double margin = std::max(0.0, 1.0 - y_true * y_pred);
            return margin * margin;
        },
        [](double y_pred, double y_true) {
            double margin = 1.0 - y_true * y_pred;
            return (margin > 0.0) ? (-2.0 * y_true * margin) : 0.0;
        },
        "SquaredHinge"
    );
}

inline LossFunction Huber(double delta = 1.0) {
    return LossFunction(
        [delta](double y_pred, double y_true) {
            double e = y_pred - y_true;
            double ae = std::abs(e);
            return (ae <= delta) ? (0.5 * e * e) : (delta * (ae - 0.5 * delta));
        },
        [delta](double y_pred, double y_true) {
            double e = y_pred - y_true;
            if (std::abs(e) <= delta) return e;
            return (e > 0.0 ? delta : -delta);
        },
        "Huber"
    );
}

inline LossFunction LogCosh() {
    return LossFunction(
        [](double y_pred, double y_true) {
            return std::log(std::cosh(y_pred - y_true));
        },
        [](double y_pred, double y_true) {
            return std::tanh(y_pred - y_true);
        },
        "LogCosh"
    );
}

inline LossFunction MSLE() {
    return LossFunction(
        [](double y_pred, double y_true) {
            double p = std::max(y_pred, -1.0 + kEps);
            double t = std::max(y_true, -1.0 + kEps);
            double diff = std::log1p(p) - std::log1p(t);
            return diff * diff;
        },
        [](double y_pred, double y_true) {
            double p = std::max(y_pred, -1.0 + kEps);
            double t = std::max(y_true, -1.0 + kEps);
            return (2.0 * (std::log1p(p) - std::log1p(t))) / (1.0 + p);
        },
        "MSLE"
    );
}

inline LossFunction Poisson() {
    return LossFunction(
        [](double y_pred, double y_true) {
            double lambda = std::max(y_pred, kEps);
            return lambda - y_true * std::log(lambda);
        },
        [](double y_pred, double y_true) {
            double lambda = std::max(y_pred, kEps);
            return 1.0 - (y_true / lambda);
        },
        "Poisson"
    );
}

inline LossFunction KLDivergence() {
    return LossFunction(
        [](double y_pred, double y_true) {
            double p = clamp_prob(y_pred);
            double t = std::max(y_true, kEps);
            return t * std::log(t / p);
        },
        [](double y_pred, double y_true) {
            double p = clamp_prob(y_pred);
            double t = std::max(y_true, kEps);
            return -(t / p);
        },
        "KLDivergence"
    );
}

// --- Categorical/Vector Functions ---

inline std::vector<double> softmax(const std::vector<double>& logits) {
    if (logits.empty()) return {};
    const double max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<double> exps(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        exps[i] = std::exp(logits[i] - max_logit);
        sum += exps[i];
    }
    for (double& v : exps) v /= sum;
    return exps;
}

inline double CategoricalCrossEntropy(const std::vector<double>& y_pred, const std::vector<double>& y_true) {
    if (y_pred.size() != y_true.size() || y_pred.empty()) {
        throw std::invalid_argument("CCE size mismatch.");
    }
    double loss = 0.0;
    for (size_t i = 0; i < y_pred.size(); ++i) {
        loss += -y_true[i] * std::log(clamp_prob(y_pred[i]));
    }
    return loss;
}

inline std::vector<double> CategoricalCrossEntropyGrad(const std::vector<double>& y_pred, const std::vector<double>& y_true) {
    std::vector<double> grad(y_pred.size());
    for (size_t i = 0; i < y_pred.size(); ++i) {
        grad[i] = -y_true[i] / clamp_prob(y_pred[i]);
    }
    return grad;
}

inline double SoftmaxCrossEntropyWithLogits(const std::vector<double>& logits, const std::vector<double>& y_true) {
    return CategoricalCrossEntropy(softmax(logits), y_true);
}

inline std::vector<double> SoftmaxCrossEntropyWithLogitsGrad(const std::vector<double>& logits, const std::vector<double>& y_true) {
    std::vector<double> probs = softmax(logits);
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] -= y_true[i];
    }
    return probs;
}

}

#endif