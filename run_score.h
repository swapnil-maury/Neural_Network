#ifndef RUN_SCORES
#define RUN_SCORES

#include<vector>
#include <stdexcept>

double r2_score(const std::vector<double>& y_true,
                const std::vector<double>& y_pred)
{
    if (y_true.size() != y_pred.size())
        throw std::runtime_error("Size mismatch");

    double mean = 0.0;
    for (double v : y_true) mean += v;
    mean /= y_true.size();

    double ss_res = 0.0, ss_tot = 0.0;

    for (size_t i = 0; i < y_true.size(); i++) {
        ss_res += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
        ss_tot += (y_true[i] - mean) * (y_true[i] - mean);
    }

    return 1.0 - (ss_res / ss_tot);
}

double r2_score(const std::vector<std::vector<double>>& y_true,
                const std::vector<std::vector<double>>& y_pred)
{
    if (y_true.size() != y_pred.size())
        throw std::runtime_error("Size mismatch");

    double total_r2 = 0.0;
    int outputs = y_true[0].size();

    for (int j = 0; j < outputs; j++) {
        std::vector<double> yt, yp;

        for (size_t i = 0; i < y_true.size(); i++) {
            yt.push_back(y_true[i][j]);
            yp.push_back(y_pred[i][j]);
        }

        total_r2 += r2_score(yt, yp);
    }

    return total_r2 / outputs;
}

double accuracy_score(const std::vector<double>& y_true,
                      const std::vector<double>& y_pred)
{
    if (y_true.size() != y_pred.size())
        throw std::runtime_error("Size mismatch");

    int correct = 0;

    for (size_t i = 0; i < y_true.size(); i++) {
        int pred = (y_pred[i] >= 0.5) ? 1 : 0;
        int actual = (y_true[i] >= 0.5) ? 1 : 0;

        if (pred == actual) correct++;
    }

    return (double)correct / y_true.size();
}

double accuracy_score(const std::vector<std::vector<double>>& y_true,
                      const std::vector<std::vector<double>>& y_pred)
{
    if (y_true.size() != y_pred.size())
        throw std::runtime_error("Size mismatch");

    int correct = 0;
    int total = 0;

    for (size_t i = 0; i < y_true.size(); i++) {
        for (size_t j = 0; j < y_true[i].size(); j++) {
            int pred = (y_pred[i][j] >= 0.5) ? 1 : 0;
            int actual = (y_true[i][j] >= 0.5) ? 1 : 0;

            if (pred == actual) correct++;
            total++;
        }
    }

    return (double)correct / total;
}
// --- NEW: MULTI-CLASS CLASSIFICATION (For MNIST) ---

// Helper function to find the index of the highest value in a vector
inline int argmax(const std::vector<double>& vec) {
    return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
}

// Calculates accuracy by comparing the highest predicted probability to the true one-hot label
inline double accuracy_score_multiclass(const std::vector<std::vector<double>>& y_true,
                                        const std::vector<std::vector<double>>& y_pred)
{
    if (y_true.size() != y_pred.size())
        throw std::runtime_error("Size mismatch");

    int correct = 0;
    for (size_t i = 0; i < y_true.size(); i++) {
        if (argmax(y_true[i]) == argmax(y_pred[i])) {
            correct++;
        }
    }
    return (double)correct / y_true.size();
}

// Generates an N x N Confusion Matrix
inline std::vector<std::vector<int>> confusion_matrix(const std::vector<std::vector<double>>& y_true,
                                                      const std::vector<std::vector<double>>& y_pred)
{
    if (y_true.size() != y_pred.size())
        throw std::runtime_error("Size mismatch");

    int num_classes = y_true[0].size();
    std::vector<std::vector<int>> matrix(num_classes, std::vector<int>(num_classes, 0));

    for (size_t i = 0; i < y_true.size(); i++) {
        int actual_class = argmax(y_true[i]);
        int predicted_class = argmax(y_pred[i]);
        matrix[actual_class][predicted_class]++;
    }

    return matrix;
}

// Beautifully prints the confusion matrix to the terminal
inline void print_confusion_matrix(const std::vector<std::vector<int>>& matrix)
{
    int num_classes = matrix.size();
    std::cout << "\n--- Confusion Matrix ---\n";
    std::cout << std::setw(6) << " ";
    for (int i = 0; i < num_classes; i++) {
        std::cout << std::setw(6) << "P_" << i;
    }
    std::cout << "\n";

    for (int i = 0; i < num_classes; i++) {
        std::cout << std::setw(4) << "T_" << i << " |";
        for (int j = 0; j < num_classes; j++) {
            std::cout << std::setw(7) << matrix[i][j];
        }
        std::cout << "\n";
    }
    std::cout << "------------------------\n";
    std::cout << "(T = True Label, P = Predicted Label)\n";
}

#endif