#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <chrono>

// my custom c++ header files
#include "color.h"
#include "model.h"
#include "run_score.h"

using namespace std;

vector<vector<double>> load_X_bin(const string &filename, int samples, int features = 784)
{
    vector<vector<double>> data(samples, vector<double>(features));
    ifstream in(filename, ios::binary);
    if (!in)
    {
        cerr << "Could not open " << filename << "\n";
        return data;
    }

    for (int i = 0; i < samples; ++i)
    {
        in.read(reinterpret_cast<char *>(data[i].data()), features * sizeof(double));
    }
    return data;
}

vector<vector<double>> load_Y_bin(const string &filename, int samples, int classes = 10)
{
    vector<vector<double>> data(samples, vector<double>(classes));
    ifstream in(filename, ios::binary);
    if (!in)
    {
        cerr << "Could not open " << filename << "\n";
        return data;
    }

    for (int i = 0; i < samples; ++i)
    {
        in.read(reinterpret_cast<char *>(data[i].data()), classes * sizeof(double));
    }
    return data;
}

void printImage(const vector<double> &image, int height, int width)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            double pixel_val = image[i * width + j];
            pixel_val = max(0.0, min(1.0, pixel_val));
            int val = static_cast<int>(pixel_val * 255.0);
            string bg_color = "rgb(" + to_string(val) + "," + to_string(val) + "," + to_string(val) + ")";
            cout << terminal_color::color("", "  ", bg_color);
        }
        cout << endl;
    }
}

vector<double> apply_softmax(const vector<double> &logits)
{
    if (logits.empty())
        return {};
    double max_logit = *max_element(logits.begin(), logits.end());
    vector<double> probs(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i)
    {
        probs[i] = exp(logits[i] - max_logit);
        sum += probs[i];
    }
    for (double &p : probs)
        p /= sum;
    return probs;
}

int main()
{
    int train_samples = 60000;
    int test_samples = 10000;
    auto program_start = chrono::high_resolution_clock::now();

    cout << "Loading data...\n";
    auto X_train = load_X_bin("train_X.bin", train_samples, 784);
    auto Y_train = load_Y_bin("train_Y.bin", train_samples, 10);

    auto X_test = load_X_bin("test_X.bin", test_samples, 784);
    auto Y_test = load_Y_bin("test_Y.bin", test_samples, 10);

    cout << "Data loaded completely!\n";

    cout << "Building/Loading Model...\n";

    // nn::SequentialNetwork model;
    // model.load_binary("model.bin");
    LossFunction loss = losses::SoftmaxCrossEntropy();
    nn::Optimizer opt = nn::Adam(0.0008);
    nn::SequentialNetwork model(loss, opt, 6);

    ActivationFunction relu = activations::ReLU();

    // Hidden Layers (Dropout pointer fixed!)
    model.add_dense_layer(784, 128, relu);
    model.add_dense_layer(128, 64, relu);

    // Output Layer (10 digits)
    model.add_dense_layer(64, 10, activations::Identity());
    model.set_threads(10);
    model.fit(X_train, Y_train, 128);

    // Predict

    cout << "Switching to evaluation mode...\n";
    cout << "Making predictions on test data...\n";
    auto logits_preds = model.predict(X_test);

    auto program_end = chrono::high_resolution_clock::now();

    auto duration =
        chrono::duration_cast<chrono::seconds>(
            program_end - program_start);

    cout << "\nTotal execution time: "
         << duration.count() << " seconds\n";

    // Get true test classification accuracy
    double acc = accuracy_score_multiclass(Y_test, logits_preds);
    cout << "\n========================================\n";
    cout << "Test Accuracy: " << acc * 100.0 << "%\n";
    cout << "========================================\n";

    // --- INTEGRATED CONFUSION MATRIX ---
    auto conf_matrix = confusion_matrix(Y_test, logits_preds);
    print_confusion_matrix(conf_matrix);
    // -----------------------------------

    while (true)
    {
        int image_number;
        cout << "\nEnter an image index (0 to " << test_samples - 1 << ") to visualize (or -1 to exit): ";
        cin >> image_number;

        if (image_number < 0 || image_number >= test_samples)
            break;

        cout << "\n--- VISUALIZING TEST IMAGE [" << image_number << "] ---\n";
        printImage(X_test[image_number], 28, 28);
        cout << "\n--- PREDICTION VS ACTUAL ---\n";

        vector<double> probabilities = apply_softmax(logits_preds[image_number]);

        int predicted_digit = 0;
        double max_prob = probabilities[0];
        int true_digit = 0;

        cout << "True Y_test[" << image_number << "] (One-Hot): ";
        for (int i = 0; i < 10; i++)
        {
            cout << Y_test[image_number][i] << " ";
            if (Y_test[image_number][i] == 1.0)
            {
                true_digit = i;
            }
        }
        cout << "\nActual Digit: " << true_digit << "\n\n";

        cout << "Model Predictions (Probabilities):\n";
        for (int i = 0; i < 10; i++)
        {
            cout << "Class " << i << ": " << fixed << setprecision(2)
                 << probabilities[i] * 100.0 << "%\n";

            if (probabilities[i] > max_prob)
            {
                max_prob = probabilities[i];
                predicted_digit = i;
            }
        }
        cout << "\nPredicted Digit: " << predicted_digit << "\n";

        if (predicted_digit == true_digit)
        {
            cout << "Result: CORRECT! 🎉\n";
        }
        else
        {
            cout << "Result: WRONG! ❌\n";
        }
    }

    cout << "\nGenerating Model Summary...\n";
    model.summary();

    // Save model to output.h before exiting
    // model.save_model("output.h");
    // model.save_binary();

    return 0;
}
