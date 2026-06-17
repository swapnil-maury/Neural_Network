#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <omp.h> // Include OpenMP

#include "model.h"
#include "run_score.h"
#include "output.h"

using namespace std;

// Reads X data (Images)
std::vector<std::vector<double>> load_X_bin(const std::string& filename, int samples, int features = 784) {
    std::vector<std::vector<double>> data(samples, std::vector<double>(features));
    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr << "Could not open " << filename << "\n"; return data; }
    
    for (int i = 0; i < samples; ++i) {
        in.read(reinterpret_cast<char*>(data[i].data()), features * sizeof(double));
    }
    return data;
}

// Reads Y data (One-Hot Labels)
std::vector<std::vector<double>> load_Y_bin(const std::string& filename, int samples, int classes = 10) {
    std::vector<std::vector<double>> data(samples, std::vector<double>(classes));
    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr << "Could not open " << filename << "\n"; return data; }
    
    for (int i = 0; i < samples; ++i) {
        in.read(reinterpret_cast<char*>(data[i].data()), classes * sizeof(double));
    }
    return data;
}

// Prints the normalized image to the terminal
void printImage(const vector<double>& image, int height, int width) {
    for(int i = 0 ; i < height; i++) {
        for(int j = 0 ; j < width; j++) {
            double pixel_val = image[i * width + j];
            if (pixel_val > 0.6) std::cout << "██";
            else if (pixel_val > 0.2) std::cout << "▒▒";
            else std::cout << "  ";
        }
        std::cout << std::endl;
    }
}

// Helper function to turn raw logits into readable probabilities for the visualization loop
std::vector<double> apply_softmax(const std::vector<double>& logits) {
    if (logits.empty()) return {};
    double max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probs(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    for (double& p : probs) p /= sum;
    return probs;
}

int main() {
    int train_samples = 60000; 
    int test_samples  = 10000; 

    // Maximize OpenMP core usage
    Eigen::setNbThreads(8);

    std::cout << "Loading data...\n";
    auto X_train = load_X_bin("train_X.bin", train_samples, 784);
    auto Y_train = load_Y_bin("train_Y.bin", train_samples, 10);
    
    auto X_test = load_X_bin("test_X.bin", test_samples, 784);
    auto Y_test = load_Y_bin("test_Y.bin", test_samples, 10);
    
    // std::cout << "Normalizing data to 0.0 - 1.0...\n";
    // for (auto& image : X_train) {
    //     for (auto& pixel : image) pixel /= 255.0;
    // }
    // for (auto& image : X_test) {
    //     for (auto& pixel : image) pixel /= 255.0;
    // }
    std::cout << "Data loaded completely!\n";
    
    std::cout << "Building/Loading Model...\n";
    
    // nn::SequentialNetwork model;
    
    // NOTE: You currently have model.load_model active. 
    // If you want to train from scratch, comment out load_model and uncomment the build/fit lines.
    // model.load_model(model_param); 

    // --- TO TRAIN FROM SCRATCH, USE THIS INSTEAD ---
    LossFunction loss_fn = losses::SoftmaxCrossEntropy();
    nn::Optimizer opt = nn::Adam(0.0003);
    int epochs = 100;
    
    nn::SequentialNetwork model(loss_fn, opt, epochs);
    model.add_dense_layer(784, 128, activations::ReLU());
    model.add_layer(nn::DropoutLayer(0.2));
    model.add_dense_layer(128, 64, activations::ReLU());
    
    // Final layer MUST be identity for the fused loss
    model.add_dense_layer(64, 10, activations::Identity()); 
    
    // Train with large batch size
    model.fit(X_train, Y_train, 256);
    // -----------------------------------------------

    // 3. Predict on UNSEEN Test Data
    std::cout << "Switching to evaluation mode...\n";
    std::cout << "Making predictions on test data...\n";
    auto logits_preds = model.predict(X_test);

    // Get true test classification accuracy
    double acc = accuracy_score_multiclass(Y_test, logits_preds);
    std::cout << "\n========================================\n";
    std::cout << "Test Accuracy: " << acc * 100.0 << "%\n";
    std::cout << "========================================\n";
    
    // --- INTEGRATED CONFUSION MATRIX ---
    auto conf_matrix = confusion_matrix(Y_test, logits_preds);
    print_confusion_matrix(conf_matrix);
    // -----------------------------------
    
    while(true) {
        int image_number;
        std::cout << "\nEnter an image index (0 to " << test_samples - 1 << ") to visualize (or -1 to exit): ";
        std::cin >> image_number;
        
        if(image_number < 0 || image_number >= test_samples) break;
    
        std::cout << "\n--- VISUALIZING TEST IMAGE [" << image_number << "] ---\n";
        printImage(X_test[image_number], 28, 28);
        std::cout << "\n--- PREDICTION VS ACTUAL ---\n";
        
        std::vector<double> probabilities = apply_softmax(logits_preds[image_number]);
        
        int predicted_digit = 0;
        double max_prob = probabilities[0]; 
        int true_digit = 0;
    
        std::cout << "True Y_test[" << image_number << "] (One-Hot): ";
        for(int i = 0; i < 10; i++) {
            std::cout << Y_test[image_number][i] << " "; 
            if (Y_test[image_number][i] == 1.0) {
                true_digit = i;
            }
        }
        std::cout << "\nActual Digit: " << true_digit << "\n\n";
    
        std::cout << "Model Predictions (Probabilities):\n";
        for(int i = 0 ; i < 10; i++) {
            std::cout << "Class " << i << ": " << std::fixed << std::setprecision(2) 
                      << probabilities[i] * 100.0 << "%\n"; 
            
            if (probabilities[i] > max_prob) {
                max_prob = probabilities[i];
                predicted_digit = i;
            }
        }
        std::cout << "\nPredicted Digit: " << predicted_digit << "\n";
        
        if (predicted_digit == true_digit) {
            std::cout << "Result: CORRECT! 🎉\n";
        } else {
            std::cout << "Result: WRONG! ❌\n";
        }
    }

    std::cout << "\nGenerating Model Summary...\n";
    // --- INTEGRATED MODEL SUMMARY ---
    model.summary();
    // --------------------------------
    
    // Save model to output.h before exiting
    model.save_model("output.h"); 
    
    std::cout << "\nExiting Framework. Great job!\n";
    return 0;
}