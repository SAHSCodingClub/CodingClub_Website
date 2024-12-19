<div class="heading" style="text-align: center; font-weight: 700; font-size: 3rem;">
Analysis of multilayer perceptrons
</div>

*By Adam Jacuch | December 17, 2024*

## Introduction

Multilayer perceptrons (MLPs) have played a pivotal role in the many breakthroughs seen by modern artificial intelligence (AI). Such neural networks, characterized by fully interconnected layers of neurons, are used anywhere from image recognition to natural language processing. This article aims to uncover the elegant math and concepts that create such powerful tools.


<img src="https://cdn-images-1.medium.com/v2/resize:fit:800/0*eaw1POHESc--l5yR.png" alt="image of multilayer perceptron" style="width: 60vw; height: auto;"/>

## Forward Pass

For starters, MLPs take numerical inputs. For instance, you can pass images as RGB or greyscale values. Inputs are in the form of an array of floating point digits. When you pass such data to an MLP, the input gets assigned to a correspoding neuron in the input layer (the size of the input layer must equal the size of the input data).

$$ \vec{inputData} = \vec{inputLayer} $$

Once the input layer contains the input data, we can begin the forward pass of the network. Forawrd pass refers to the data being passed forwards through layers to some arbitrary output layer. To begin, you first iterate through all the input layers and multiply them by a weight (this will be learned by the MLP). You then add this value into a respective neuron in the next layer. In MLPs, each neuron in a preceding layer has weighted connections to every neuron in the following layer. Therefore, we can represent a neurons value as:

$$ \sum_{n=0}^s \vec{neurons} \times \vec{weights} $$

Where s represents the previous layer size. You can simplify this further to:

$$ \vec{neurons} \cdot \vec{weights} $$

In hidden layers, once this operation is complete a bias (this is learnable by the MLP) is added and an activation function is applied. Biases allow the MLP to shift the activation function and better capture complex patterns in data. Similarly, activation functions introduce non-linearity into the network, enhasing the power of biases. Common examples of activation functions include sigmoid, hyperbolic tangent, and rectified liear units.

<img src="./images/ActivationFunctions.jpg" alt="Activation Functions" style="width: 60vw; height: auto;"/>

The math behind these operations looks like this:

$$ \sum_{n=0}^s a(value + bias) $$

## Backward Pass

Once the data reaches the output layer it would idealy contain some logical interpretation of the data. However, a randomly initalized MLP will return random values. This is where the backwards pass comes into place. The purpose of the backwards pass is to adjust the models weights in a way that minimizes loss, or the difference between what the AI predicts and the expected output. The expected output is user-defined, and should be present in the data that you feed the network. The loss function is typically a very high dimensional function (the exact number depends of the number of weights in a model). If a model has five weights the loss function will look like:

$$ l(a, b, c, d, e) $$

to minimize the loss, the caluate the partial derivatives of a change in a weight with respect to change in loss

$$ \frac{\partial w}{\partial l} $$

This is equivalent to caluclating the gradient of a function, shown below.

<img src="https://aleksandarhaber.com/wp-content/uploads/2023/05/gradient1-1.png" alt="Function Gradients" style="width: 60vw; height: auto;"/>

A gradient at a single point represents a vector pointing in the direction of steepest ascent. To minimize loss we move in a small direction opposite to the gradient. This small value is known as a learning rate, or how much the AI adjusts itself per iteration. This hyperparameter (a parameter you define yourself) value is crucial to an MLPs performance—set it too high and it over-adjusts, too low and it takes too mamy iterations (and thus computing power) to learn sufficiently. Learning rates are typically around 0.01. The models weght adjustment can be represented by:

$$ \Delta \vec{w} = -lr \times \vec{gradient} $$

Because the gradient is all of the change in weights with respect to loss, we can modify each weight by its respective partial derivative (times a small negative lerning rate). This will move the loss function in the direction of steepest descent. Repeat this process until the loss is acceptably low, or if it gets stuck at a local minimum (a place where going in any direction within a near vicinity increases loss).

## Training

Modern MLPs typically employ batch stochastic gradient descent. In batch stochastic gradient descent, gradients are accumulated for multiple instances of data before the model is updated. This prevents the model from "over fitting" to any one data peice (in the case of updating weights after each data forwards pass) or getting stuck at local minimums (in the case of accumulating gradients for an entire dataset before updating). However, because the model only updates in batches, this introduces stochasticity into the training (hence *stochastic* gradient descent) as the model is still influenced by biases from data samples.

Atfer enough forward and backward passes, the loss should become acceptably low, and you can begin using the model to make inferences on unseen data. A common practice to assess model quality is to leave some labeled (meaning that you already labeld the expected output) data out of the training set, and use this data to "test" the trained model. Ideally, the models test score should be similar (athough lower) than the training score. However, this difference can vary widely base on the data the model is trained on. For example, a model trained on a complex dataset will likely have a greater training-test gap and one traing on a straightforward one.

## Deployment

Assuming that the test score is acceptably high, you can begin depolying your model to real world tasks. When running your trained model, you no longer use the backwards pass, as there is no labeld output for the model to asses itself on.

## Implementation

Now that you are familiar with MLPs, it is time to write one. The idea is that this should help clarify any confusion from the above text, and give you a lower level look into how such models work. To begin, we define a neural network class that inherits a weight class (this is so the weight class can double for the gradients too). The following code is in c++:

```cpp
#include <iostream>
#include <ctime>
#include <vector>
#include <random>

using std::cout;
using std::vector;
using std::runtime_error;
using std::endl;

class Weights {

public:
    vector<vector<vector<double>>> connections;
    vector<vector<double>> biases;
};

class NN {

public:
    int networkSize;

    vector<vector<double>> neurons;
    Weights weights;
    Weights updates;
    int batchSize;

};
```

Next, I declare (and implement some) functions for the MLP. Aside from the functions already talked about (forward pass, backward pass, update, sigmoid and its derivative, many of which will be implemented later) we have init, clear, printResult, and randomVector (which respectivley initialize, clear previous forward passes, print the output layer, and return random vectors of a given size). These functions are not central to MLPs and therefore are only breifly covered.

```cpp
class NN {

public:
    int networkSize;

    vector<vector<double>> neurons;
    Weights weights;
    Weights updates;
    int batchSize;

    void init(int sizes[], int length) {
        networkSize = length;
        batchSize = 0;

        for (int i = 0; i < networkSize; i++) {
            neurons.push_back(vector<double> (sizes[i]));

            if (i < networkSize - 1) {
                if (i) weights.biases.push_back(randomVector(sizes[i]));

                vector<vector<double>> weightLayer, updateLayer;
                for (int j = 0; j < sizes[i]; j++) {
                    weightLayer.push_back(randomVector(sizes[i + 1]));
                    updateLayer.push_back(vector<double>(sizes[i + 1], 0.0));
                }
                weights.connections.push_back(weightLayer);
                updates.connections.push_back(updateLayer);
                if (i) updates.biases.push_back(vector<double>(sizes[i], 0.0));
            }
        }
    }

    void clear(int sizes[]) {
        neurons.clear();
        for (int i = 0; i < networkSize; i++) neurons.push_back(vector<double> (sizes[i]));
    }

    void printResult(vector<double> output) {
        for (double val : output) {
            cout << val << " ";
        }
        cout << endl;
    }

    vector<double> forwardPass(const vector<double>& input);
    void backwardPass(const vector<double>& expectedOutput);
    void update(double learningRate);

private:
    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    double sigmoidDerivative(double x) {
        return x * (1.0 - x);
    }

    vector<double> randomVector(int size) {
        srand(time(0));
        vector<double> result(size);
        for (int i = 0; i < size; i++) {
            result[i] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
        return result;
    }
};
```

Next, I implement the forward pass. As illustrated below, all neurons in one layer have connections to all neurons in the next layer, and each connection represents a unique weight that will be learned through the backwards pass. In hidden layers, biases and activation functions are applied, introducing nonlinearity.

```cpp
vector<double> NN::forwardPass(const vector<double>& input) {
    [[unlikely]] if (input.size() != neurons[0].size()) {
        throw runtime_error("Input size does not match the input layer size.");
    }

    neurons[0] = input;

    for (int i = 0; i < networkSize - 1; i++) {
        vector<double>& currentLayer = neurons[i];
        vector<double>& nextLayer = neurons[i + 1];

        for (size_t j = 0; j < nextLayer.size(); j++) {
            double sum = (i < networkSize - 2) ? weights.biases[i][j] : 0;
            for (size_t k = 0; k < currentLayer.size(); k++) {
                sum += currentLayer[k] * weights.connections[i][k][j];
            }

            nextLayer[j] = (i < networkSize - 2) ? sigmoid(sum) : sum;
        }
    }

    return neurons.back();
}
```
