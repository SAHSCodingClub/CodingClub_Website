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

In hidden layers, once this operation is complete a bias (this is learnable by the MLP) is added and an activation function is applied. Biases allow the MLP to shift the activation function and better capture complex patterns in data. Similarly, activation functions introduce non-linearity into the network, enhansing the power of biases. Common examples of activation functions include sigmoid, hyperbolic tangent, and rectified linear units.

<img src="./images/ActivationFunctions.jpg" alt="Activation Functions" style="width: 60vw; height: auto;"/>

The math behind these operations looks like this:

$$ \sum_{n=0}^s a(value + bias) $$

## Backward Pass

Once the data reaches the output layer it would ideally contain some logical interpretation of the data. However, a randomly initalized MLP will return random values. This is where the backward pass comes into place. The purpose of the backward pass is to adjust the models weights in a way that minimizes loss, or the difference between what the AI predicts and the expected output. The expected output is user-defined, and should be present in the data that you feed the network. The loss function is typically a very high dimensional function (the exact number depends of the number of weights in a model). If a model has five weights the loss function will look like:

$$ l(a, b, c, d, e) $$

to minimize loss, you calculate the partial derivatives of a change in a weight with respect to change in loss

$$ \frac{\partial w}{\partial l} $$

This is equivalent to caluclating the gradient of a function, shown below.

<img src="https://aleksandarhaber.com/wp-content/uploads/2023/05/gradient1-1.png" alt="Function Gradients" style="width: 60vw; height: auto;"/>

A gradient at a single point represents a vector pointing in the direction of steepest ascent. To minimize loss we move in a small direction opposite to the gradient. This small value is known as a learning rate, or how much the AI adjusts itself per iteration. This hyperparameter (a parameter you define yourself) value is crucial to an MLPs performance—set it too high and it over-adjusts, too low and it takes too mamy iterations (and thus computing power) to learn sufficiently. Learning rates are typically around 0.01. The models weght adjustment can be represented by:

$$ \Delta \vec{w} = -lr \times \vec{gradient} $$

Because the gradient is all of the change in weights with respect to loss, we can modify each weight by its respective partial derivative (times a small negative lerning rate). This will move the loss function in the direction of steepest descent. Repeat this process until the loss is acceptably low, or if it gets stuck at a local minimum (a place where going in any direction within a near vicinity increases loss).

## Training

Modern MLPs typically employ batch stochastic gradient descent. In batch stochastic gradient descent, gradients are accumulated for multiple instances of data before the model is updated. This prevents the model from "over fitting" to any one data peice (in the case of updating weights after each data forwards pass) or getting stuck at local minimums (in the case of accumulating gradients for an entire dataset before updating). However, because the model only updates in batches, this introduces stochasticity into the training (hence *stochastic* gradient descent) as the model is still influenced by biases from data samples.

Atfer enough forward and backward passes, the loss should become acceptably low, and you can begin using the model to make inferences on unseen data. A common practice to assess model quality is to leave some labeled (meaning that you already labeld the expected output) data out of the training set, and use this data to "test" the trained model. Ideally, the models test score should be similar (athough lower) than the training score. However, this difference can vary widely base on the data the model is trained on. For example, a model trained on a complex dataset will likely have a greater training-test gap than one trained on a straightforward one.

## Deployment

Assuming that the test score is acceptably high, you can begin depolying your model to real world tasks. When running your trained model, you no longer use the backwards pass, as there is no labeled output for the model to assess itself on.

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

Next, I implement the backward pass. The following code differs slightly from what was previously stated due to mathematical optimizations. To start, the function calculates the deltas of the neurons (essentially it assigns blame; neurons that contribute more to loss have higher magnitude deltas). It is important to note that deltas do not represent how a neurons value should be changed to set loss to zero—rather, it represents the partial derivative of the neuron with respect to loss. When storing weight gradients, the biases are simply the deltas as they directly affect how a neuron fires. However, because connection weights are a function of the neurons value

$$ neuron \times weight $$

it is necessary that we multiply the delta by its respective neuron value when calculating connection gradients. Once the gradient is calculated it is added to a gradient pool (one gradient that stores the sum of all caluclated gradients), where it will later be used to update the MLPs weights.


```cpp
void NN::backwardPass(const vector<double>& expectedOutput) {
    [[unlikely]] if (expectedOutput.size() != neurons.back().size()) {
        throw runtime_error("Expected output size does not match output layer size.");
    }

    vector<vector<double>> deltas(networkSize);

    // Compute output layer error
    vector<double>& outputLayer = neurons.back();
    deltas[networkSize - 1] = vector<double>(outputLayer.size());
    for (size_t i = 0; i < outputLayer.size(); i++) {
        double error = outputLayer[i] - expectedOutput[i];
        deltas[networkSize - 1][i] = error;
    }

    // Backpropagate through hidden layers
    for (int i = networkSize - 2; i > 0; i--) {
        deltas[i] = vector<double>(neurons[i].size());
        for (size_t j = 0; j < neurons[i].size(); j++) {
            double sum = 0.0;
            for (size_t k = 0; k < neurons[i + 1].size(); k++) {
                sum += deltas[i + 1][k] * weights.connections[i][j][k];
            }
            deltas[i][j] = sum * sigmoidDerivative(neurons[i][j]);
        }
    }

    // Accumulate updates
    for (int i = 0; i < networkSize - 1; i++) {
        for (size_t j = 0; j < neurons[i].size(); j++) {
            for (size_t k = 0; k < neurons[i + 1].size(); k++) {
                updates.connections[i][j][k] += deltas[i + 1][k] * neurons[i][j];
            }
        }
        if (i < networkSize - 2) {
            for (size_t j = 0; j < neurons[i + 1].size(); j++) {
                updates.biases[i][j] += deltas[i + 1][j];
            }
        }
    }

    batchSize++;
}
```

Next, I implement the update function. This is resposible for updating the MLPs parameters using the stored gradients from the backward passes. It also clears the gradient pool after so that future gradients are not biased by outdated ones.

```cpp
void NN::update(double learningRate) {
    for (int i = 0; i < networkSize - 1; i++) {
        for (size_t j = 0; j < neurons[i].size(); j++) {
            for (size_t k = 0; k < neurons[i + 1].size(); k++) {
                weights.connections[i][j][k] -= learningRate * (updates.connections[i][j][k] / batchSize);
                updates.connections[i][j][k] = 0.0;
            }
        }
        if (i < networkSize - 2) {
            for (size_t j = 0; j < neurons[i + 1].size(); j++) {
                weights.biases[i][j] -= learningRate * (updates.biases[i][j] / batchSize);
                updates.biases[i][j] = 0.0;
            }
        }
    }

    batchSize = 0;
}
```

Now that I wrote the MLP, it is time to write code that runs one. Essentially, I call the functions above recursivley, repeating forward and backward passes until the loss is acceptably low. The code below is very basic and does not utilize the full power of batch stochastic gradient descent, as I immediately update weights after each forward and backward pass. Furthermore, the example below only uses one peice of data.

```cpp
int main() {
    int networkSize[] = {3, 5, 2};
    NN nn;
    nn.init(networkSize, 3);

    vector<double> input = {0.5, 0.8, 0.2};

    for (int i = 0; i < 1000; i++) {
        vector<double> output = nn.forwardPass(input);
        if (i % 100 == 0) nn.printResult(output);
        nn.backwardPass({1, 0});
        nn.clear(networkSize);
        nn.update(0.01);
    }

    return 0;
}
```

In order to train in batches, simply limit the amount of times update is called. For example, call it after every ten forward and backward passes. In order to train the model on a large number of data, it would be wise to implement some sort of csv reading/store the data in a separate file for readability. I have not done this in order to make the code as simple and readable as possible, and also because that is not the main focus of this article.

## Conclusion

In summary, MLPs are very powerful tools for developing AI. This article covered the math behind them and also a basic implementation of one. By understanding the fundamental principles and coding an MLP from scratch, you gain deeper insights into how neural networks function and their potential applications. With this knowledge, you are better equipped to explore more advanced architectures and tackle complex AI challenges.
