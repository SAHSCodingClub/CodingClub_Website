<div style="text-align: center; font-weight: 700; font-size: 3rem;">
Analysis of multilayer perceptrons
</div>

*By Adam Jacuch | December 17, 2024*

## Introduction

Multilayer perceptrons (MLPs) have played a pivotal role in the many breakthroughs seen by modern artificial intelligence (AI). Such neural networks, characterized by fully interconnected layers of neurons, are used anywhere from image recognition to natural language processing. This article aims to uncover the elegant math and concepts that create such powerful tools.


<img src="https://cdn-images-1.medium.com/v2/resize:fit:800/0*eaw1POHESc--l5yR.png" alt="image of multilayer perceptron" style="width: 60vw; height: auto;"/>


For starters, MLPs take numerical inputs. For instance, you can pass images as RGB or greyscale values. Inputs are in the form of an array of floating point digits. When you pass such data to an MLP, the input gets assigned to a correspoding neuron in the input layer (the size of the input layer must equal the size of the input data).

$$ \vec{inputData} = \vec{inputLayer} $$

Once the input layer contains the input data, we can begin the forward pass of the network. Forawrd pass refers to the data being passed forwards through layers to some arbitrary output layer. To begin, you first iterate through all the input layers and multiply them by a weight (this will be learned by the MLP). You then add this value into a respective neuron in the next layer. In MLPs, each neuron in a preceding layer has weighted connections to every neuron in the following layer. Therefore, we can represent a neurons value as:

$$ \sum_{n=0}^s \vec{neurons} \times \vec{weights} $$

Where s represents the previous layer size. You can simplify this further to:

$$ \vec{neurons} \cdot \vec{weights} $$

In hidden layers, once this operation is complete a bias (this is learnable by the MLP) is added and an activation function is applied. Biases allow the MLP to shift the activation function and better capture complex patterns in data. Similarly, activation functions introduce non-linearity into the network, enhasing the power of biases. Common examples of activation functions include sigmoid, hyperbolic tangent, and rectified liear units.

<img src="https://www.researchgate.net/publication/376193024/figure/fig3/AS:11431281209315146@1701720609015/Activation-Functions-ReLU-tanh-sigmoid.jpg" alt="Activation Functions" style="max-width: 60vw; height: auto;"/>

The math behind these operations looks like this:

$$ \sum_{n=0}^s a(value + bias) $$

Once the data reaches the output layer it would idealy contain some logical interpretation of the data. However, a randomly initalized MLP will return random values. This is where the backwards pass comes into place. The purpose of the backwards pass is to adjust the models weights in a way that minimizes loss, or the difference between what the AI predicts and the expected output. The expected output is user-defined, and should be present in the data that you feed the network. The loss function is typically a very high dimensional function (the exact number depends of the number of weights in a model). If a model has five weights the loss function will look like:

$$ l(a, b, c, d, e) $$

to minimize the loss, the caluate the partial derivatives of a change in a weight with respect to change in loss

$$ \frac{\partial w}{\partial l} $$

This is equivalent to caluclating the gradient of a function, shown below.

<img src="https://aleksandarhaber.com/wp-content/uploads/2023/05/gradient1-1.png" alt="Function Gradients" style="max-width: 60vw; height: auto;"/>

A gradient at a single point represents a vector pointing in the direction of steepest ascent. To minimized loss we move in a small direction opposite to the gradient. This small value is known as a learning rate, or how much the AI adjusts itself per iteration. This hyperparameter (a parameter you define yourself) value is crucial to an MLPs performance—set it too high and it over-adjusts, too low and it takes too mamy iterations (and thus computing power) to learn sufficiently. Learning rates are typically around 0.01. The models weght adjustment can be represented by:

$$ \Delta \vec{w} = -lr \times \vec{gradient} $$

Because the gradient is all of the change in weights with respect to loss, we can modify each weight by its respective partial derivative (times a small negative lerning rate). This will move the loss function in th direction of steepest descent. Repeat this process until the loss is acceptably low, or if it gets stuck at a local minimum (a place where going in any direction within a near vicinity increases loss).

## Implementation

Here is a simple C++ program:

```cpp
#include <iostream>
using namespace std;

int main() {
    cout << "Hello, World!" << endl;
    return 0;
}
```
