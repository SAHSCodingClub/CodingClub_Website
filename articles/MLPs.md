<div style="text-align: center; font-weight: 700; font-size: 3rem;">
Analysis of multilayer perceptrons
</div>

*By Adam Jacuch | December 17, 2024*

## Introduction

Multilayer perceptrons (MLPs) have played a pivotal role in the many breakthroughs seen by modern artificial intelligence (AI). Such neural networks, characterized by fully interconnected layers of neurons, are used anywhere from image recognition to natural language processing. This article aims to uncover the elegant math and concepts that create such powerful tools.

<p align="center">
  <img src="https://cdn-images-1.medium.com/v2/resize:fit:800/0*eaw1POHESc--l5yR.png" alt="image of multilayer perceptron">
</p>

For starters, MLPs take numerical inputs. For instance, you can pass images as RGB or greyscale values. Inputs are in the form of an array of floating point digits. When you pass such data to an MLP, the input gets assigned to a correspoding neuron in the input layer (the size of the input layer must equal the size of the input data).

$$ \vec{inputData} = \vec{inputLayer} $$

Once the input layer contains the input data, we can begin the forward pass of the network. Forawrd pass refers to the data being passed forwards through layers to some arbitrary output layer. To begin, you first iterate through all the input layers and multiply them by a weight (this will be learned by the MLP). You then add this value into a respective neuron in the next layer. In MLPs, each neuron in a preceding layer has weighted connections to every neuron in the following layer. Therefore, we can represent a neurons value as:

$$ \sum_{n=0}^s \vec{neurons} \times \vec{weights} $$

Where s represents the previous layer size. You can simplify this further to:

$$ \vec{neurons} \cdot \vec{weights} $$

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
