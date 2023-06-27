# 🦾 Regressify
Regressify is a multi-task neural network coded from scratch that performs simultaneous regression and classification. 

## 🛠️ Tools Used
<img src='https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54'> <img src='https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white'> <img src='https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white'> <img src='https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black'> <img src='https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252'> 

<details>
  <summary><h2> 📑 Data </h2></summary>
The neural network has been trained on a dummy dataset of 2 features F1 and F2, and 2 target variables T1 and T2, corresponding to a class label and a regression value respectively. Here is a sample subset of the same:

| F1 | F2 | T1 (class label) | T2 (regression value) |
|---|---|---|---|
| 1.4 | 0.2 | 1 | 0.28 |
| 1.6 | 0.2 | 1 | 0.32 | 
| 4 | 1.2 | 2 | 4.8 | 
| 3.3 | 1 | 2 | 3.3 |
| ... | ... | ... | ... |

- There are **150 samples** in the dataset
- Each sample belongs to one of **2 classes** where the class ID is either 1 or 2
</details>

<details>
<summary><h2>🏗️ Neural Network Architecture</h2></summary>
  
### Input Layer (Layer 0)
Consists of two neurons, each corresponding to one of the two feature variables, F1 and F2. 
### Hidden Layers
- Controlled by the `num_layers` hyperparameter. Note: `num_layers` includes the last (output) layer. Thus, the number of hidden layers is `num_layers - 1`
- Each hidden layer can have either a `sigmoid` or a `tanh` activation function applied to it. Hyperparameter: `layer_activation_fns`
- `num_neurons` controls the number of neurons in each layer
### Output Layer
- Has two neurons, one for a classification prediction, and the other for a regression value
- The matrix output by the penultimate layer undergoes:
  + a `sigmoid` function to obtain the respective probabilities of each sample belonging to one of the two classes
  + a `linear` function to obtain regression values corresponding to each sample
</details>

<details>
<summary><h2>♾️ Math</h2></summary>
  
### Forward Propagation
Used to product an output in the forward direction by sequentially processing the input data through each layer of the neural network. 

This processing involves mutliplying each layer's input with its corresponding weight matrix and then passing the product to an activation function

For each layer L, 

$$Z^{[L]} = W^{[L]}A^{[L-1]} + b^{[L]}$$

$$A^{[L]} = g^{[L]}(Z^{[L]})$$

where 

$W^{[L]} =$ Weight matrix of layer L

$b^{[L]} =$ Bias vector of layer L

$A^{[L]} =$ Output matrix of layer L $=$ Input to layer L+1

$g^{[L]} =$ Activation function of layer L

### Backward Propagation
Involves computation of losses in the backward direction which in turn allows for changes in layer weights to make the network produce more accurate outputs.

For each layer L,

$$\frac{\partial C}{\partial Z^{[L]}} = W^{[L+1]^T} \frac{\partial C}{\partial Z^{[L+1]}} \odot g'^{[L]}(Z^{[L]})$$

$$\frac{\partial C}{\partial W^{[L]}} = \frac{\partial C}{\partial Z^{[L]}}A^{[L-1]^T}$$

$$\frac{\partial C}{\partial b^{[L]}} = \frac{\partial C}{\partial Z^{[L]}}$$

$$W^{[L]} = W^{[L]} - \alpha \frac{\partial C}{\partial W^{[L]}}$$

$$b^{[L]} = b^{[L]} - \alpha \frac{\partial C}{\partial b^{[L]}}$$

where 

$C =$ Cost/loss computed by neural network 

$\odot =$ Element-wise multiplication

$\alpha =$ Learning rate

Note for the last layer:

$$\frac{\partial C}{\partial Z^{[Last]}} = \frac{\partial C}{\partial A^{[Last]}} \odot g'^{[Last]}(Z^{[Last]})$$ 
</details>

<details>
<summary> <h2>📝 Performance Analysis </h2></summary>

### Training
Classification accuracy: `0.975`

Regression $R^2$ score: `0.9712041992163881`

### Validation
Classification accuracy: `1.0`

Regression $R^2$ score: `0.723742056420074`

</details>
