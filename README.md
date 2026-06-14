## Loss Function Class
Represents the loss function used to measure error between prediction and target.

**Attributes:**
- name → name of the loss function

**Functions:**
- compute_loss(y_pred, y_true) → returns raw loss value for a single sample
- derivative(y_pred, y_true) → returns raw gradient (dL/dy_pred)
- lossvec(y_pred, y_true) → returns mean loss over a vector
- gradvec(y_pred, y_true) → returns gradient vector

**Mathematical Formulas:**
- **MSE (Mean Squared Error):** 
  - Loss L = (y_pred - y_true)²
  - Derivative dL/dy_pred = 2 * (y_pred - y_true)
- **MAE (Mean Absolute Error):**
  - Loss L = |y_pred - y_true|
  - Derivative dL/dy_pred = 1 if y_pred > y_true else -1
- **Binary Cross Entropy:**
  - Loss L = -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
  - Derivative dL/dy_pred = (y_pred - y_true) / (y_pred * (1 - y_pred))

**Notes:**
- No need for separate gradient for weights or bias in this class
- Loss only provides gradient at the output layer


## Activation Function Class
Represents activation applied to layer outputs to introduce non-linearity.

**Attributes:**
- name → name of activation function

**Functions:**
- activate(x) → applies activation function
- grad(x) → returns derivative of activation function


## Layer Class
Represents a fully connected (dense) layer.

**Attributes:**
- weights → matrix (output_size × input_size)
- bias → vector (output_size)
- input → stores input vector (used in backprop)
- output → stores activated output
- activation → activation function object

**Functions:**

### Forward
- Z = W * X + b
- A = activation(Z)
- returns output

### Backward
- delta = grad_output * activation_derivative
- dW = delta ⊗ input
- db = delta
- W = W - learning_rate * dW
- b = b - learning_rate * db
- call the optimiser to update the weight and gradient
- grad_input = Wᵀ * delta

**Notes:**
- Gradients are computed during backward pass only

## optimizer

**Attributes:**
- learning rate
- beeta and epsilion 
- it contain velocity and momentum variables

**Function**
- update( take gradient from the layer and update on teh basis of momentum and velocity)



**NOTEs**

- in some optimizer there is not need of velocity and moementum in that we define the object in such a way that there is no need of using velocity and momentun and this is uninitialised.





## Sequential Network Class
Represents the neural network as a sequence of layers.

**Attributes:**
- layers → list of layer objects
- loss → loss function object
- epochs → number of training iterations
- learning rate 
- optimizer

**Functions:**
- add(layer)
- forward(input)
- backward(y_pred, y_true, learning_rate)(in this we have to backward propogate and update using optimiser and )
- fit(x_train, y_train)


## Perceptron Class (Not Recommended)
Avoid perceptron-level design.

**Reasons:**
- inefficient (one object per neuron)
- difficult to scale
- not suitable for matrix operations
- not GPU-friendly

**Correct approach:**
- use matrix-based layer design instead


## Key Design Principles
- use matrix operations instead of individual neurons
- compute gradients during backpropagation
- keep components modular (loss, layer, activation, optimizer decoupled)
- design system for future GPU support


## Summary
- loss → provides gradient at output
- layer → handles forward, backward (calculates gradients)
- activation → non-linearity and derivative
- optimizer → updates weights based on gradients
- sequential → manages training and batches
