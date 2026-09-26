# 🧠 Artificial Neural Network (ANN) – Student Study Hours Prediction

A beginner-friendly **Artificial Neural Network (ANN) regression project built with PyTorch** to predict a student's study hours based on their sleep hours.

This project demonstrates the basic workflow of building and training a neural network using **PyTorch**, including data loading, tensor conversion, model creation, loss calculation, optimization, training, and prediction.

---

## 📌 Project Overview

The project uses a student dataset containing information about students' sleep and study hours.

The model takes:

* **Input:** `sleep_hours`
* **Output / Target:** `study_hours`

A simple neural network is trained to learn the relationship between these two variables.

The model architecture consists of a single linear layer:

```text
Sleep Hours
     │
     ▼
┌─────────────┐
│ Linear(1,1) │
└─────────────┘
     │
     ▼
Study Hours
```

The project is primarily intended for learning the fundamentals of **neural networks and regression using PyTorch**.

---

## 🚀 Features

* Load student data using Pandas
* Convert dataset columns into PyTorch tensors
* Build a simple neural network using `nn.Linear`
* Use Mean Squared Error (MSE) as the loss function
* Train the model using Stochastic Gradient Descent (SGD)
* Perform prediction on new input data
* Access the trained model's weight and bias

---

## 🛠️ Technologies Used

| Technology   | Purpose                                 |
| ------------ | --------------------------------------- |
| Python       | Programming language                    |
| Pandas       | Dataset loading and manipulation        |
| PyTorch      | Neural network development and training |
| Google Colab | Development environment                 |

---

## 📂 Project Structure

```text
ANN-Student-Study-Prediction/
│
├── ann.py
├── student_dataset_10000_rows.csv
└── README.md
```

> Make sure the CSV dataset is available in the same directory when running the Python script.

---

## 📊 Dataset

The project uses:

```text
student_dataset_10000_rows.csv
```

The relevant columns used by the model are:

| Column        | Description                         |
| ------------- | ----------------------------------- |
| `sleep_hours` | Number of hours the student sleeps  |
| `study_hours` | Number of hours the student studies |

The dataset contains **10,000 rows** according to the project setup.

---

## 🧩 How the Model Works

### 1. Load the Dataset

The dataset is loaded using Pandas:

```python
data = pd.read_csv("student_dataset_10000_rows.csv")
```

---

### 2. Prepare Input and Output

The `sleep_hours` column is used as the input feature, while `study_hours` is used as the target:

```python
ip = torch.tensor(
    data['sleep_hours'].values,
    dtype=torch.float32
).reshape(-1, 1)

op = torch.tensor(
    data['study_hours'].values,
    dtype=torch.float32
).reshape(-1, 1)
```

This converts the Pandas data into PyTorch tensors suitable for training.

---

### 3. Create the Neural Network

A simple linear neural network is created:

```python
model = nn.Linear(1, 1)
```

The model contains:

* 1 input feature
* 1 output value
* Trainable weight
* Trainable bias

Mathematically, the model learns:

```text
y = wx + b
```

where:

* `x` = sleep hours
* `y` = predicted study hours
* `w` = learned weight
* `b` = learned bias

---

### 4. Define the Loss Function

The project uses **Mean Squared Error (MSE)**:

```python
error = nn.MSELoss()
```

MSE measures the difference between the predicted study hours and the actual study hours.

---

### 5. Configure the Optimizer

The model uses **Stochastic Gradient Descent (SGD)**:

```python
opt = torch.optim.SGD(
    model.parameters(),
    lr=0.001
)
```

The learning rate is:

```text
0.001
```

---

### 6. Train the Model

The model is trained for **1,000 iterations**:

```python
for i in range(1000):
    pred = model(ip)
    loss = error(pred, op)

    opt.zero_grad()
    loss.backward()
    opt.step()
```

During training:

1. The model generates predictions.
2. MSE calculates the error.
3. Existing gradients are cleared.
4. Backpropagation calculates gradients.
5. SGD updates the model parameters.

---

## 🔮 Prediction

After training, the model is used to predict study hours for a student who sleeps for **3 hours**:

```python
model(torch.tensor([[3.0]]))
```

The trained model's parameters can also be inspected using:

```python
model.weight
model.bias
```

---

## 📈 Learning Outcomes

By completing this project, you can understand:

* What an ANN is
* How neural network models are created in PyTorch
* How input data is converted into tensors
* How a regression model works
* Mean Squared Error loss
* Gradient descent
* Backpropagation
* Model parameter optimization
* Weight and bias in a neural network
* Making predictions with a trained PyTorch model

---

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/ANN-Student-Study-Prediction.git
```

### 2. Navigate to the Project

```bash
cd ANN-Student-Study-Prediction
```

### 3. Install Dependencies

```bash
pip install pandas torch
```

### 4. Add the Dataset

Place:

```text
student_dataset_10000_rows.csv
```

inside the project directory.

### 5. Run the Python File

```bash
python ann.py
```

---

## ☁️ Google Colab

The original implementation was created in **Google Colab**.

You can also upload the Python notebook/script and dataset to Google Colab and run the cells there.

---

## 🔬 Model Configuration

| Parameter           | Value            |
| ------------------- | ---------------- |
| Input Features      | 1                |
| Output Features     | 1                |
| Model               | `nn.Linear(1,1)` |
| Loss Function       | MSE Loss         |
| Optimizer           | SGD              |
| Learning Rate       | 0.001            |
| Training Iterations | 1000             |
| Input Feature       | `sleep_hours`    |
| Target              | `study_hours`    |

---

## ⚠️ Limitations

This is a **basic learning project** rather than a production-ready machine learning system.

Current implementation:

* Uses only one input feature
* Uses a single linear layer
* Does not perform train/test splitting
* Does not include feature normalization
* Does not calculate evaluation metrics such as MAE or R²
* Does not include visualization of training loss
* Does not save the trained model

Because the model is a single linear layer, this implementation is essentially demonstrating **linear regression using PyTorch's neural-network API** rather than a deep multi-layer ANN.

---

## 🚀 Possible Future Improvements

The project can be extended by:

* Adding more student features
* Splitting data into training and testing sets
* Normalizing input features
* Adding hidden layers
* Using activation functions such as ReLU
* Tracking training loss
* Plotting the loss curve
* Evaluating the model using MAE, MSE, and R²
* Saving and loading the trained model
* Building a simple prediction interface
* Deploying the model as a web application

---

## 🎯 Project Goal

The main goal of this project is to understand the fundamental workflow of training a neural-network-based regression model with PyTorch.

```text
Dataset
   ↓
Data Preprocessing
   ↓
PyTorch Tensors
   ↓
Neural Network
   ↓
Loss Calculation
   ↓
Backpropagation
   ↓
SGD Optimization
   ↓
Trained Model
   ↓
Prediction
```

---
