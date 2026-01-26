# Lecture 1

- Text to image generation:
  - Prompt hero 
  - Gemini (nano banana)
- Underlying AI model of ChatGPT: transformer
- How to train the ChatGPT model: Next-token prediction
- Computational bio = creation of tool. More about engineering.
- Bioinfo = study of bio. More about discovery. 
- Types of biological data
  - Sequence data (protein/DNA)
  - High-dimensional data (gene expression, biomedical images)
  - Structure data (protein/small-molecule)
  - Network data (graph)

# ML foundations  

## Lecture 3: Regression & Gradient descent

- gradient descent to minimize MSE for parameter optimization
- hyperparameter tuning - help determine step size
- Normal equations; Ordinary Least Squares
- conclusion
  - linear regression
  - loss function
  - optimization
    - gradient descent
    - Closed-form

## Lecture 4: Classification & ML Toolbox

- Sigmoid/logistic function - binary classification algorithm 
  - output always between 0 and 1
  - probabilistic interpretation (f0(x) gives the probability that x belogs to class y=1)
- minimize loss function for gradient descent - cross-entropy loss
- logistic regression in scikit learn
- predict y=1, if f(x) > threshold otherwise y=0
- softmax regression - multi-class classification
- other models
  - support vector machine
    - select boundaries with the maximum margin - we are as confiident as possible for every point and far from the  boundary  
  - decision trees
  - random forest - combines many decision trees
  - neaural networks
- "toolbox"
  - model evaluation
    - training/test set
    - k-fold cross-validation
    - accuracy, confusion matrix
  - feature engineering
