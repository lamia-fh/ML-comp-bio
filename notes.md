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
## Lecture 5: Applied Toolbox & Neural Network I
- Reciever Operating Characteristic (ROC)
  - TPR = True positive rate
  - FPR
- Small t —> high TPR 
- High t —> low TPR
- Area under the ROC curve (AUC) —> single measure of classifier performance
- FI score
- precision-recall curve
- regression metrics
  - mean squared error
  - absolute (L1) error
  - correlation efficiency 
- 1- model evaluation
### Feature engineering 
- polynomial regression
- overfitting with too high order polynomial
- training examples to fit model
- test/data samples to evaluate 
### model selection
- hyperparameter tuning
  - train multiple models with different values of p & evalute on the validation set (never on test set)
## Lecture 6: Neural Network II 
- Design Space of Neural networks 
  - eg # of hidden layers (depth)
  - num of units in each layer (width)
- Activation function 
  - in bio neurons: if input signals are strong enough, neuron fires an output 
  - rectified linear unit (ReLU) —> used in modern deep neural networks 
- Loss functions & output layers
- Objective functions
  - Regression loss
  - (binary) classification loss 
- Optimize NN 
  - Calculus: Chain Rule
  - backpropagation - repeated application of the chain rule
  - backpropagation of errors
  - backpropagation in logistic regression
- Stochastic gradient descent — reduces computational cost
- Mini-batch stomachastic gradient descent 
  - a compromise between GD and SGD
# Learning from high-dim data
## Lecture 8: Unsupervised Learning (Clustering & Dimenstionality Reduction)
- Supervised learning: learn a function
- Unsupervised learning: can we infer the underlying structure of X?
- Why unsupervised?
  - Raw data cheap. Labelled data expensive
### Clustering
- eg single-cell expression analysis
- K-means clustering
  - Initialize: Pick K random points as cluster centers
  - Repeat
    - Assign data points to closest cluster center
    - Change the cluster center to the average of its assigned points
  - Stop when no points’ assignments change
  - K-means convergence 
- Hierarchial clustering
  - Agglomerative clustering --> produces dendogram
  - Cluster distance to find most similar clusters
### Dimenstionality reduction 
- High-dimension —> curse of dimensionality 
- approaches
  - Linear transformation
    - PCA
    - NMF
  - Non-linear transformation
    - Autoenconder, VAE
    - MDS
    - tSNE
    - UMAP
- PCA
  - Goal: Find a projection of the data onto directions that maximize variance of the original data 
  - Intuition: those are directions in wihch most information is encoded
- Singular value decomposition (SVD) - find eigenvalues/eigenvectors

## Lecture 9: Important Ideas in GenAI
- Video generation: Sora (OpenAI)
- data distribution of image (eg probability of an image in a certain category)
- Autoregressive models - likelihood-based
  - eg ChatGPT
  - eg gradient-based optimization
- Latent variable models
- Variational autoencoder (VAE) - approximate density
- generative adversarial network (GAN) - implicit density 
- diffusion model 
  - learning to generate by denoising
  - forward diffusion: adds noise
  - reverse denoising: removes noise