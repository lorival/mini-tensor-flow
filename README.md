[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

[<img src="https://avatars1.githubusercontent.com/u/36938641?s=200&u=b2d470fe66acc157d8ca8cb3fb815dee47d4466d&v=4" align="right" />](https://github.com/machine-learning-experiments)

# Mini tensor flow
This project is a simplified version of TensorFlow, which uses a neural network to predict the price of homes in the Boston area.

### Motivation

Understand differentiable graphs and backpropagation, implementing a neural network from scratch.

### Built With

- [Python 3](https://www.python.org/download/releases/3.0/) - Language
- [Anaconda](https://www.anaconda.com/what-is-anaconda/) - Python Data Science Platform 

### Dataset

The dataset was obtained from the scikit-learn library.

## Getting Started

### Prerequisites
1. Download and install [Anaconda](https://www.anaconda.com/download/)
2. Update Anaconda
> ``` 
> $ conda upgrade conda 
> $ conda upgrade --all 
> ```

### Install

1. Clone and enter into the project's root directory by command line
> ``` 
> $ git clone https://github.com/machine-learning-experiments/mini-tensor-flow.git
> ```
2. Create and activate enviroment
> ``` 
> $ conda env create -f enviroment.yaml 
> $ conda activate mini-tensor-flow
> ```
or
> ``` 
> conda create --name mini-tensor-flow python=3
> source activate mini-tensor-flow
> conda install numpy scikit-learn
> ```
3. Execute neural network for see the loss value tend to zero
> ``` 
> $ python neural_network.py 
> ```

## Author

[Lorival Smolski Chapuis](https://github.com/lorival)
> This project was developed during the [deep-learning](https://br.udacity.com/course/deep-learning-nanodegree-foundation--nd101) nanodegree from [Udacity](https://br.udacity.com/) 
