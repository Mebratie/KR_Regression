# Machine Learning Conservation Laws of Dynamical Systems

## Introduction  

This repository contains code for learning the conservation laws of dynamical systems using Kernel Ridge Regression (KRR).
 Below are the instructions to get started with the code.

## Instructions

### Getting Started

1. Clone the Repository:
   * First, download the zipped files of the GitHub repository.
   *  Unzip the files
2. Directory Structure:
    * The repository contains the following directories:
       - **System**: Contains the dynamical systems to generate data
       - **scripts**: Contains the Jupyter notebook to run the code (to work with KRR) and display the results.
       - **results**: Contains the results of the code execution, including figures and other important results.
       - **Data**: Contains the generated data.
       - **src/kernelCL**: Includes the standalone ``kcl.py`` Python code for ease of use.

### Data Generation

* Data are generated using the Jupyter notebook located inside the system folder.
* You can generate data for your own dynamical system or use the provided dynamics within the system folder to check our results.
* The generated data are saved inside the data folder

### Running the Code

1. **Dependencies**:
   * The code requires the following common packages:
       - scipy, numpy, sklearn, matplotlib, os, sympy, itertools, pandas
2. **Quick Demonstration**:
   
We really recommend starting with our example Lorenz system to understand how to discover single and multiple conservation laws, as well as the sparsification procedure.

   * For example, run ```Lorenz_System_T5_40.py``` in the terminal located in the scripts folder. In this script, the user can specify the polynomial kernel degree and the constant c. For simplicity, you can set, but any value for c can be used.
   *  The illustrative figures, conserved quantity, and other important results will be saved to the results folder.

   To explore more comprehensive examples, utilize the ```kcl.py``` script. This general-purpose code is designed to be executed 
   for various systems or example scenarios. For each specific system or example, run the ```kcl.py```

   *  To check the previous version of the algorithm, one can go to version v1.3, download the code, and experiment with it.
