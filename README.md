# Machine Learning Conservation Laws of Dynamical Systems

## Introduction  

This repository contains code to learn conservation laws of dynamical systems
using KRR. Below are the instructions to get started with the code.

## Instructions

### Getting Started

1. Clone the Repository:
   * First, download the zipped files of the GitHub repository.
   *  Unzip the files and ensure that you have the correct directory structure on your computer
2. Directory Structure:
    * The repository contains the following directories:
       - **System**: Contains the dynamical systems to generate data
       - **scripts**: Contains the Jupyter notebook to run the code (to work with KRR) and display the results.
       - **results**: Contains the results of the code execution, including figures and other important results.
       - **Data**: Contains the generated data.

### Data Generation

* Data are generated using the Jupyter notebook located inside the system folder.
* You can generate data for your own dynamical system or use the provided dynamics within the system folder to check our results.
* The generated data are saved inside the data folder

### Running the Code

1. **Dependencies**:
   * The code requires the following common packages:
       - scipy, numpy, sklearn, matplotlib, os, sympy, itertools, pandas
2. **Quick Demonstration**:
   
Learn a conservation law for the 4D Lotka-Volterra system using our method by following these steps

   *  Run the Jupyter notebook ```4D_Lotka_T5_N20.ipynb``` located inside the system folder, where ```T5``` represents 5 trajectories, and ```N20``` represents 20 data points per trajectory, resulting in a total of 100 data points.
   *  The generated data will be automatically saved to the data folder, more specifically to the folder 4D_Lotka_Volterra.
   *  Next, run the Jupyter notebook ```4D_Lotka_T5_N20.ipynb``` located inside the scripts folder.
   *  The illustrative figures, conserved quantity, and other important results will be saved to the results folder.
