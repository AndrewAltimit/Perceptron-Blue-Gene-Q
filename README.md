# Overview
Multi-Layer Perceptron implemented on the Blue Gene/Q. See the uploaded PDF for implementation details and experimental results. 

## Contributors
* Andrew Showers
* Dylan Elliott
* Peter Horak

## Key Files
* main.c - used to train and test the network on the MNIST data
* experiments.c - bare bones version of main.c used for scaling experiments
* Makefile - compile options for different experiments
* runs.sh - run experiments on the BG/Q
* kratosruns.sh - run experiments on Kratos

## Additional Files
* data_prep - Python scripts for generating and converting input files into the expected format
* tensorflow_validation - TensorFlow scripts used as a reference to validate the MLP
* results - scaling experiment times in .csv files
* scaling_analysis - Matlab scripts for analyzing the timing results and creating plots

## Usage

	MLP.OUT <file 1> <file 1> <eta> <niter> <ninput> <layer 1> ... <layer l>

The program creates an MLP with l-1 hidden layers and 1 output layer with widths specified by the <layer i> arguments. The MLP is trained for <niter> iterations on the labeled data in <file 1> using backpropagation and gradient descent with a learning rate of <eta>. <ninput> sets the number of input bytes to expect per data sample. The MLP is tested on all the labeled data in <file 2> and its performance is printed to stdout.

	EXP.XL <width>
The program creates an MLP with 1 input, 1 hidden, and 1 output layer each with <width> neurons. The path to the input file is hard-coded but is configured to vary depending on the network width. The program trains the MLP for 2000 iterations while collecting timing data. It prints out a single line with various comma-separated timing results.

## Examples
EXAMPLE 1: run experiments on BG/Q

	make exp_bgq
	sbatch --partition medium --nodes 128 --time 50 runs.sh

EXAMPLE 2: run experiments on Kratos

	make exp_kratos
	./kratosruns.sh

EXAMPLE 3: train and test the MLP on MNIST data on Kratos

	make mlk_kratos
	mpirun -np 10 ./mlp.out mnist_train_bytes mnist_test_bytes 0.01 300000 784 100 100 10
