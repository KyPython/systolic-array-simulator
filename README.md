Systolic Array Matrix Multiplication (Python + Colab)
This project is a small Python simulator for systolic array matrix multiplication, implemented in a Google Colab notebook.

Instead of running matrix multiplication as a long serial loop on a CPU, this notebook models it as a grid of simple processing elements (PEs). Each PE sits in a fixed position, holds a tiny accumulator, and on every “clock cycle”:

Takes one value from the left (from matrix A).

Takes one value from the top (from matrix B).

Multiplies them, adds the result to its local sum, and passes the values along to its neighbors.

What’s inside
ProcessingElement class

Simulates a single PE with:

Registers for A and B inputs.

An accumulator that performs multiply-accumulate (MAC) each cycle.

SystolicArray class

Builds an N×N grid of PEs.

Streams:

Matrix A left → right across the rows.

Matrix B top → bottom down the columns.

Runs for a fixed number of cycles and then reads out each PE’s accumulator as one entry of the result matrix.

standard_matmul function

Reference CPU-style matrix multiplication (triple nested loop) used to verify correctness.

compare_implementations helper

Runs both standard_matmul and SystolicArray.multiply on the same input matrices.

Prints both results side by side.

2×2 “detailed operation” example

A small run with verbose output that shows how each PE’s accumulator changes over cycles, so you can see the result “grow” step by step.

How to run
Open the notebook in Google Colab.

Run cells in order:

Define ProcessingElement and SystolicArray.

Define standard_matmul and compare_implementations.

Run compare_implementations() to compare CPU vs systolic array.

Optionally run the 2×2 verbose example to inspect accumulator states.

No external libraries are required beyond standard Python.

Why this matters
Systolic arrays are a core idea behind modern AI accelerators (like TPUs and specialized AI GPUs). They:

Perform many MAC operations in parallel.

Use predictable dataflow.

Reuse data locally to reduce memory traffic.

This notebook is a small, educational bridge between “I can write matrix multiplication in Python” and “I understand how hardware like a systolic array can accelerate neural network workloads.
