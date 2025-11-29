# 💻 Systolic Array Matrix Multiplication (Python + Colab)

This project provides a small **Python simulator** for **systolic array matrix multiplication**, implemented as a Google Colab notebook.

Instead of running matrix multiplication as a long serial loop on a CPU, this notebook models it as a grid of simple **processing elements (PEs)**. This architecture is central to modern AI accelerators. 

---

## ✨ The Core Concept: Systolic Array

A systolic array executes matrix multiplication by using a fixed, interconnected grid of PEs.

On every **"clock cycle"**, each PE performs the following actions:
1.  Takes one value from the left (from **Matrix A**).
2.  Takes one value from the top (from **Matrix B**).
3.  Multiplies them, adds the result to its local sum (**Multiply-Accumulate** or **MAC** operation).
4.  Passes the values along to its neighbors (Matrix A value to the right, Matrix B value to the bottom).

This design allows for high **parallelism** and **data reuse**, minimizing memory traffic—key features of hardware accelerators like **Google TPUs**.

---

## 🏗️ What's Inside

The simulator is built around the following components:

| Class/Function | Description | Role |
| :--- | :--- | :--- |
| **`ProcessingElement`** class | Simulates a single PE. | Contains registers for **A** and **B** inputs and an **accumulator** that performs the MAC operation each cycle. |
| **`SystolicArray`** class | Builds an **N×N grid** of PEs. | Streams **Matrix A** left $\rightarrow$ right and **Matrix B** top $\rightarrow$ bottom. Runs for a fixed number of cycles, then reads out each PE's final accumulator value as one entry of the result matrix. |
| **`standard_matmul`** function | Reference implementation. | Performs standard CPU-style matrix multiplication (triple nested loop) for correctness verification. |
| **`compare_implementations`** helper | Verification routine. | Runs both `standard_matmul` and `SystolicArray.multiply` on the same input matrices and prints the results side-by-side. |
| **2×2 “detailed operation” example** | Debugging and visualization. | A small run with verbose output that shows how each PE’s accumulator changes cycle-by-cycle, allowing you to trace the result's accumulation step-by-step. |

---

## ▶️ How to Run

The entire project is contained in a single Colab notebook. **No external libraries are required** beyond standard Python.

1.  **Open the notebook** in Google Colab.
2.  Run the cells in the following order:
    * Define the `ProcessingElement` and `SystolicArray` classes.
    * Define the `standard_matmul` and `compare_implementations` functions.
    * Run `compare_implementations()` to compare the **CPU** result against the **Systolic Array** result.
    * *Optionally*, run the **2×2 verbose example** to inspect the internal accumulator states over time.

---

## 🧠 Why This Matters

Systolic arrays are not just a theoretical concept; they are the foundation for high-performance computing in modern machine learning.

* They perform many MAC operations **in parallel**.
* They use a **predictable dataflow** (the "systolic" rhythm) that is highly efficient for hardware implementation.
* They **reuse data locally** within the PE grid, significantly reducing the bottleneck of retrieving data from off-chip memory.

This notebook serves as an educational bridge between writing basic matrix multiplication code and understanding how specialized **hardware** can fundamentally accelerate demanding neural network workloads.
