# Limited-Flip Othello AI

## Overview

This project implements a variant of Othello (Reversi) called **Limited-Flip Othello**, where at most *k* opponent disks are flipped per direction during a move.

The project includes:
- A full game environment
- A baseline AI using **Minimax with Alpha-Beta pruning**
- Enhancements including:
  - **Transposition Tables (TT)**
  - **Iterative Deepening (ID)**
- Experimental evaluation of performance and gameplay behavior

---

## Requirements

- Python 3.12 (recommended)
- Dependencies:
  - `numpy==2.4.2`

## Setup and Execution Instructions

Follow the steps below to set up and run the project.

### 1. Clone the Repository

```bash
git clone https://github.com/lukebushur/cs572-limited-flip-othello.git
cd cs572-limited-flip-othello
```

### 2. (Optional) Create a virtual environment

It is recommended (but not required) to use a virtual environment to isolate dependencies.

#### On macOS/Linux

```bash
python -m venv venv
source venv/bin/activate
```

#### On Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python main.py
```

Running `main.py` will execute test cases and experiments, printing results such as game outcomes and performance metrics (nodes expanded, cutoffs, and search time).