# Robust GNN: Coordinated Cyber Attack Detection in Cloud Environments

## 📖 What is this System?
This repository contains a state-of-the-art machine learning framework designed to detect coordinated cyber attacks (such as Distributed Denial-of-Service or Botnets) within distributed cloud environments. 

Traditional firewalls analyze single network packets in isolation, which makes them highly vulnerable to coordinated attacks that slip past the perimeter. This system solves that problem by treating the cloud infrastructure as a mathematical **Graph**. By using Graph Neural Networks (GNNs), the system analyzes the *relationships* and *topological structure* of internal cloud traffic to catch sophisticated attackers trying to evade detection.

---

## 🚀 How to Run the System (Interactive UI)
**IMPORTANT:** This system should **ONLY** be run using the interactive Web Dashboard (Streamlit). 

While headless CLI scripts are available in the codebase, the Web UI is the intended and primary way to interact with the project. The UI is significantly more interactive and allows the user to know exactly what is being done at all times. From the dashboard, you can:
*   Dynamically upload cloud network datasets (CSVs).
*   Visually tweak deep learning parameters (k-Nearest Neighbors, Epochs, Learning Rates).
*   Watch the Graph Neural Network learn in real-time via live training curves.
*   Compare the GNN's performance against traditional Machine Learning baselines (Random Forest, Logistic Regression) with a single click.

---

## ⚙️ Installation & Requirements

To run this system, you need Python installed on your computer.

### Step 1: Open your Terminal / Command Prompt
Navigate to the project folder (`gnn-cyber-detection`).

### Step 2: Create and Activate a Virtual Environment
It is highly recommended to isolate the project's dependencies.

**For Mac & Linux:**
```bash
python3 -m venv gnn-env
source gnn-env/bin/activate
```

**For Windows:**
```cmd
python -m venv gnn-env
gnn-env\Scripts\activate
```

### Step 3: Install the Requirements
With your virtual environment activated, install the required libraries (PyTorch, PyTorch Geometric, Streamlit, Scikit-learn, etc.):
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application!
Once everything is installed, launch the interactive web dashboard:
```bash
streamlit run streamlit_app.py
```
*(Your web browser will automatically open to `http://localhost:8501`)*

---

## 📂 Structure of the Code (Modules Explained)
The codebase is highly modular, ensuring strict separation between data processing, neural network training, and the user interface.

*   **`streamlit_app.py`:** The main entry point. It creates the Web UI, collects the user's hyper-parameters from the sidebar, and dynamically draws the charts and metrics on the screen.
*   **`src/pipeline.py`:** The orchestrator. The `run_experiment()` function takes the user's settings from the UI, asks the data loader to fetch the data, asks the graph builder to map it, trains the model, and finally saves the learned model weights to the `outputs/` folder.
*   **`src/data_loader.py`:** Contains the `TrafficDataLoader`. It reads the raw network CSV files, cleans the data, infers the statistical features, and handles the Train/Validation/Test splits.
*   **`src/graph_builder.py`:** The topological engine. It contains `FlowKnnGraphBuilder` (which connects similar network flows mathematically) and `TrafficGraphBuilder` (which connects explicit IP addresses if they are available).
*   **`src/train.py`:** The deep learning math. It handles the PyTorch Geometric message-passing algorithms, trains the Graph Convolutional Network (GCN) layer by layer, and evaluates the final exam (test set).
*   **`src/utils.py`:** Computes the final academic grades (Accuracy, Precision, Recall, F1-Score) using Scikit-Learn.
*   **`scripts/run_baselines.py`:** Contains standard "flat" Machine Learning models (Random Forest, Logistic Regression). It allows the system to prove the superiority of the Graph approach by running direct comparisons.
*   **`scripts/test_robustness.py`:** A simulation suite. It artificially acts as an attacker by dropping communication links (Edge Dropout) or spoofing traffic (Feature Noise) to prove the GNN is highly resilient.

---

## ✨ Core Features & Capabilities

*   **Topological Threat Detection:** Utilizes PyTorch Geometric to build k-Nearest Neighbor (kNN) flow graphs, mapping the structural relationships of network traffic rather than analyzing flows in isolation.
*   **Dual-Graph Modes:** Dynamically constructs interaction graphs based on raw IP routing data (Hosts Mode) or multi-dimensional feature similarity (Flows Mode).
*   **Adversarial Robustness:** Built-in testing suite to simulate network degradation (Edge Dropout) and payload obfuscation (Feature Noise), proving the model's resilience in chaotic environments.
*   **Interactive Web Dashboard:** Includes a fully featured Streamlit UI for dynamic parameter tuning, live training visualization, and baseline model comparison.
*   **Baseline Benchmarking:** Automatically trains and compares standard Machine Learning models (Random Forest, Logistic Regression) against the GNN architecture.

---

## 📊 Experimental Results

Tested on the **CIC-IDS2017** dataset (specifically isolating coordinated DDoS flows against benign background traffic).

| Model Architecture | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Traditional Random Forest** | 99.97% | 0.9997 | 0.9997 | 0.9997 |
| **Proposed GNN (Graph Conv)** | 99.51% | 0.9951 | 0.9951 | 0.9951 |

### 🛡️ Adversarial Robustness
While traditional flat models perform slightly better on pristine, unperturbed datasets, the Graph Neural Network exhibits **extreme structural resilience** against evasion tactics:
*   **80% Edge Dropout:** Even when 80% of communication links are artificially dropped to simulate packet loss or evasive routing, the GNN maintains a **99.68% F1-Score**.
*   **Severe Feature Noise:** When Gaussian noise (Scale: 2.0) is injected into the flow features to simulate payload obfuscation, the GNN maintains a **99.55% F1-Score**.

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute. 

## 📝 License
This project is open-source and available under the [MIT License](LICENSE).
