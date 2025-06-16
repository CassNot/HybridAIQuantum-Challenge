# Welcome to the First Perceval Quest!

<table align="center" bgcolor="black">
<tr>
  <td bgcolor="black" align="center" width="200"><img src="./logos/quandela-logo-blackbg.png" width="200" alt="Quandela Logo"></td>
  <td valign="middle" style="font-size: 24px; font-weight: bold; padding: 0 20px;">×</td>
  <td bgcolor="black" align="center" width="200"><img src="./logos/scaleway-logo-blackbg.svg" width="200" alt="Scaleway Logo"></td>
</tr>
</table>

<div align="center">
  <img width="48%" alt="Challenge-img" src="Challenge.png">
  <p><em>Image credit: The valiant knight Dall-E</em></p>
</div>

### :zap: Presentation of the Challenge

The First Perceval Quest was jointly organized by Quandela and Scaleway to explore the intersection of quantum computing and machine learning through one of the most iconic machine learning benchmarks - the MNIST dataset.

The challenge was to tackle the well-known MNIST problem using a hybrid quantum model on a subset of the original dataset. The MNIST dataset consists of 70,000 handwritten digit images, each 28x28 pixels. For this quest, you'll work with a reduced dataset of 6,000 images and use a quantum kernel to predict the digits.

:calendar: This challenge lasted from November 2024 to March 2025. We are now happy to share the top submissions we received !

## Historical Context & Challenge Overview

The MNIST (Modified National Institute of Standards and Technology) dataset was introduced by Yann LeCun et al. in 1994 and has served as a fundamental benchmark in the machine learning community for almost 30 years. This collection of handwritten digits has been instrumental in testing and validating numerous computer vision approaches, from traditional machine learning to deep neural networks.

While modern classical methods have achieved near-perfect accuracy on MNIST, our challenge takes a different approach. We're revisiting this iconic benchmark through the lens of quantum machine learning, not with the goal of surpassing classical accuracy records, but to explore novel quantum techniques and methodologies. To make the challenge more suitable for quantum processing, we're working with a reduced dataset of 6,000 images instead of the original 70,000, adding an interesting constraint that makes the problem more challenging and relevant for quantum approaches.

### :rocket: The results

Overall, 64 teams joined the Perceval Quest and 11 were selected for the second phase. 
The final jouxt opposed 11 hybrid quantum-classical models that are presented in [src](./src). 
You will also find a detailed description for each model in the associated [ReadMe](./src/README.md)

## :arrow_forward: To run the solutions
For you to run the proposed solutions, we suggest you to create your own python environment:
 Create a virtual environment:
   ```bash
   python -m venv MerLin-env
   source MerLin-env/bin/activate
   pip install -r requirements.txt
   ```
This will download the MerLin framework ([documentation](https://merlinquantum.ai/index.html)) for easier and faster photonic QML implementation !

## :bulb: Photonic Quantum Computing at Quandela

This challenge leverages photonic quantum computing, a promising quantum computing paradigm that uses light particles (photons) as quantum bits. Participants will use the Perceval framework, an open-source platform developed by Quandela for programming photonic quantum computers. You can learn more about Perceval and its capabilities at [perceval.quandela.net](https://perceval.quandela.net).

### Quantum Computing Resources

Participants could develop small scale algorithms using local simulation and in phase 2, they were able to access to [Scaleway's Quantum-as-a-Service platform](https://labs.scaleway.com/en/qaas/), which provided both large-scale quantum simulators and actual QPU access. This platform enabled participants to test and run their quantum algorithms in both simulated and real quantum environments.

### Organization of the repository
The dataset is located in the `data` folder, containing `train.csv` and `test.csv` files. 
The notebook `MNIST_classification_quantum.ipynb` and its equivalent script, `training.py`, contain the training loop used for model training.

An example code for building quantum embeddings and integrating them into a basic classical model is split in separate scripts: the model is defined in `model.py`, the Boson Sampler in `boson_sampler.py` and some helper functions (dataset class for the reduced dataset, accuracy function...) can be found in `utils.py`. 

### Challenge Rules

Use any classical machine learning model and demonstrate improved performance with a quantum model (see Evaluation Criteria).
Submit your solution as a reproducible Jupyter notebook.
Modify the provided quantum model as needed. It can rely on quantum kernels or other methods. 


### :inbox_tray: Contact & Support

Email perceval-challenge@quandela.com with your team description (individual or group entries welcome). You'll receive confirmation and submission instructions.

For any general questions, please use Perceval Forum at https://perceval.quandela.net/forum/ with the tag `Perceval Quest`.
For technical questions, please use the GitHub Discussions tab in this repository.
