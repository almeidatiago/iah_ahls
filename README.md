# Input-Aware Heuristic with High-Level Synthesis of Approximate Hardware

## Abstract

We introduce an input-aware heuristic approach that uses application inputs to model output errors more effectively. In this approach, operators in accelerators, such as adders and multipliers, are mapped to a library of precharacterized approximate components. Applications are simulated with a set of training inputs, and candidate solutions are selected based on a metric that combines output errors and estimated resource utilization.

## Main setup

The necessary components were obtained from the EvoApprox library (version 1.1), which includes 16-bit signed operators. Available at: [https://github.com/ehw-fit/evoapproxlib/tree/v1.0](https://github.com/ehw-fit/evoapproxlib/tree/v1.0).

**Important:** it is required to follow the steps in the EvoApprox repository to install the components used during the search.

The Xilinx Alveo U55C accelerator was employed for resource experiments involving FPGA synthesis ([https://github.com/Xilinx/xacc](https://github.com/Xilinx/xacc)). The HLS procedure utilized Vitis version 2022.2. The image processing applications were simulated with a parallel implementation utilizing OpenMP, where each thread processes a distinct image.

For short, the software requirements are:

- EvoApprox library (version 1.1)
- Jupyter notebook
- Vitis version 2022.2
- OpenMP

## Files

The folder `apps` has the source code for each application, and their respective `Makefile`. Also, all the Jupyter files for each application are basically the same; however, each one calls its specific application during the run. Inside the folder, the structure of files is (for instance, for application **algo1**):

```
algo1
|-- JSON file
|-- Makefile
|-- algo1.py
|-- testing.in
|-- training.in
|-- _src_
templates (_for Vitis_)
|-- Makefile
|-- source code
```

Meaning:

- `JSON file` is a setup file used in the Jupyter file;
- `algo1.py` is a file that calculates the error metric for the application;
- `testing.in` is a file that points to the test dataset;
- `training.in` is a file that points to the training dataset;
- `_src_` is a folder with the source code of the application;
- `components_rev.csv` is the features of the components regrading error and LUT+FF usage

**Observation:** this repository is lacking the input dataset for the image applications.  
**Observation:** other important requirements to execute the Jupyter files are listed in the `requirements.txt` file.
 
