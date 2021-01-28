# Machine Learning Assisted GST

Final year project for MEng. Computer Science at the University of Bristol. Current working title:

> Facilitating the calibration of complex quantum photonic circuits with machine learning 
> assisted gate set tomography

At the moment, this repository is just for my dissertation notes and introduction chapters.

## Paper

Paper can be found in `build/main.pdf`. If you want to build it yourself for whatever reason, run:


```bash
cd src
make build
```

which will output to the `build/` directory.

## Targets

### Preliminary Reading

- [ ] Understand the importance of the problem of estimating the average fidelity of fundamental gates for quantum computing in any platform.
- [ ] Understand the importance of process tomography and its limitations due to reference frames.
    - [x] Read Nielsen & Chuang Chapter 8.
    - [ ] Read [Qiskit Documentation](https://qiskit.org/textbook/ch-quantum-hardware/randomized-benchmarking.html) to understand Randomized Benchmarking.
    - [ ] Know how the above are related to measurable quantities, and how to measure expectation values and POVMs in terms of projectors.
- [ ] Understand the importance of Gate Set Tomography (GST).
    - [ ] Read the following papers ([1](https://www.nature.com/articles/ncomms14485), [2](https://arxiv.org/pdf/2009.07301.pdf)).
    - [ ] Understand the impossibility of reproducing the same exact procedure to a photonic path-encoded qubit.
    - [ ] Rephrase the GST problem to learning the phase shifter calibration, grouping the beam splitters at the start.

### Milestones

- [ ] Get familiar with using Quantum Hamiltonian Learning to estimate the Mach-Zehnders’ calibration, including  crosstalk.
- [ ] Estimate the feasibility of the whole set of parameter estimation. How many parameters have to be estimated for L = 5, 7, 11? Is it computationally feasible? What computational tricks can we use to speed up?
- [ ] Data is already available for L = 5, 7. Create some Python scripts to analyse it.
- [ ] If possible, do longer chain experiments.
- [ ] Include GST type of estimates to state preparation and measurement to the QHL approach. See [pyGSTi](https://github.com/pyGSTio/pyGSTi).
- [ ] Final output: an estimate of a photonic 1-qubit average gate infidelity to compare with other platforms. 
- [ ] Possible comments on context-dependence tests, related to crosstalk and the role of environment. Read [Macroscopic instructions vs microscopic operations in quantum circuits](https://arxiv.org/abs/1708.08173).
