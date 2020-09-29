# Gate-Set Tomography 

> **Quantum State Tomography**: the process of reconstructing a quantum state based on a series of **identical** quantum states. i.e. by repeatedly measuring the same quantum state, the frequency counts can be used to infer the actual probabilities and therefore the quantum state. We can use Born's Rule to define a density matrix which is most likely to fit with the observations.

> **Quantum Process Tomography**: the process of characterising an unknown quantum process, i.e. describing the behaviour of the process for every possible input. The standard method is to simply feed a set of known quantum states and then use quantum state tomography to reconstruct the output states.

There are actually many different methods for the above. Add more info if needed? It's clear that something like Bayesian statistics is needed since we're reconstructing some unknown based on observations.

## Catarina's Paper

### Overview

> Still not sure on what I'm actually supposed to know from this paper !!

If we want to fully characterise a quantum circuit, then we must characterise each quantum logic gate. Gate-Set tomography is a method of doing so and was originally proposed by Robin Blume-Kohout et al. in 2013; it  is able to consistently characterise an entire set of quantum logic gates in a 'black-box' device. 

We want to identify the Gate-Set $\mathscr{G}$, which will characterise the process gates, initial states, and measurements. It is as follows:

$$
\mathscr{G} = \big\{ | \rho \rangle, \langle E |, \{ G_k \} \big\}
$$

where:

* $| \rho \rangle$ is the initialisation of repeatable 1-qubit states
* $ \langle E | $ is the measurement of 2-outcome POVMs
* $\{ G_1, \dots, G_k \}$ is a set of 1-qubit quantum operations

> **Measure**: a function that assigns a positive (or $+\infty$) number to each suitable subset of a set. It can be interpreted as the generalisation of concepts such as length, area, and volume.

> **Positive-Operator Valued Measure (POVM)**: I'll get there :sob:

The frequency of the two outcomes are collected from a set of $N$ experiments, which are each characterised by the sequence of process gates $s = \{ G_{s_1}, \dots, G_{s_L} \}$ where $L$ is the length of the process chain. 

A key point of this is that we're characterising the system as a path through a chain of Mach-Zehnder interferometers. Read Caterina's paper for the real explanation, but essentially this method is interested in the beam splitters and phase shifters.

Data is analysed by several $\chi^2$-estimations, progressively adding more data from longer sequences of gates and then maximizing the likelihood function $\mathscr{L}(\mathscr{G}) = \Pr (data|\mathscr{G})$.

### ML-Assisted Gate-Set Tomography

The above method is bad. It takes multiple steps of $\chi^2$-minimisation and MLE which is computationally intensive. An alternative is to use machine learning techniques inspired by Quantum Hamiltonian Learning (QHL), which aims to describe the dynamical Hamiltonian evolution of the system. 

> This essentially means that we're not interested in the intermediate steps (i.e. what the splitters and interferometers are doing), we're only interested in the evolution of the state. As long as we get the desired output, then we don't care about anything else.

## Experimental quantum Hamiltonian Learning

<div align="center">
    <a href="https://doi.org/10.1038/nphys4074">Wang, J., Paesani, S., Santagati, R. et al. Experimental quantum Hamiltonian learning. Nature Phys 13, 551–555 (2017)</a><br/><br/>
</div>

This paper outlines a Quantum Hamiltonian Learning (QHL), which is a method of characterising quantum systems, aided by machine learning. Draws mainly on some information from an older paper [Hamiltonian Learning and Certification Using Quantum Resources
](https://arxiv.org/abs/1309.0876) which is probably a lot more useful for information actually _about_ QHL.

Three modes of estimating the Hamiltonian, proposed in the QHL paper:

* **Classical Likelihood Evaluation (CLE)**: pretty much just regular bayesian where we calculate the likelihood $\Pr (D|x_i)$ using a classical computer given an experiment outcome $D$. This is very inefficient since the likelihood is a quantum process which is difficult for classical computers.
* **Quantum Likelihood Evaluation (QLE)**: same as above, but we use a *trusted* quantum simulator to estimate the likelihood $\Pr (D|x_i)$ to be the number of times that an outcome $D$ occurs in a large number of experiments. This is good if the likelihood is polynomially small.
* **Interactive Quantum Likelihood Evaluation (IQLE)**: an improvement on QLE which attempts to deal with the Loschmidt Echo, which means that for complex systems, two near identical Hamiltonians will diverge after a short amount of time. i.e. QLE is only useful for short quantum processes. (MORE STUFF ON IQLE HERE, ITS COMPLICATED)

## Original Gate-Set Tomography Paper

<div align="center">
    <a href="https://arxiv.org/pdf/1310.4492.pdf">Robust, self-consistent, closed-form tomography of quantum logic gates on a trapped ion qubit</a><br/><br/>
</div>
