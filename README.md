# Synthpop Sandbox
## A suite of development tools for creating synthetic populations

Our aim is to produce a set of reproducible examples to showcase the various methods available for creating synthetic population data, either directly from survey data or from generative populations, such as those produced by a GAN or through IPF. The suite consists of the following functionality that can be bolted together into a comprehensive data generation pipeline.

- Some examples of implementation of IPF (iterative proportional fitting) in both Python and R (via IPFN/PyIPF and MIPFP, respectively) for creation of both generative and synthetic populations
- A Python implementation of the TRS (truncate, replicate, sample) algorithm for integerisation of populations produced by the IPF algorithm, adapted from the R version [here](https://spatial-microsim-book.robinlovelace.net/smsimr#sintegerisation), paper [here](https://www.sciencedirect.com/science/article/pii/S0198971513000240)
- Basic examples of parallellisation with the Python `multiprocessing` module, intended to be of use for (e.g.) small-area applications where processes can be massively parallellised
- An implementation of a random replacement algorithm for creating synthetic populations from survey data; this is a simplified version of simulated annealing in that it omits a temperature
- A Python methodology for simulated annealing using real or generative populations as a source, allowing for complex and/or custom constraints
- A toy GAN for testing and development of generative populations
- A basic generative population methodology using the SDV (Synthetic Data Vault) package

Developed for use within the [PHI-UK research consortium](https://www.phiuk.org/), "Innovating with people, places and communities".

Contact the development team below with queries, or raise an issue here.
- Hugh P. Rice (lead), h.p.rice@leeds.ac.uk
- Ric Colasanti, r.l.colasanti@leeds.ac.uk
- Andreas Hoehn, andreas.hoehn@glasgow.ac.uk

You can install the package in multiple ways.

1. The simplest way is to install via `pip`.

```
pip install git+https://github.com/paddy-r/synthpop_sandbox.git
```

2. Alternatively, add Synthpop Sandbox to your Anaconda/Miniconda/Micromamba environment file in the `pip` section.

``` 
pip:
    - git+https://github.com/paddy-r/synthpop_sandbox#egg=synthpop_sandbox
```

3. If you'd like to contribute to the development of Synthpop Sandbox, clone the repository using Git, then install it in development mode.

```
git clone https://github.com/paddy-r/synthpop_sandbox.git
pip install -e .
```