# k-Center Benchmark

This repository provides a comprehensive examination of the metric $k$-center problem, including implementations of various approximation and heuristic algorithms, and experiments evaluating their practical performance. The data is collected from the [OR-Library](http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/) `pmed` datasets.

The algorithms include: Gon, Sh, HS, Gr, CDS, CDSh, CDSh+, Scr, LS, LS Scr, SA Swp, SA Swp Scr, and Scr_p (not mentioned in the survey).

## Repository Structure

- `src/`
  - `utils.py`: Utility functions for data processing and metric computations.
  - `algorithms.py`: Implementations of approximation and heuristic algorithms for the $k$-center problem.
- `main.py`: Main script to run experiments and evaluations

Note that there is an `Scr_p` algorithm in `algorithms.py` that is not mentioned in the survey. Actually, this algorithm is invented by the author based on the Scr algorithm and the idea of CDSh+ algorithm. However, this algorithm does not perform well in the experiments, hence it is not included in the survey.

To evaluate the algorithms, use `run()` function in `main.py` and run the script.