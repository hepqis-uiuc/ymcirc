# Issues round 3

## Background
The functionality documented in forder-refactor-plan.md, issues-resolution-plan.md, and issues-round-2-resolution-plan.md (in that order) has been succesfully implemented. However, while testing the this work, I uncovered an independent performance issue which I would now like to address.

## The problem
There appears to be a significant performance bottleneck in the "walk through lattice and decode" logic of the constructor for `ParsedLatticeResult` (see roughly lines 82-105 in ymcirc/parsed_lattice_result.py). I uncovered this by running run/time_evol.py with the following script_options:
```Python
script_options = configure_script_options(
        dimensionality_string="d=2",
        truncation_string="T1",
        lattice_size=2,
        sim_times=np.linspace(0.0, 2.5, num=20),
        n_trotter_steps=1,
        n_shots=10000,
        use_ancillas=True,
        control_fusion=True,
        prune_controls=True,
        warn_unphysical_links=True,
        method='matrix_product_state',
        cache_mag_evol_circuit=True,
        load_circuit_from_file=None,
        save_circuit_to_qasm=False,
        save_circuit_to_qpy=False,
        save_circuit_diagrams=False,
        save_plots=False,
        save_sim_data=False,
        serialized_circ_dir=PROJECT_ROOT / "serialized-circuits",
        plots_dir=PROJECT_ROOT / "plots",
        sim_results_dir=PROJECT_ROOT / "sim-results",
        mag_hamiltonian_matrix_element_threshold=0.9
    )
```

With these options, after running circuit simulations, the script would appear to hang for a long time. I traced this to the call to create a `MeasurementResults` instance in line 435 of the `run_circuit_simulations` method in run/functions.py. When handed a dictionary with string keys, the `MeasurementResults` class internally constructs a `ParsedLatticeResult` for each string. As mentioned above, the init method for the `ParsedLatticeResult` class is currently expensive due to the current implementation of the "walk through lattice and decode" logic. Moreover, this part of the code is hot because of the loop in functions.py over all the circuit simulation jobs which have been run, followed by a loop over every measurement bit string appearing in a given job result.

I can think of two parallel possibilities for addressing this problem: (a) compute and cache `ParsedLatticeResult` instances at the level of the functions.py script, and then hand off a counts dict with `ParsedLatticeResult` keys to `MeasurementResults`, and (b) improving the performance of the init method in `ParsedLatticeResult`. I think (a) is insufficient since generically there are many possible global lattice measurement bit strings, and enough unique ones will be encountered such that even with caching there would still be poor performance.

## Task
The following steps should be taken:
1. Investigate the codebase to ensure a detailed understanding of why this performace bottleneck arises. Confirm or invalidate the explanations of the performance issues indicated above.
2. Try to come up with a solution to the problem that's as good as or better than the one I sketched above. If you can't think of anything better, then continue with my proposed solution strategy.
3. Write a report `issues-round-3-resolution-plan.md` which consists of the following sections:
 1. A one or two paragraph summary of your findings.
 2. A one or two paragraph summary of your solution.
 3. A detailed implementation plan with clear stages.
 4. A final section breaking down tje implementation plan into concrete todo items.
