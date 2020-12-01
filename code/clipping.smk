clip_bounds = [0.5, 1, 2, 4, 6, 8, 10, 1000]
inds = range(10)
dims = [2, 10]

result_dir = "results/clipping/"

rule rwmh_chain:
    input: "rwmh_clipping.py"
    output: result_dir + "rwmh_{bound}_{dim}_{i}.csv"
    shell: "python {input} {wildcards.bound} {wildcards.dim} {wildcards.i} {output}"

rule hmc_chain:
    input: "hmc_clipping.py"
    output: result_dir + "hmc_{bound}_{dim}_{i}.csv"
    shell: "python {input} {wildcards.bound} {wildcards.dim} {wildcards.i} {output}"

rule results:
    input: 
        rwmh = expand(result_dir + "rwmh_{bound}_{dim}_{i}.csv", bound=clip_bounds, dim=dims, i=inds),
        hmc = expand(result_dir + "hmc_{bound}_{dim}_{i}.csv", bound=clip_bounds, dim=dims, i=inds)
    output: result_dir + "results.csv"
    shell: "cat {input} > {result_dir}results.csv"

rule figures:
    input: 
        py = "plot_clipping.py",
        results = result_dir + "results.csv"
    output: "../Thesis/figures/clipping.pdf"
    shell: "python {input.py} {input.results}"
