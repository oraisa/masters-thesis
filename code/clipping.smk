clip_bounds = [0.1, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1000]
inds = range(20)
dims = [2, 10]

result_dir = "results/clipping/"

rule rwmh_chain:
    input: "rwmh_clipping.py"
    output: result_dir + "rwmh_{bound}_{dim}_{i}.csv"
    shell: "python {input} {wildcards.bound} {wildcards.dim} {wildcards.i} {output}"

rule results:
    input: expand(result_dir + "rwmh_{bound}_{dim}_{i}.csv", bound=clip_bounds, dim=dims, i=inds)
    output: result_dir + "results.csv"
    shell: "cat {input} > {result_dir}results.csv"

rule figures:
    input: 
        py = "plot_clipping.py",
        results = result_dir + "results.csv"
    # output: 
    shell: "python {input.py} {input.results}"
