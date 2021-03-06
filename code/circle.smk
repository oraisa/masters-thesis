param_dir = "params/"
result_dir = "results/circle/"

rule circle_chain:
    input:
        # py = "run_chain.py",
        par = param_dir + "{algo}_{exp}.py"
    params:
        py = "run_chain.py",
        par_mod = "{algo}_{exp}"
    output: result_dir + "{algo}_{exp}_{eps}_{i}.csv"
    shell: "python {params.py} {wildcards.algo} {params.par_mod} {wildcards.exp} {wildcards.eps} {wildcards.i} {output}"

rule circle_results:
    input:
        chains = expand(
            result_dir + "{algo}_circle_{eps}_{i}.csv",
            i=range(20), algo=["hmc", "dpps", "dppa"], eps=[0.5, 0.75, 1]
        )
    output: result_dir + "results.csv"
    shell: "cat {input} > {output}"

rule circle_figures:
    input:
        py = "plot_circle.py",
        results = result_dir + "results.csv"
    output:
        "../Thesis/figures/circle.pdf"
    shell:
        "python {input.py} {input.results}"
