epsilons = [1, 2, 3, 4, 5, 6, 7, "inf"]
dims = [2]
inds = range(10)

result_dir = "results/banana/"

rule hmc_chain:
    input: "hmc_banana.py"
    output: result_dir + "hmc_{eps}_{dim}_{i}.csv"
    shell: "python {input} {wildcards.eps} {wildcards.dim} {wildcards.i} {output}"

rule penalty_chain:
    input: "penalty_banana.py"
    output: result_dir + "penalty_{eps}_{dim}_{i}.csv"
    shell: "python {input} {wildcards.eps} {wildcards.dim} {wildcards.i} {output}"

rule results:
    input:
        hmc = expand(result_dir + "hmc_{eps}_{dim}_{i}.csv", eps=epsilons, dim=dims, i=inds),
        dp_penalty = expand(result_dir + "penalty_{eps}_{dim}_{i}.csv", eps=epsilons, dim=dims, i=inds)
    output: result_dir + "results.csv"
    shell: "cat {input} > {output}"

rule figures:
    input:
        py = "plot_banana.py",
        results = result_dir + "results.csv"
    output:
        "../Thesis/figures/banana.pdf"
    shell: "python {input.py} {input.results}"
