epsilons = [1, 2, 3, 4, 5, 6]
inds = range(20)
experiments = [
    "easy-2d", "easy-10d", "tempered-2d", "tempered-10d",
    "hard-2d", "gauss-30d", "hard-gauss-2d"
]
algorithms = [
    "hmc", "dppa", "dpps", "mdppa", "mdpps", "barker"
]

result_dir = "results/banana/"
param_dir = "params/"

# rule all:
#     input: expand(param_dir + "{algo}_{exp}.py", algo=algorithms, exp=experiments)

rule create_params:
    output: param_dir + "{algo}_{exp}.py"
    shell: "touch {output}"

rule chain:
    input:
        # py = "run_chain.py",
        par = param_dir + "{algo}_{exp}.py"
    params:
        py = "run_chain.py",
        par_mod = "{algo}_{exp}"
    output: result_dir + "{algo}_{exp}_{eps}_{i}.csv"
    shell: "python {params.py} {wildcards.algo} {params.par_mod} {wildcards.exp} {wildcards.eps} {wildcards.i} {output}"

rule results:
    input:
        minibatch = expand(
            result_dir + "{algo}_{exp}_{eps}_{i}.csv",
            algo=algorithms[3:], exp=experiments[0:4], i=inds, eps=epsilons
        ),
        non_minibatch = expand(
            result_dir + "{algo}_{exp}_{eps}_{i}.csv",
            algo=algorithms[0:3], exp=experiments, i=inds, eps=epsilons
        )
    output: result_dir + "results.csv"
    # shell: "cat {input} > {output}"
    # cat-ing too many files seems to fail
    shell: "cat " + result_dir + "*.csv > {output}"

rule figures:
    input:
        py = "plot_banana.py",
        results = result_dir + "results.csv"
    output:
        "../Thesis/figures/banana_mmd.pdf",
        "../Thesis/figures/banana_clipping.pdf",
        "../Thesis/figures/banana_extra.pdf",
        "../Thesis/figures/banana_extra_clipping.pdf",
        "../Thesis/figures/banana_grad_clipping.pdf",
    shell: "python {input.py} {input.results}"
