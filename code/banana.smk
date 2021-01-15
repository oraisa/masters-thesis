epsilons = [1, 2, 3, 4, 5, 6]
inds = range(15)
experiments = [
    "easy-2d", "easy-10d", "tempered-2d", "tempered-10d",
    # "hard-2d",
    # "gauss-50d"
]
algorithms = [
    "hmc", "dppa", "dpps", "mdppa", "mdpps", "barker"
]

result_dir = "results/banana/"
param_dir = "params/"
chain_shell_string = "python {input} {wildcards.exp} {wildcards.eps} {wildcards.i} {output}"

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

# rule hmc_chain:
#     input: "hmc_banana.py"
#     output: result_dir + "hmc_{exp}_{eps}_{i}.csv"
#     shell: chain_shell_string

# rule penalty_chain:
#     input: "penalty_banana.py"
#     output: result_dir + "penalty_{exp}_{eps}_{i}.csv"
#     shell: chain_shell_string

# rule adv_penalty_chain:
#     input: "adv_penalty_banana.py"
#     output: result_dir + "adv_penalty_{exp}_{eps}_{i}.csv"
#     shell: chain_shell_string

# rule minibatch_penalty_chain:
#     input: "minibatch_penalty_banana.py"
#     output: result_dir + "mini_penalty_{exp}_{eps}_{i}.csv"
#     shell: chain_shell_string

# rule adv_minibatch_penalty_chain:
#     input: "adv_minibatch_penalty_banana.py"
#     output: result_dir + "adv_mini_penalty_{exp}_{eps}_{i}.csv"
#     shell: chain_shell_string

# rule barker_chain:
#     input: "barker_banana.py"
#     output: result_dir + "barker_{exp}_{eps}_{i}.csv"
#     shell: chain_shell_string

# rule results:
#     input:
#         hmc = expand(
#             result_dir + "hmc_{exp}_{eps}_{i}.csv",
#             eps=epsilons, exp=experiments, i=inds
#         ),
#         dp_penalty = expand(
#             result_dir + "penalty_{exp}_{eps}_{i}.csv",
#             eps=epsilons, exp=experiments, i=inds
#         ),
#         minibatch_penalty = expand(
#             result_dir + "mini_penalty_{exp}_{eps}_{i}.csv",
#             eps=epsilons, exp=experiments, i=inds
#         ),
#         adv_dp_penalty = expand(
#             result_dir + "adv_penalty_{exp}_{eps}_{i}.csv",
#             eps=epsilons, exp=experiments, i=inds
#         ),
#         adv_minibatch_penalty = expand(
#             result_dir + "adv_mini_penalty_{exp}_{eps}_{i}.csv",
#             eps=epsilons, exp=experiments, i=inds
#         ),
#         barker = expand(
#             result_dir + "barker_{exp}_{eps}_{i}.csv",
#             eps=epsilons, exp=experiments, i=inds
#         )
#     output: result_dir + "results.csv"
#     shell: "cat {input} > {output}"

rule results:
    input: expand(result_dir + "{algo}_{exp}_{eps}_{i}.csv", algo=algorithms, exp=experiments, i=inds, eps=epsilons)
    output: result_dir + "results.csv"
    shell: "cat {input} > {output}"

rule figures:
    input:
        py = "plot_banana.py",
        results = result_dir + "results.csv"
    output:
        "../Thesis/figures/banana_mmd.pdf",
        "../Thesis/figures/banana_clipping.pdf"
    shell: "python {input.py} {input.results}"
