epsilons = [1, 2, 3, 4, 5, 6]
dims = [2, 10]
temp_scales = [False, True]
inds = range(15)

result_dir = "results/banana/"

rule hmc_chain:
    input: "hmc_banana.py"
    output: result_dir + "hmc_{eps}_{dim}_{T}_{i}.csv"
    shell: "python {input} {wildcards.eps} {wildcards.dim} {wildcards.T} {wildcards.i} {output}"

rule penalty_chain:
    input: "penalty_banana.py"
    output: result_dir + "penalty_{eps}_{dim}_{T}_{i}.csv"
    shell: "python {input} {wildcards.eps} {wildcards.dim} {wildcards.T} {wildcards.i} {output}"

rule adv_penalty_chain:
    input: "adv_penalty_banana.py"
    output: result_dir + "adv_penalty_{eps}_{dim}_{T}_{i}.csv"
    shell: "python {input} {wildcards.eps} {wildcards.dim} {wildcards.T} {wildcards.i} {output}"

rule minibatch_penalty_chain:
    input: "minibatch_penalty_banana.py"
    output: result_dir + "mini_penalty_{eps}_{dim}_{T}_{i}.csv"
    shell: "python {input} {wildcards.eps} {wildcards.dim} {wildcards.T} {wildcards.i} {output}"

rule adv_minibatch_penalty_chain:
    input: "adv_minibatch_penalty_banana.py"
    output: result_dir + "adv_mini_penalty_{eps}_{dim}_{T}_{i}.csv"
    shell: "python {input} {wildcards.eps} {wildcards.dim} {wildcards.T} {wildcards.i} {output}"

rule barker_chain:
    input: "barker_banana.py"
    output: result_dir + "barker_{eps}_{dim}_{T}_{i}.csv"
    shell: "python {input} {wildcards.eps} {wildcards.dim} {wildcards.T} {wildcards.i} {output}"

rule results:
    input:
        hmc = expand(
            result_dir + "hmc_{eps}_{dim}_{T}_{i}.csv", eps=epsilons, dim=dims, i=inds, T=temp_scales
        ),
        dp_penalty = expand(
            result_dir + "penalty_{eps}_{dim}_{T}_{i}.csv", eps=epsilons, dim=dims, i=inds, T=temp_scales
        ),
        minibatch_penalty = expand(
            result_dir + "mini_penalty_{eps}_{dim}_{T}_{i}.csv",
            eps=epsilons, dim=dims, i=inds, T=temp_scales
        ),
        adv_dp_penalty = expand(
            result_dir + "adv_penalty_{eps}_{dim}_{T}_{i}.csv",
            eps=epsilons, dim=dims, i=inds, T=temp_scales
        ),
        adv_minibatch_penalty = expand(
            result_dir + "adv_mini_penalty_{eps}_{dim}_{T}_{i}.csv",
            eps=epsilons, dim=dims, i=inds, T=temp_scales
        ),
        barker = expand(
            result_dir + "barker_{eps}_{dim}_{T}_{i}.csv", eps=epsilons, dim=dims, i=inds, T=temp_scales
        )
    output: result_dir + "results.csv"
    shell: "cat {input} > {output}"

rule figures:
    input:
        py = "plot_banana.py",
        results = result_dir + "results.csv"
    output:
        "../Thesis/figures/banana.pdf"
    shell: "python {input.py} {input.results}"
