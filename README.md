# AlphaZero Evolutionary Strategies

![57662149-7e51-4280-a875-e67d04d388ab.gif](resources%2F57662149-7e51-4280-a875-e67d04d388ab.gif)

This project is inspired by [AlphaZeroES: Direct score maximization outperforms planning loss minimization](https://arxiv.org/abs/2406.08687), by Carlos Martin, Tuomas Sandholm.

Martin and Sandholm observe that the objective loss in AlphaZero is a surrogate objective, and does not optimize the search directly.  This is because the AlphaZero search function is not differentiable.  They therefore propose to optimize the objective directly using black box optimization techniques, specifically Open AI ES .

In this repo, I improve on their work using the latest frameworks available in JAX to demonstrate viability of this approach given limited compute resources.

[Report on experiments with Minatar.](https://api.wandb.ai/links/duanenielsen/pksqv309)