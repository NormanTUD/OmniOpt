import torch
from ax.plot.pareto_frontier import plot_pareto_frontier
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

# Plotting imports and initialization
from ax.utils.notebook.plotting import init_notebook_plotting, render
from botorch.test_functions.multi_objective import BraninCurrin

# Load our sample 2-objective problem
branin_currin = BraninCurrin(negate=True).to(
    dtype=torch.double,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

ax_client = AxClient()

ax_client.create_experiment(
    name="moo_experiment",
    parameters=[
        {
            "name": f"x{i+1}",
            "type": "range",
            "bounds": [0.0, 1.0],
        }
        for i in range(2)
    ],
    objectives={
        # `threshold` arguments are optional
        "result1": ObjectiveProperties(minimize=False),
        "result2": ObjectiveProperties(minimize=False),
    },
    overwrite_existing_experiment=True,
    is_test=True,
)

def evaluate(parameters):
    evaluation = branin_currin(
        torch.tensor([parameters.get("x1"), parameters.get("x2")])
    )
    # In our case, standard error is 0, since we are computing a synthetic function.
    # Set standard error to None if the noise level is unknown.
    return {"result1": (evaluation[0].item(), 0.0), "result2": (evaluation[1].item(), 0.0)}

for i in range(5):
    parameters, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

objectives = ax_client.experiment.optimization_config.objective.objectives
frontier = compute_posterior_pareto_frontier(
    experiment=ax_client.experiment,
    data=ax_client.experiment.fetch_data(),
    primary_objective=objectives[1].metric,
    secondary_objective=objectives[0].metric,
    absolute_metrics=["result1", "result2"],
    num_points=20,
)
render(plot_pareto_frontier(frontier, CI_level=0.90))
