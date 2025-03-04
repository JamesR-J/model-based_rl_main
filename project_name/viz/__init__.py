from collections import defaultdict
from project_name.viz.plot import (
    plot_pendulum,
    plot_cartpole,
    plot_pilco_cartpole,
    plot_acrobot,
    # noop,
    make_plot_obs,
    plot_generic,
    scatter,
    plot,
    noop,
)

_plotters = {
    "bacpendulum-v0": plot_pendulum,
    "bacpendulum-test-v0": plot_pendulum,
    "bacpendulum-tight-v0": plot_pendulum,
    "bacpendulum-medium-v0": plot_pendulum,
    "petscartpole-v0": plot_cartpole,
    "pilcocartpole-v0": plot_pilco_cartpole,
    "bacrobot-v0": plot_acrobot,
    "bacswimmer-v0": plot_generic,
    "bacreacher-v0": plot_generic,
    "bacreacher-tight-v0": plot_generic,
    "betatracking-v0": plot_generic,
    "betatracking-fixed-v0": plot_generic,
    "plasmatracking-v0": plot_generic,
    "bachalfcheetah-v0": noop,
}
plotters = defaultdict(lambda: plot_generic)
plotters.update(_plotters)
