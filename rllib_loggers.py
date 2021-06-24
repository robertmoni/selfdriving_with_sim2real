from tensorboardX import SummaryWriter
import wandb
import logging
import ray.tune.logger
from ray.tune.result import (NODE_IP, TRAINING_ITERATION, TIMESTEPS_TOTAL)
from ray.tune.utils import flatten_dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from trajectory_plot import plot_trajectories

logger = logging.getLogger(__name__)
weights_and_biases_project = 'duckietown-rllib'


class TensorboardImageLogger(ray.tune.logger.Logger):
    def __init__(self, config, logdir, trial):
        super(TensorboardImageLogger, self).__init__(config, logdir, trial)
        self._writer = SummaryWriter(logdir=logdir, filename_suffix="_img")

    def on_result(self, result):
        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]

        traj_fig = plot_trajectories(result['hist_stats']['_robot_coordinates'])
        traj_fig.savefig("Trajectory.png")
        self._writer.add_figure("TrainingTrajectories", traj_fig, global_step=step)
        plt.close(traj_fig)

        self.flush()

    def flush(self):
        if self._writer is not None:
            self._writer.flush()



