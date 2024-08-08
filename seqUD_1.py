import numpy as np
import optuna
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState
import itertools
from numbers import Real
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
import warnings
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.distributions import CategoricalDistribution
import pyunidoe as pydoe
import matplotlib.pyplot as plt

class SeqUD(BaseSampler):
    def __init__(self, param_space, max_runs, n_runs_per_stage, random_state=None):
        super().__init__()
        self.param_space = param_space
        self.max_runs = max_runs
        self.n_runs_per_stage = n_runs_per_stage
        self.random_state = random_state
        self.current_stage = 0
        self.current_design = None
        self.lower_bounds = {k: v['Range'][0] for k, v in param_space.items()}
        self.upper_bounds = {k: v['Range'][1] for k, v in param_space.items()}
        self.study = None
        self._initialize_search_space()

    def _initialize_search_space(self):
        self.current_design, self.base_ud = self._generate_initial_design()

    def _generate_initial_design(self):
        n_params = len(self.param_space)
        np.random.seed(self.random_state)
        stat = pydoe.gen_ud(n=self.n_runs_per_stage, s=n_params, q=self.n_runs_per_stage, init="rand", crit="CD2", maxiter=100, vis=False)
        base_ud = stat["final_design"]

        if base_ud.shape[0] != self.n_runs_per_stage:
            raise ValueError(f"Unexpected base_ud size: {base_ud.shape[0]} rows, expected {self.n_runs_per_stage}")

        ud_space_scaled = np.zeros((self.n_runs_per_stage, n_params))
        for i, (k, v) in enumerate(self.param_space.items()):
            ud_space = np.linspace(self.lower_bounds[k], self.upper_bounds[k], self.n_runs_per_stage)
            design_column = np.clip(base_ud[:, i], 0, self.n_runs_per_stage - 1).astype(int)
            ud_space_scaled[:, i] = ud_space[design_column]

        return ud_space_scaled, base_ud

    def _shrink_search_space(self):
        best_trial = self.study.best_trial
        best_params = best_trial.params

        new_lower_bounds = {}
        new_upper_bounds = {}
        for k in self.param_space.keys():
            range_span = (self.upper_bounds[k] - self.lower_bounds[k]) / 2
            new_lower_bounds[k] = max(self.lower_bounds[k], best_params[k] - range_span / 2)
            new_upper_bounds[k] = min(self.upper_bounds[k] , best_params[k] + range_span / 2)
        self.lower_bounds = new_lower_bounds
        self.upper_bounds = new_upper_bounds

        n_params = len(self.param_space)

        stat = pydoe.gen_ud(n=self.n_runs_per_stage, s=n_params, q=self.n_runs_per_stage, init="rand", crit="CD2", maxiter=100, vis=False)
        additional_design = stat["final_design"]

        if additional_design.shape[0] != self.n_runs_per_stage:
            raise ValueError(f"Unexpected additional_design size: {additional_design.shape[0]} rows, expected {self.n_runs_per_stage}")

        additional_design_scaled = np.zeros((self.n_runs_per_stage, n_params))
        for i, (k, v) in enumerate(self.param_space.items()):
            ud_space = np.linspace(self.lower_bounds[k], self.upper_bounds[k], self.n_runs_per_stage)
            design_column = np.clip(additional_design[:, i], 0, self.n_runs_per_stage - 1).astype(int)
            additional_design_scaled[:, i] = ud_space[design_column]

        self.base_ud = np.vstack([self.base_ud, additional_design_scaled])

        return additional_design_scaled

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        if self.study is None:
            self.study = study

        if trial.number % self.n_runs_per_stage == 0 and trial.number > 0:
            self.current_stage += 1
            self.current_design = self._shrink_search_space()

        trial.system_attrs["grid_id"] = trial.number % self.n_runs_per_stage

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        pass

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        return {param_name: dist for param_name, dist in self.param_space.items()}

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:
        return {}

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        grid_id = trial.system_attrs.get("grid_id")
        if grid_id is None:
            return param_distribution.sample()

        if param_name not in self.param_space:
            raise ValueError(f"The parameter name, {param_name}, is not found in the given space.")

        param_value = self.current_design[grid_id][list(self.param_space.keys()).index(param_name)]
        contains = param_distribution._contains(param_distribution.to_internal_repr(param_value))
        if not contains:
            warnings.warn(
                f"The value {param_value} is out of range of the parameter {param_name}. "
                f"The value will be used but the actual distribution is: {param_distribution}."
            )

        return param_value

    def optimize(self, objective):
        for _ in range(self.max_runs):
            self.study.optimize(objective, n_trials=self.n_runs_per_stage)


if __name__ == "__main__":
    def octopus(trial):
        x1 = trial.suggest_uniform('x1', 0, 1)
        x2 = trial.suggest_uniform('x2', 0, 1)
        y = 2 * np.cos(10 * x1) * np.sin(10 * x2) + np.sin(10 * x1 * x2)
        return y
    
    param_space = {
        'x1': {'Type': 'continuous', 'Range': [0, 1], 'Wrapper': lambda x: x},
        'x2': {'Type': 'continuous', 'Range': [0, 1], 'Wrapper': lambda x: x}
    }
    
    sampler = SeqUD(param_space, max_runs=10, n_runs_per_stage=40, random_state=None)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(octopus, n_trials=400)
    
    print(f"Best trial value: {study.best_value}")
    print(f"Best trial params: {study.best_params}")
    
    def plot_trajectory(xlim, ylim, func, study, title):
        grid_num = 25
        xlist = np.linspace(xlim[0], xlim[1], grid_num)
        ylist = np.linspace(ylim[0], ylim[1], grid_num)
        X, Y = np.meshgrid(xlist, ylist)
        Z = np.zeros((grid_num, grid_num))
        for i, x1 in enumerate(xlist):
            for j, x2 in enumerate(ylist):
                Z[j, i] = func(optuna.trial.FixedTrial({"x1": x1, "x2": x2}))
    
        cp = plt.contourf(X, Y, Z)
        plt.scatter([t.params['x1'] for t in study.trials],
                    [t.params['x2'] for t in study.trials], color="red")
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.colorbar(cp)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(title)
    
    plot_trajectory([0, 1], [0, 1], octopus, study, "SeqUD")
