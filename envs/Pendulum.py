from learning_control.core.systems import InvertedPendulum
from matplotlib.pyplot import figure
from numpy import array, zeros
import numpy as np
from scipy.integrate import solve_ivp
import gym
from os import path

class StochasticInvertedPendulum(InvertedPendulum, gym.Env):
    """docstring for StochasticInvertedPendulum"""
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, *args, **kw):
        super(StochasticInvertedPendulum, self).__init__(*args, **kw)
        self.random = np.random.RandomState(0)
        self.viewer = None
        self.xerr = 1e-3
        self.max_cost = 1e2

    def reset(self):
        high = np.array([np.pi, 1])
        # high = np.array([1, 1])
        state = self.random.uniform(low=-high, high=high)
        self.state = np.array(state)
        return state

    def step_from_state(self, x, u, dt=1/50):
        t = 0
        x = super(StochasticInvertedPendulum, self).step(x, u, t, t+dt)
        th,thdot = x
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
        return x, -costs, False, {'goal_reached':False}

    def step(self, u, dt=1/50):
        x = self.state
        t = 0
        x = super(StochasticInvertedPendulum, self).step(x, u, t, t+dt)
        th,thdot = x
        costs = np.minimum(angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2), self.max_cost)
        self.state = x
        return x, -costs, False, {'goal_reached':False}

    def simulate(self, x_0, controller, ts, processed=True, uerr=1e-3, xerr=1e-3, atol=1e-6, rtol=1e-6):
        """Simulate system from initial state with specified controller.

        Approximated using Runge-Kutta 4,5 solver.

        Actions computed at time steps and held constant over sample period.

        Inputs:
        Initial state, x_0: numpy array
        Control policy, controller: Controller
        Time steps, ts: numpy array
        Flag to process actions, processed: bool
        Absolute tolerance, atol: float
        Relative tolerance, rtol: float

        Outputs:
        State history: numpy array
        Action history: numpy array
        """

        assert len(x_0) == self.n

        N = len(ts)
        xs = zeros((N, self.n))
        obs = zeros((N, self.n))
        us = [None] * (N - 1)

        controller.reset()

        xs[0] = x_0
        obs[0] = x_0 + self.noise(0, xerr)
        for j in range(N - 1):
            x = xs[j]
            t = ts[j]
            u = controller.eval(x, t) + self.noise(0, uerr)
            us[j] = u
            u = controller.process(u)
            xs[j + 1] = super(StochasticInvertedPendulum, self).step(x, u, t, ts[j + 1])
            obs[j + 1] = xs[j + 1] + self.noise(0, xerr)

        if processed:
            us = array([controller.process(u) for u in us])

        return obs, xs, us

    def noise(self, mean, std):
        return self.random.normal(mean, std**2)

    def render(self, mode='human', state=None):
        state = self.state if state is None else state
        last_u = None
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(state[0] + np.pi / 2)
        if last_u:
            self.imgtrans.scale = (-last_u / 2, np.abs(last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def cost_np_vec(self, x,u,x_):
        cost = np.minimum(angle_normalize(x[:,0]) ** 2 + .1 * x[:,1] ** 2 + .001 * (u ** 2), self.max_cost)
        return cost

    def is_done(self, *args, **kw):
        return np.array([False])


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


