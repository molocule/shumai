import * as sm from "@shumai/shumai"

/*
Adapted from python code here: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
*/

// util functions
function getRandomInt(max) {
    return Math.floor(Math.random() * max);
}

function assert(condition, message) {
    if (!condition) {
        throw new Error(message || "Assertion failed");
    }
}

// classes from gym env that are unimplemented

class Discrete {
    constructor (low = 0, high) {
        this.values = sm.arange(low, high)
        this.low = low
        this.high = high
    }

    contains(action) {
        return this.low <= action.valueOf() <= this.high
    }

    sample() {
        const ind = getRandomInt(this.values.shape[0])
        return this.values.index([ind])
    }
}

class Box {
    constructor(low, high) {
        this.low = low
        this.high = high
    }
}

// actual cartpole env

export class CartPoleEnv {
    constructor(render_mode = null) {
        this.gravity = 9.8
        this.masscart = 1.0
        this.masspole = 0.1
        this.total_mass = this.masspole + this.masscart
        this.length = 0.5  // actually half the pole's length
        this.polemass_length = this.masspole * this.length
        this.force_mag = 10.0
        this.tau = 0.02  // seconds between state updates
        this.kinematics_integrator = "euler"

        // Angle at which to fail the episode
        this.theta_threshold_radians = 12 * 2 * Math.pi / 360
        this.x_threshold = 2.4

        const box_high = new Float32Array([
                this.x_threshold * 2,
                Number.MAX_VALUE,
                this.theta_threshold_radians * 2,
                Number.MAX_VALUE,
        ])
        const box_low = box_high.map(value => -value)

        this.action_space = new Discrete(0, 2)
        this.observation_space = new Box(box_low, box_high)

        this.render_mode = render_mode

        this.screen_width = 600
        this.screen_height = 400
        this.isopen = true
        this.state = null

        this.steps_beyond_terminated = null

    }

    step(action) {
        const err_msg = `${action} invalid`
        assert(this.action_space.contains(action), err_msg)
        assert(this.state !== null, "Call reset before using step method.")
        let [x, x_dot, theta, theta_dot] = this.state.valueOf()
        const force = action == 1 ? this.force_mag : - this.force_mag
        const costheta = Math.cos(theta)
        const sintheta = Math.sin(theta)

        const temp = ( force + this.polemass_length * theta_dot**2 * sintheta) / this.total_mass
        const thetaacc = (this.gravity * sintheta - costheta * temp) / ( this.length * (4.0 / 3.0 - this.masspole * costheta**2 / this.total_mass))
        const xacc = temp - this.polemass_length * thetaacc * costheta / this.total_mass

        if (this.kinematics_integrator == "euler") {
            x = x + this.tau * x_dot
            x_dot = x_dot + this.tau * xacc
            theta = theta + this.tau * theta_dot
            theta_dot = theta_dot + this.tau * thetaacc
        } else {
            // semi-implicit euler
            x_dot = x_dot + this.tau * xacc
            x = x + this.tau * x_dot
            theta_dot = theta_dot + this.tau * thetaacc
            theta = theta + this.tau * theta_dot
        }

        this.state = new Float32Array([x, x_dot, theta, theta_dot])

        const terminated = Boolean(
            x < -this.x_threshold
            || x > this.x_threshold
            || theta < -this.theta_threshold_radians
            || theta > this.theta_threshold_radians
        )

        let reward = -1.0
        if (!terminated) {
            reward = 1.0
        } else if (this.steps_beyond_terminated == null) {
            // pole just fell
            this.steps_beyond_terminated = 0
            reward = 1.0
        } else {
            if (this.steps_beyond_terminated == 0) {
                console.log(
                    "You are calling 'step()' even though this environment has already returned terminated = True. You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior."
                )
            }
            this.steps_beyond_terminated += 1
            reward = 0.0
        }

        if (this.render_mode == "human") {
            this.render()
        }
        return [sm.tensor(this.state), reward, terminated, false, {}]
    }

    reset() {
        // super().reset(seed=seed)
        const lower = -0.05
        const upper = 0.05

        const rand = sm.randn([4])
        const scale = sm.full([4], upper - lower)
        const diff = sm.full([4], lower)
        this.state = sm.add(sm.mul(rand, scale), diff)
        this.steps_beyond_terminated = null
        if (this.render_mode == "human") {
            this.render()
        }
        return [this.state, {}]
    }

    render() {
        if (this.render_mode == null) {
            console.log("You are calling render method without specifying any render mode.")
            return
        }

        var params = {
            fullscreen: true
          };
        var elem = document.body;

        const world_width = this.x_threshold * 2
        const scale = this.screen_width / world_width
        const polewidth = 10.0
        const polelen = scale * (2 * this.length)
        const cartwidth = 50.0
        const cartheight = 30.0
        var two = new Two(params).appendTo(elem);

        // Two.js has convenient methods to make shapes and insert them into the scene.
        var x = two.width * 0.5;
        var y = two.height * 0.1;
        var cart = two.makeRectangle(x, y, cartwidth, cartheight);
        y = y + 0.5 * cartheight
        var pole = two.makeRectangle(x, y, cartwidth, cartheight);

        cart.fill = '#FF8000';
        cart.stroke = 'orangered';
        cart.linewidth = 5;

        pole.fill = 'rgb(0, 200, 255)';
        pole.opacity = 0.75;
        pole.noStroke();

        two.update();

        // this.clock = pygame.time.Clock()
        }
}
