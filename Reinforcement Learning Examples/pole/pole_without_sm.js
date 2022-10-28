import * as sm from "@shumai/shumai"

/*
    main : controls simulation iterations and implements the learning system

    cart_and_pole: the cart and pole dynamics; given action and current state, estimates the next state

    get_box: cart-pole's state space divided into 162 boxes

*/




export class pole {
    constructor() {
        this.N_BOXES = 162
        this.ALPHA = 1000
        this.BETA = 0.5
        this.GAMMA = 0.95
        this.LAMBDAw = 0.9
        this.LAMBDAv = 0.8
        this.MAX_FAILURES = 100
        this.MAX_STEPS = 100000

        this.state = new Float32Array([0, 0, 0, 0]) // x, x_dot, theta, theta_dot
        // this.w = sm.full([this.N_BOXES], 0)
        // this.v = sm.full([this.N_BOXES], 0)
        // this.e = sm.full([this.N_BOXES], 0)
        // this.xbar = sm.full([this.N_BOXES], 0)
        this.w = new Array(this.N_BOXES).fill(0);
        this.v = new Array(this.N_BOXES).fill(0);
        this.e = new Array(this.N_BOXES).fill(0);
        this.xbar = new Array(this.N_BOXES).fill(0);

        // parameters required for simulation
        this.GRAVITY = 9.8
        this.MASSCART = 1.0
        this.MASSPOLE = 0.1
        this.TOTAL_MASS = this.MASSCART + this.MASSPOLE
        this.LENGTH = 0.5
        this.POLEMASS_LENGTH = this.MASSPOLE * this.LENGTH
        this.FORCE_MAG = 10.0
        this.TAU = 0.02
        this.FOURTHIRDS = 1.3333333333333

        // parameters required for getting box from current state
        this.one_degree = 0.0174532
        this.six_degrees = 0.1047192
        this.twelve_degrees = 0.2094384
        this.fifty_degrees = 0.87266
    }

    prob_push_right(s) {
        return (1.0 / (1.0 + Math.exp(-1 * Math.max(-50.0, Math.min(s, 50.0)))))
    }

    get_box() {
        let box = 0
        const x = this.state[0]
        const x_dot = this.state[1]
        const theta = this.state[2]
        const theta_dot = this.state[3]

        if (x < -2.4 || x > 2.4  || theta < -this.twelve_degrees || theta > this.twelve_degrees) {
                return (-1) // failure
        }

        if(x < -0.8) {
            box = 0
        } else if(x < 0.8) {
            box = 1
        } else {
            box = 2
        }

        if(x_dot < -0.5) {
            box = box
        } else if(x < 0.5) {
            box += 3
        } else {
            box += 6
        }

        if (theta < -this.six_degrees) {
            box = box
        } else if(theta < -this.one_degree) {
            box += 9
        } else if (theta < 0) {
            box += 18
        } else if (theta < this.one_degree) {
            box += 27
        } else if (theta < this.six_degrees) {
            box += 36
        } else {
            box += 45
        }

        if (theta_dot < -this.fifty_degrees) {
            box = box
        } else if(theta_dot < this.fifty_degrees) {
            box += 54
        } else {
            box += 108
        }

        return box
    }

    cart_pole(action) {
        const force = (action > 0) ? this.FORCE_MAG : -1 * this.FORCE_MAG
        const costheta = Math.cos(this.state[2])
        const sintheta = Math.sin(this.state[2])

        const temp = (force + this.POLEMASS_LENGTH * this.state[3] * this.state[3] * sintheta) / this.TOTAL_MASS
        const thetaacc = (this.GRAVITY * sintheta - costheta * temp) / (this.LENGTH * (this.FOURTHIRDS - this.MASSPOLE * costheta * costheta / this.TOTAL_MASS))
        const xacc = temp - this.POLEMASS_LENGTH * thetaacc * costheta / this.TOTAL_MASS

        this.state[0] += this.TAU * this.state[1]
        this.state[1] += this.TAU * xacc
        this.state[2] += this.TAU * this.state[3]
        this.state[3] += this.TAU * thetaacc
    }

    main() {
        let steps = 0
        let failures = 0

        let box = this.get_box()

        while (steps++ < this.MAX_STEPS && failures < this.MAX_FAILURES) {
            // const y = (Math.random() < this.prob_push_right(this.w.index([box]).valueOf()))
            const y = (Math.random() < this.prob_push_right(this.w[box]))
            this.e[box] = (1.0 - this.LAMBDAw) * (y - 0.5)
            this.xbar[box] = 1.0 - this.LAMBDAv
            // const e_o = sm.full([1], (1.0 - this.LAMBDAw) * (y - 0.5))
            // this.e = this.e.indexedAssign(e_o, [box])
            // const xbar_o = sm.full([1], (1.0 - this.LAMBDAv))
            // this.xbar = this.xbar.indexedAssign(xbar_o, [box])

            // const oldp = this.v.index([box]).valueOf()
            const oldp = box == -1 ? 0 : this.v[box]
            this.cart_pole(y)
            box = this.get_box()
            let r, p, failed

            if (box < 0) {
                failed = 1
                failures++
                console.log(`Trial ${failures} was ${steps} steps. \n`)
                steps = 0
                this.state = new Float32Array([0,0,0,0])
                r = -1.0
                p = 0.
            } else {
                failed = 0
                r = 0
                // p = this.v.index([box]).valueOf()
                p = this.v[box]
            }

            const rhat = r + this.GAMMA * p - oldp

            // this.w = sm.add(this.w, sm.mul(sm.scalar(this.ALPHA * rhat), this.e))
            // this.v = sm.add(this.v, sm.mul(sm.scalar(this.BETA * rhat), this.xbar))
            const weighted_e = this.e.map(x => x * (this.ALPHA * rhat))
            // this.w = this.w.map((x, idx) => x + weighted_e[idx])
            const weighted_xbar = this.xbar.map(x => x * (this.BETA * rhat))
            // this.v = this.v.map((x, idx) => x + weighted_xbar[idx])

            for(var i = 0; i < this.w.length; i++){
                this.w[i] = this.w[i] + weighted_e[i]
                this.v[i] = this.v[i] + weighted_xbar[i]
            }

            if (failed) {
                    // this.e = sm.full([this.N_BOXES], 0)
                    // this.xbar = sm.full([this.N_BOXES], 0)
                    this.e = new Array(this.N_BOXES).fill(0)
                    this.xbar = new Array(this.N_BOXES).fill(0)
            } else {
                this.e = this.e.map(x => x * this.LAMBDAw);
                this.xbar = this.xbar.map(x => x * this.LAMBDAv);
                // this.xbar = sm.mul(sm.scalar(this.LAMBDAv), this.xbar)
            }
        }
        if (failures ==this.MAX_FAILURES) {
            console.log(`Pole not balanced. Stopping after ${failures} failures.`)
        } else {
            console.log(`Pole balanced successfully for at least ${steps} steps\n`)
        }
    }

}


const p = new pole()
p.main()
