import * as sm from "@shumai/shumai"

function maxOf(vals) {
    let max = 0
    // const vals = tensor.valueOf()
    const lent = vals.length
    for (let i = 0; i < lent; i++) {
        max = (vals[i] > max) ? vals[i] : max
    }
    return max      
}

function argMax(array) {
    // let array = tensor.valueOf()
    return [].map.call(array, (x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
  }

function one_step_lookahead(s, V, rewards, p_h, discount_factor) {
    /*
        Helper function to calculate the value for all action in a given state.
        
        Args:
            s: The gambler’s capital. Integer.
            V: The vector that contains values at each state. 
            rewards: The reward vector.
            p_h: probability that a coin comes up heads 
                        
        Returns:
            A vector containing the expected value of each action. 
            Its length equals to the number of actions.
    */

    const A = new Float32Array(101).fill(0)
    for (let a = 1; a <= Math.min(s, 100 - s); a++) {
        const new_val = p_h * (rewards[s+a] + V[s+a]*discount_factor) + (1-p_h) * (rewards[s-a] + V[s-a]*discount_factor)
        A[a] = new_val
    }

    return A
    // const A_tensor = sm.tensor(A)
    // return A_tensor
}

function value_iteration_for_gamblers(p_h = 0.5, theta = 0.0001, discount_factor = 1.0) {
    /*
        Args: 
            p_h: probability that a coin comes up heads 
    */

    const rewards = new Float32Array(101).fill(0)
    rewards[100] = 1

    let V = new Float32Array(101).fill(0)
    
    while (true) {
        let delta = 0
        for (let s = 1; s < 100; s++) {
            let A_tensor = one_step_lookahead(s, V, rewards, p_h, discount_factor)
            const best_action_value = maxOf(A_tensor)
            delta = Math.max(delta, Math.abs(best_action_value - V[s]))
            V[s] = best_action_value
        }
        if (delta < theta) {
            break
        }
    } 

    // return V
    const policy = new Float32Array(100).fill(0)
    for (let s = 1; s < 100; s++) {
        let A_tensor = one_step_lookahead(s, V, rewards, p_h, discount_factor)
        const best_action = argMax(A_tensor)
        policy[s] = best_action
    }
    return policy, V
}

const t0 = performance.now()
let policy, V = value_iteration_for_gamblers(0.25)
console.log((performance.now() - t0) / 1e3, 'seconds')