import * as sm from "@shumai/shumai"

function maxOf(tensor) {
    let max = 0
    for (let i = 0; i < tensor.shape[0]; i++) {
        max = (tensor.index([i]) > max) ? tensor.index([i]) : max
    }
    return max          
}

function argMax(tensor) {
    let max = 0
    let ind = 0
    for (let i = 0; i < tensor.shape[0]; i++) {
        max = (tensor.index([i]) > max) ? tensor.index([i]) : max
        ind = (tensor.index([i]) > max) ? i : ind
    }
    return ind          
}

function one_step_lookahead(s, V, rewards, p_h, discount_factor) : sm.Tensor {

    let A = sm.full([101], 0)
    for (let a = 1; a <= Math.min(s, 100 - s); a++) {
        const new_val = p_h * (rewards[s+a] + V[s+a]*discount_factor) + (1-p_h) * (rewards[s-a] + V[s-a]*discount_factor)
        const o = sm.full([1], new_val)
        A = A.indexedAssign(o, [a])
    }

    return A
}

function value_iteration_for_gamblers(p_h = 0.5, theta = 0.0001, discount_factor = 1.0) {
    const B = sm.full([101], 0)
    const o = sm.full([1], 1)
    let rewards = B.indexedAssign(o, [100])

    let V = sm.full([101], 0)
    
    while (true) {
        let delta = 0
        for (let s = 1; s < 100; s++) {
            const A_tensor = one_step_lookahead(s, V, rewards, p_h, discount_factor)
            const best_action_value = maxOf(A_tensor)
            delta = Math.max(delta, Math.abs(best_action_value - V.index([s])))
            const best_action_value_tensor = sm.full([1], best_action_value)
            V = V.indexedAssign(best_action_value_tensor, [s])
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