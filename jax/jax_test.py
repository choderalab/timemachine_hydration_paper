import time
import numpy as vnp
from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np


import scipy.stats as stats

BOLTZMANN = 1.380658e-23
AVOGADRO = 6.0221367e23
RGAS = BOLTZMANN*AVOGADRO
BOLTZ = RGAS/1000
ONE_4PI_EPS0 = 138.935456
VIBRATIONAL_CONSTANT = 1302.79 # http://openmopac.net/manual/Hessian_Matrix.html

def harmonic_bond_nrg(
        coords,
        params):
    kb = params[0]
    b0 = params[1]

    src_idxs = [0, 0, 0, 0]
    dst_idxs = [1, 2, 3, 4]

    ci = coords[src_idxs]
    cj = coords[dst_idxs]

    dx = ci - cj
    dij = np.linalg.norm(dx, axis=1)


    # print("dij", dij, dij-b0)
    energy = np.sum(kb*np.power(dij - b0, 2)/2)

    return energy


def harmonic_bond_grad(coords, params):
    return jax.jacrev(harmonic_bond_nrg, argnums=(0,))

def analytic_grad(coords, params):
    kb = params[0]
    b0 = params[1]

    src_idxs = [0, 0, 0, 0]
    dst_idxs = [1, 2, 3, 4]

    ci = coords[src_idxs]
    cj = coords[dst_idxs]

    dx = ci - cj
    dij = np.linalg.norm(dx, axis=1)
    db = dij - b0

    lhs = np.expand_dims(kb*db/dij, axis=-1)
    rhs = dx
    src_grad = lhs * rhs
    dst_grad = -src_grad

    dx0 = np.sum(src_grad, axis=0, keepdims=True)
    res = np.concatenate([dx0, dst_grad], axis=0)

    return res

def nose_hoover_integrator(x0, params, dt=0.01, friction=1.0, temp=300.0):

    masses = np.array([12.0107, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

    num_atoms = len(masses)
    num_dims = 3

    dt = dt
    v_t = np.zeros((num_atoms, num_dims))

    friction = friction # dissipation speed (how fast we forget)
    temperature = temp           # temperature

    vscale = np.exp(-dt*friction)

    if friction == 0:
        fscale = dt
    else:
        fscale = (1-vscale)/friction
    kT = BOLTZ * temperature
    nscale = np.sqrt(kT*(1-vscale*vscale)) # noise scale
    # normal = tf.distributions.Normal(loc=0.0, scale=1.0)
    invMasses = (1.0/masses).reshape((-1, 1))
    sqrtInvMasses = np.sqrt(invMasses)

    coeff_a = vscale
    coeff_bs = fscale*invMasses
    coeff_cs = nscale*sqrtInvMasses

    start_time = time.time()

    agj = jax.jit(analytic_grad)
    r_t = x0
    z_t = 0
    f_t = -agj(r_t, params)

    Q = friction # or vscale?

    for step in range(10000):

        # f_t = -g
        r_dt = r_t + v_t*dt + (f_t*invMasses - z_t*v_t)*dt*dt/2
        v_dt_2 = v_t + dt/2*(f_t*invMasses - z_t*v_t)
        f_dt = -agj(r_dt, params)

        KE_dt = np.sum(0.5*v_t*v_t/invMasses)
        KE_dt_2 = np.sum(0.5*v_dt_2*v_dt_2/invMasses)

        z_dt_2 = z_t + (dt/(2*Q))*(KE_dt-kT*(3*num_atoms+1)/2)
        z_dt = z_dt_2 + (dt/(2*Q))*(KE_dt_2-kT*(3*num_atoms+1)/2)

        v_dt = (v_dt_2+(dt/2)*(f_dt))/(1+(dt/2)*z_dt)

        v_t = v_dt
        r_t = r_dt
        f_t = f_dt
        z_t = z_dt

        PE = harmonic_bond_nrg(x0, params)
        # KE = np.sum(0.5*v_t*v_t/invMasses)
        TE = (PE + KE_dt).aval

        print(step, "NH speed", (time.time() - start_time)/(step+1), np.amax(v_t).aval, "TE", TE, "Z_T", z_t.aval)

    return r_t

def integrator(x0, params, dt=0.01, friction=1.0, temp=300.0):

    masses = np.array([12.0107, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

    num_atoms = len(masses)
    num_dims = 3

    dt = dt
    v_t = np.zeros((num_atoms, num_dims))

    friction = friction # dissipation speed (how fast we forget)
    temperature = temp           # temperature

    vscale = np.exp(-dt*friction)

    if friction == 0:
        fscale = dt
    else:
        fscale = (1-vscale)/friction
    kT = BOLTZ * temperature
    nscale = np.sqrt(kT*(1-vscale*vscale)) # noise scale
    # normal = tf.distributions.Normal(loc=0.0, scale=1.0)
    invMasses = (1.0/masses).reshape((-1, 1))
    sqrtInvMasses = np.sqrt(invMasses)

    coeff_a = vscale
    coeff_bs = fscale*invMasses
    coeff_cs = nscale*sqrtInvMasses

    start_time = time.time()

    agj = jax.jit(analytic_grad)
    for step in range(1000):

        g = agj(x0, params)

        # random normal
        noise = vnp.random.normal(size=(num_atoms, num_dims)).astype(x0.dtype)

        # truncated normal
        mu, sigma = 0, 1.0
        lower, upper = -0.5*sigma, 0.5*sigma
        X = stats.truncnorm(
            (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        noise = X.rvs(num_atoms*num_dims).reshape((num_atoms, num_dims))
        nscale = 0
        print("noise max/min mean", np.amax(noise), np.amin(noise), np.mean(noise))

        v_t = vscale*v_t - fscale*invMasses*g + nscale*sqrtInvMasses*noise


        # nose-hoover
        dx = v_t * dt

        PE = harmonic_bond_nrg(x0, params)
        KE = np.sum(0.5*v_t*v_t/invMasses)
        TE = (PE + KE).aval


        print(step, "speed", (time.time() - start_time)/(step+1), np.amax(v_t).aval, "TE", TE)
        x0 += dx

    print(coeff_a, coeff_bs, coeff_cs)

    return x0

if __name__ == "__main__":

    x = np.array([
        [-0.0036,  0.0222,  0.0912],
        [-0.0162, -0.8092,  0.7960],
        [ 0.9404,  0.0222, -0.4538],
        [-0.1092,  0.9610,  0.6348],
        [-0.8292, -0.0852, -0.6123]
    ], dtype=np.float64)

    theta = np.array([5000.0, 1.15], dtype=np.float64)


    a = harmonic_bond_grad(x, theta)(x, theta)[0]
    b = analytic_grad(x, theta)

    print(a - b)
    # assert np.max(a-b) < 1e-7

    dxdp = jax.jacfwd(nose_hoover_integrator, argnums=(1,))
    res = dxdp(x, theta)[0]
    print(res, np.amax(res), np.amin(res))