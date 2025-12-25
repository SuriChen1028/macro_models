The algorithm

given p, w

solve a $\bar{v}(\phi, p, w)$ in the following way:
- guess a v_init as a function of $\phi, p, w$, in the meantime $\phi$ is also evenly distributed.
- solve for $v_new = Tv := max \pi(\phi, p, w) + \max\{ 0, \int v(\phi, p, w) d \mu(\phi) \} $ for every $\phi$ in the state space
    - given $v$ and $\phi$
    - update to $A_{t+1}\phi_t$, and extrapolate to $v(A_{t+1}\phi_t)$ and compute the mean
    - compute $Tv$ and $||v-Tv||$, iterate if the norm is larger than the tolerance level
- for each $\phi$, interate until $|| v - Tv || < tol$, and this gives the $\bar{v}(\phi, p, w)$.
- And follow the iteration rule, the distribution of $\phi$ is also the same as the previous period
- Now, extrapolate $v$ to the space of $\gamma$ which is the distribution of entry firms. 
- compute $E[v(\phi^\prime, p, w) \mid \phi_e] - c_e$. 

If entry value > 0, the price is too high, adjust upper price to mid point. solve for $\var(\phi, new mid, w)

If entry value < 0, the price is too low, adjust lower price to mid point, solve for a new value function.

Stop until the the two points converges.

The mid point is therefore the equilibrium price.

Next step to get the stationary distribution. given $p^*$ and $\bar{v}(\phi, p^*, w)$, simulate over a long period of time. so that the sample $\phi$'s are the the stationary ones.


[[run.jl]] runs the discrete-time version. Essential solvers are in the [[src/model.jl]].

[[hopenhayn.jl]] runs the continuous time problem.