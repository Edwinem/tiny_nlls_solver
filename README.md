# Tiny MinimizerMethod


This is a small non-liner least squares solver. I created it cause I found myself creating an ugly hack version when trying to implement dense visual slam problems.
A good amount of this project derives from the tiny solver in Ceres.

## To Do


- [ ] Maybe add ability to add weights. Currently user can do it manually in the cost function
- [ ] Add the ability for the cost function to create the Hessian + grad (normal equations)
 directly. Many of Direct slam methods like DSO do this with SIMD.
- [ ] Gauss newton
- [ ] Dogleg
- [ ] PCG
- [ ] Add increase in cost function as a failure to LM
- [ ] LM calls the cost function twice in one iteration. This could be insanely
expensive depending on how the residuals are computed.


## Credits

Ceres(http://ceres-solver.org/index.html)

The base of this tiny solver comes from the tiny solver in Ceres. Also I took their cost functions design and autodiff to make certain things easier.