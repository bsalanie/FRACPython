# FRACPython

Python implementation of the Salanie-Wolak  FRAC method to estimate  quasi-linear random coefficients models.

### inputs
The model must have the form  $E_\varepsilon A^\ast(Y, \eta + f_1(Y)\beta, \varepsilon)=0$, where 
we assume that
* $\eta$ and $A^\ast$ have the same dimension
* $E(\eta \vert Z)=0$ 
* $\varepsilon$
has mean zero.

Our aim is to estimate $\beta$ and $\Sigma$.

The user must specify 
1. the data $Y$ as a matrix
2. the function $f_1$ (or its values in the sample)
3. and the function $A^\ast$;

and, optionally:
* the function $f_0$ such that $A^\ast(Y,f_0(Y),0)=0$, or at least its values in the sample
* the artificial regressors $K$ given by
$$
A^\ast_2(Y, f_0(Y),0)  K = A^\ast_{33}(Y, f_0(Y),0)/2.
$$
* the instruments $Z$
* if using corrected 2SLS, the user must also choose a distributional form for $\varepsilon$ and provide the 
the function $f_\infty(Y,\Sigma)$  such that $E_\varepsilon A^\ast(Y, E_\varepsilon A^\ast(Y, f_\infty(Y,\Sigma), \varepsilon))=0$
   when the variance-covariance of  $\varepsilon$ is  $\Sigma$.
  
### method 
The program does 2SLS of $f_0(Y)$ on $f_1(Y)$ and $K$ with instruments $Z$; the FRAC estimators $\hat{\beta}_2$ 
and $\hat{\Sigma}_2$ are the estimated coefficients of $f_1(Y)$ and of $K$. 

The corrected version replaces $f_0(Y)$
with
$$
f_\infty(Y, \hat{\Sigma}_2) +K\hat{\Sigma}_2
$$
and re-runs the 2SLS regression.
