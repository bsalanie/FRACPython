# FRACPython

Python implementation of the Salanie-Wolak  FRAC method to estimate  quasi-linear random coefficients models.

### inputs
The model must have the form  $E_\varepsilon A^\ast(Y, \eta + f_1(Y)\beta, \varepsilon)=0$, where 
we assume that
* $\eta$ and $A^\ast$ have the same dimension $J$
* $E(\eta_j\vert Z_j)=0$ for all $j$ 
* $\beta$ has $p$ elements and $f_1(Y)$ is $(J,p)$
* $\varepsilon$ has $M$ elements, a 
 mean zero and a variance $\Sigma$.

 Observations are $t=1,\ldots,T$ and the above holds for any $t$.

Our aim is to estimate $\beta$ and $\Sigma$.

The user must specify 
1. the data $Y$ as a Numpy  array $(T,J,n_Y)$
2. the function that computes $f_1$ for each observation:  $f_1(Y_t,a_t)$ (or its values $(f_{1,t})$ in the sample, an array  $(T,J,p)$); where $a_t$ represents all the other arguments needed;
3.  the function that computes $A^\ast$ for each  observation
4. the instruments $Z$, a Numpy array $(T,J,n_Z)$;

and, optionally:
5. the function $f_0(Y_t,a_t)$ such that $A^\ast(Y,f_0(Y),0)=0$, or at least its values in the sample as a $(T,J)$ array
6. the artificial regressors $K$ given by
$$
A^\ast_2(Y, f_0(Y),0)  K = A^\ast_{33}(Y, f_0(Y),0)/2.
$$
7. if using corrected 2SLS, the user must also choose a distributional form for $\varepsilon$ and provide the  function $f_\infty(Y_t,\Sigma,a_t)$  such that $E_\varepsilon A^\ast(Y, E_\varepsilon A^\ast(Y, f_\infty(Y,\Sigma), \varepsilon))=0$
   when the variance-covariance of  $\varepsilon$ is  $\Sigma$.
  
### method 
The program does 2SLS of $f_{0,tj}$ on $f_{1,tj}$ and $K_{tj}$ with instruments $Z_{tj}$; the FRAC estimators $\hat{\beta}_2$ 
and $\hat{\Sigma}_2$ are the estimated coefficients of $f_{1,tj}$ and of $K_{tj}$. 

The corrected version replaces $f_{0,tj}$ with
$$
f_{\infty,j}(Y_t, \hat{\Sigma}_2,a_t) +K_t\hat{\Sigma}_2
$$
and re-runs the 2SLS regression.


### solving for $f_0$ and $f_\infty$
For $f_0$ we  have $A(f)=A^\ast(Y, f, 0)$ and $A^\prime(f)=A^\ast_2(Y, f,0)$.

For $f_\infty$, given a parameter-free distribution $F$  for $v$, we draw $v^{(s)}, s=1,\ldots,S$ from $F$ and we compute
$$
A(f) =\frac{1}{S}\sum_{s=1}^S A^\ast(Y, f, Lv^{(s)})
$$
and 
$$
A^\prime(f) =\frac{1}{S}\sum_{s=1}^S A^\ast_2(Y, f, Lv^{(s)}).
$$
#### using `scipy`
Or use `scipy.optimize.root(vfun, vx0, jac=True)` where `vfun` returns $A$ and $A^\prime$.

#### or Newton

We iterate Newton:

* solve $A^\prime(f^{(k+1)}) d = -A(f^{(k)})$ 
* make $f^{(k+1)}=f^{(k)}+d$.

For $f_0$ we start from a reasonable value; for $f_\infty$ we start from , starting from $f^{(0)}=f_0(Y)-K\Sigma$.



