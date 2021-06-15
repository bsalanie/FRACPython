# FRACPython

Python implementation of the Salanie-Wolak  FRAC method to estimate  quasi-linear random coefficients models.

## inputs
The model must have the form  <img src="https://render.githubusercontent.com/render/math?math=G^\ast(Y, E_\varepsilon A^\ast(Y, \eta + f_1(Y)\beta, \varepsilon))=0">. 

The user must input:
1. the data <img src="https://render.githubusercontent.com/render/math?math=Y">
2. the function $f_1(Y)$
3. the function $f_0(Y)$ such that $G^\ast(Y, A^\ast(Y, f_0(Y), 0))=0$
4. either the artificial regressors $K(Y)$, or the functions $d(Y)=A^\ast_2(Y, f_0(Y), 0)$ and 
   $$
   t(Y) = \frac{\partial^2 A^\ast}{\partial \varepsilon \partial \varepsilon^\prime}(Y, f_0(Y), 0),
   $$
  which allow the program to evaluate the artificial regressors.
   
If using corrected 2SLS, the user must also choose a distributional form for $\varepsilon$ and input 
5. the function $f_\infty(Y,\Sigma)$ such that $G^\ast(Y, E_\varepsilon A^\ast(Y, f_\infty(Y,\Sigma), \varepsilon))=0$ when
the variance-covariance of $\varepsilon$ is $\Sigma$. 
   
The program then does 2SLS with the corrected $f_0(Y)+f_\infty(Y,\hat{\Sigma}_2)-f_2(Y)\hat{\Sigma}_2$, where 
$\hat{\Sigma}_2$ comes from the uncorrected 2SLS.

  

   
