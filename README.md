# FRACPython

Python implementation of the Salanie-Wolak  FRAC method to estimate  quasi-linear random coefficients models.

## inputs
The model must have the form  <img src="https://render.githubusercontent.com/render/math?math=G^\ast(Y, E_\varepsilon A^\ast(Y, \eta + f_1(Y)\beta, \varepsilon))=0">, where 
we assume that <img src="https://render.githubusercontent.com/render/math?math=E(\eta \vert Z)=0"> and <img src="https://render.githubusercontent.com/render/math?math=\varepsilon">
has mean zero and a variance-covariance matrix <img src="https://render.githubusercontent.com/render/math?math=\Sigma">. 

FRAC estimates <img src="https://render.githubusercontent.com/render/math?math=\beta"> and <img src="https://render.githubusercontent.com/render/math?math=\Sigma">
using 2SLS, corrected or not.


The user must input:
1. the data <img src="https://render.githubusercontent.com/render/math?math=Y">
2. the function <img src="https://render.githubusercontent.com/render/math?math=f_1(Y)">
3. the function <img src="https://render.githubusercontent.com/render/math?math=f_0(Y)"> such that 
   <img src="https://render.githubusercontent.com/render/math?math=G^\ast(Y, A^\ast(Y, f_0(Y), 0))=0">
4. either the artificial regressors <img src="https://render.githubusercontent.com/render/math?math=K(Y)">, 
   or the functions <img src="https://render.githubusercontent.com/render/math?math=d(Y)=A^\ast_2(Y, f_0(Y), 0)"> and 
   <img src="https://render.githubusercontent.com/render/math?math=t(Y) = \frac{\partial^2 A^\ast}{\partial \varepsilon \partial \varepsilon^\prime}(Y, f_0(Y), 0),">
  which allow the program to evaluate the artificial regressors.
   
If using corrected 2SLS, the user must also choose a distributional form for <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"> and input

5. the function <img src="https://render.githubusercontent.com/render/math?math=f_\infty(Y,\Sigma)"> such that <img src="https://render.githubusercontent.com/render/math?math=G^\ast(Y, E_\varepsilon A^\ast(Y, f_\infty(Y,\Sigma), \varepsilon))=0"> 
   when
the variance-covariance of <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"> is <img src="https://render.githubusercontent.com/render/math?math=\Sigma">. 
   
The program then does 2SLS with the corrected <img src="https://render.githubusercontent.com/render/math?math=f_0(Y)+f_\infty(Y,\hat{\Sigma}_2)-f_2(Y)\hat{\Sigma}_2">, where 
<img src="https://render.githubusercontent.com/render/math?math=\hat{\Sigma}_2"> is the uncorrected 2SLS estimator of the variance-covariance.

  

   
