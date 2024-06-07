sqrt-lasso 
=========================

Overview
--------
***sqrt-lasso*** is a Matlab toolbox for solving square-root LASSO problem.
It implements various reconstruction algorithms and their comparisons on the synthetic data.

__Reconstruction Methods for square-root LASSO__
* Iteratively Reweighted Least squares algorithm [1]
* Proximal gradient and Newton methods [2]
* Smooth concomitant LASSO [3]
* Information-theoretic exact method [4]
* Frank-Wolfe algorithm with epigraphic lifting [1]  

References
----------
When you are using this code, please cite the paper

[1] Claudio Mayrink Verdun, Oleh Melnyk, Felix Krahmer, Peter Jung, 
Fast, noise-blind, and accurate: Tuning-free sparse regression with global linear convergence,
COLT 2024

This paper also provides an overview of other algorithms in Appendix F.

The rest of the algorithms are implemented based on the respective papers:

[2] Xinguo Li, Haoming Jiang, Jarvis Haupt, Raman Arora, Han Liu, Mingyi Hong, and Tuo Zhao.
On fast convergence of proximal algorithms for SQRT-Lasso optimization: Don't worry about its nonsmooth loss function. 
In Proceedings of The 35th Uncertainty in Artificial Intelligence Conference, volume 115 of Proceedings of Machine Learning Research, pages 49–59, Tel Aviv, Israel, 22–25 Jul 2020. PMLR.

[3] Eugene Ndiaye, Olivier Fercoq, Alexandre Gramfort, Vincent Lecl`ere, and Joseph Salmon. 
Efficient smoothed concomitant lasso estimation for high dimensional regression. 
Journal of Physics: Conference Series, 904:012006,  2017.

[4] Adrien Taylor and Yoel Drori. 
An optimal gradient method for smooth strongly convex minimization. 
Mathematical Programming, 199(1-2):557–594, 2023. 

[5] Tuo Zhao. Han Liu. Tong Zhang. 
Pathwise coordinate optimization for sparse learning: Algorithm and theory. 
Ann. Statist. 46 (1) 180 - 218, February 2018.

Dependencies
------------
Matlab implementation does not have dependencies.
For figure generation (see .tex files) tikz and pgfplots libraries are used. 

Directory structure
-------------------

File/Folder        | Purpose
------------------:| ---------------------------------------------------------------------
algorithms (dir)   | Contains implementations of algortihms for square-root LASSO. The folder "relative_iterate_error_timeout_stopping" contains implementations of the algorithms with the relative steps size and timeout stopping criteria. The folder "other" contains implementations of IRLS with fixed number of iterations.
figures (dir) 	   | Source code for the numerical experiments depicted in Figures 1 and 2.
s_ratio.m 		   | Helper function for SFR computation
COPYING            | License information
README.md          | This file

Feedback
--------
Your comments are welcome! This is the first version of the library and may
not be as robust or well documented as it should be. Please keep track of bugs
or missing/confusing instructions in Guthub Issues.
Alternatively, you might contact
[Oleh Melnyk](mailto:oleh.melnyk@tu-berlin.de)
or
[Claudio Mayrink Verdun](mailto:claudioverdun@seas.harvard.edu).

Legal Information & Credits
---------------------------
Copyright (c) 2024 Oleh Melnyk and Claudio Mayrink Verdun

This software was written by [Oleh Melnyk](https://olehmelnyk.xyz/) and [Claudio Mayrink Verdun](https://seas.harvard.edu/person/claudio-mayrink-verdun).
It was developed jointly at: 
* Institute of Mathematics, TU Berlin
* School of Computation, Information and Technology, TU Munich
* Harvard John A. Paulson School of Engineering and Applied Sciences
* Institute of Computational Biology, Helmholtz Munich

C. Mayrink Verdun is supported by the German Science Foundation (DFG) within the Gottfried Wilhelm Leibniz Prize under Grant BO 1734/20-1, under contract number PO-1347/3-2, under Emmy Noether junior research group KR 4512/1-1 and within Germany's Excellence Strategy EXC-2111 390814868. 
O. Melnyk acknowledges support from the Helmholtz Association under the contracts No.~ZT-I-0025 (Ptychography 4.0), No.~ZT-I-PF-4-018 (AsoftXm), No.~ZT-I-PF-5-28 (EDARTI), No.~ZT-I-PF-4-024 (BRLEMMM).

sqrt-lasso is free software. You can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version. If not stated otherwise, this applies to all files contained in this
package and its sub-directories.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA



