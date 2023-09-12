# BEM
Use boundary element method to find the resonances in homogeneous and inhomogeneous 2D cavities with arbitrary shape.

### 2023.8.25
Nothing is done right now. 
Start with resonance in a circular cavity for TM-mode.

### 2023.9.12
Successfully calculated the resonance in a circular cavity for TM-mode.
The program "compute_B_C_approx.py" compute the determinant of 2N by 2N BEM matrix (B,C) for several k and save the results into a 3d numpy.ndarry with elements \[Re(k), Im(k), det(M)\].
The local minimum of the determiant is the wavenumber of resonance.
Using "compute_wfunc.py" to compute the spatial intensity pattern for each resonance.

### 2023.9.12
Added stadium-shaped cavity in "domain.py".

Next step is making a general class for those cavities defined in polar coordinate and reducing the computational consuming by using the symmetry.
