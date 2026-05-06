# CD-spectra-deconvolution
This repository is made to host a script that allows the deconvolution of nucleic acids CD spectra into a composition in percentage of elementary structure.
## Input
This script take on input one excel file, containing in its first column the wavelength, then the CD values of the reference spectra and afterwards values of spectra to deconvolute. It has to be mentionned in the input what are the name of the spetra to exclude from the references (name are to be used as headers of the column) and which spectra you want to deconvolute.
## What does it does
This code deconvolute the target spectrum into elementary components composition. This deconvolution follow several steps.
### First step
The first step consist in the selection of the most relevant reference spectra, by evaluating it through a stepwise forward progress selection. Every possible parameter for the deconvolution is evaluated trhough a non-negative linear regression (NNLS). The selection is made using a criter that can be choosen by the user. The available criterion are BIC, AIC and BIC_RENFORCED.
### Second step (optional)
Once the criter has been selected, a bootstrap is led to stabilize the corefficients and estimate the uncertainties can be done.
### Third step
Runs a diagnostic on residuals, in order to detect if the fitting respects linear models rules or not.
### Fourth step (optional)
Depending of the parameter from the user, the code can evaluate a "semi-non-linear" NNLS using non-linear terms established using quadratic and interaction combinations of structural references. This can be done depending on the results from the previous diagnostics, or force.
## Output
This scripts gives as an output an excel file containing all the fitting parameters, on different pages, aswell as graphs containing the fits over the whole spectrum, the bootstrap fits with bands containing the 95% IC, and in a separate folder all the non linearity diagnostic results, consisting of 2 png wth the QQ plots and acf, and a csv file containing the results from Durbin Watson and Breusch Pagan test.
