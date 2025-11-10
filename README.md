# pyGBIS-
Bayesian Inference of a point Compound Dislocation model based on forward model of Nikkhoo, M., Walter, T. R., Lundgren, P. R., Prats-Iraola, P. (2016)
Now with Okada model 


Run in order input is 3 unit vector files in geocoded tif format,  *.geo.E.tif *geo.N.tif, *geo.U.tif take from COMET LICSAR portal 
https://comet.nerc.ac.uk/comet-lics-portal/

run in order 
step01 - Converts from tif to npy files clips and downsamples - clips is 3 times the radius of the circle used for down sampling 
step02 - Calculated sill nugget and range 
step03 - Once edited to own data runs baysian inference. 

for step03:
change variables in the if __name__ == __main__ to use own data 

Inputs are a 
dict of initial guesses for values 'custom_initial'
dict of custom priors 'custom_priors'
dict of learning rates 'custom_learning_rates'
dict of max_step_sizes 'Max_step_sizes'

The learning rates are the initial learning rate for each value, which is adaptively adjusted to produce a uniform change across the params i.e the change in each param causes a similar amount of change in the output model. 

The maximum step size is there so that this adaptive step size does not blow up to very large values. 
<img width="1995" height="1425" alt="Figure_2" src="https://github.com/user-attachments/assets/284e7c63-4aae-46d0-8d3d-afdce6eec0f3" />
<img width="1500" height="800" alt="Figure_3" src="https://github.com/user-attachments/assets/dddeec4e-e678-4118-ac53-59f7111c674a" />
<img width="2053" height="1322" alt="Figure_4" src="https://github.com/user-attachments/assets/e5ede3e3-22ed-4c20-8a7b-7e2dee67987d" />
<img width="2213" height="1359" alt="Figure_7" src="https://github.com/user-attachments/assets/1b0dedc1-0848-4154-87df-97a828d022cd" />
<img width="1200" height="800" alt="Figure_5" src="https://github.com/user-attachments/assets/f6848020-2f6d-43a8-95df-f05de696949e" />
<img width="2035" height="924" alt="Figure_6" src="https://github.com/user-attachments/assets/74230810-933d-4565-a2bd-1914ecd09152" />
<img width="1488" height="595" alt="Screenshot from 2025-10-17 16-54-14" src="https://github.com/user-attachments/assets/1e67f54f-5c14-4815-b284-31ba7dd50a94" />
<img width="1501" height="599" alt="Screenshot from 2025-10-17 16-54-45" src="https://github.com/user-attachments/assets/d4b0e164-669b-473c-b000-d9643b5f77d9" />

# suggested Additions 
Add the ability to add in multiple datasets for the inversion i.e can do Asc and Dsc at same time.
Add more models

