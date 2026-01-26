This document captures the process of iterating towards the optimal network architecture. 

#Attempts and conclusions - focusing on the digit1hot representation (90)
##A simple MLP
Variation on the form: [90, 900, 900, 900, 900, 300, 200, 20, 1] ~6M params.
train gets to 95+%, eval never exceed 20%. The issue seems to be memorization, since regularization was applied.

##Tower per 3 digits
* Small mlp (tower) which extratcs representations from every 3 digits (mils, thous, ones) - same for all three layers
* Tower  outputs concatenated and passed through another MLP head to produce prediction
*  tower: [30, 768, 1024, 512] , head: [1536, 512, 128, 1]  ~2M params
- train for high precision, try dropout if needed

##Tower per digit
##Tower per 1/3 digits + simple MLP for extra sentence length (spaces, and) -> all concatenated into input to final prediction mlp