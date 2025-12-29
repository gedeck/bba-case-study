Reviewer: 1
Comments to the Author
The paper shows that BBA leads to more stable estimates of regression parameters in two case studies when compared to OLS and several other approaches to bootstrapping a DoE.

There are two issues with the paper that I would like to see resolved. One is that the examples in the case study are just CRDs  that have been replicated in an intentionally unbalanced way. The description of BBA for a "simple case" on page 3 is actually much more complicated than how it is applied in the case studies since the case studies are only replicating at the "L" level from the page 4 detailed description. Either simplifying section 3 or adding a variance component/blocking effect to the simulation would harmonize these two aspects of the paper.

> Simplify approach

The second issue is that the most striking result shown in the paper is the decreasing BBA standard deviations as imbalance increases. This should be explained and explored in more detail. It looks like the bootstrap standard deviation is becoming too small to be useful because there are not enough replicates to resample. This would make the bootstrap distribution too discrete with variability that is too low to be used as the basis for inference. Some statement about the implications of this decreasing variability would be useful.

> Had statement, but extra simulation can shed more light on it.

If the goal of the study is to test factors then reporting confidence interval coverage probabilities would be more direct. On the other hand, the intention of a study like the piston one is more likely to obtain a predictive model to find factor settings that achieve some sort of response goals. Since these are simulations, we should be able to know something 'true' about the way the data are simulated and that should be the basis of evaluation, either from 'true' values of the regression coefficients and/or the the 'true' response surface. In the case of the BBA variability going down with increased imbalance, we can't tell if this is a good thing in that the methods is getting at the truth much more efficiently, or if this is a bad thing in that the distance to the truth from the estimate is getting higher despite having much less bootstrap-to-bootstrap variation within each of the 10000 simulation reps.

> Check if the actual value is within the standard deviation of the estimate as a function of imbalance. Compare different bootstrap approaches.

Reviewer: 2

Comments to the Author
1. The paper presents a novel resampling technique called Befitting Bootstrap Analysis (BBA) and studies its performance when analyzing data from Central Composite Design (CCD) - type experiments.

2. The main appeal of BBA appears to be reduction of the estimated standard error (relative to competing techniques) as the data become more imbalanced. However, the authors do not explore in sufficient detail the properties of estimates obtained via BBA, such as bias / variance, and do not explain the derivative properties of the standard error estimates, such as ability to produce confidence bounds for the effects of interest with nominal probabilities of coverage.  Such analysis is essential for presenting BBA as a viable competitor to other techniques and for specifying its domain of dominance.

3. Sec. 4.1.

(a) You may want to relate it better to Sec. 3. For example, it may not be obvious to readers what variables play the role of Z1, Z2, Z3. I assumed that (Z1, Z2, Z3) = (x1, x2, x3).  However, it is not clear whether the results would be different if we used a different combination, e.g., (Z1, Z2, Z3) = (x3, x2, x1)?

> nomenclature

(b) In the formula for k (middle of p. 7), it is not clear what are the constraints for f. You mentioned that varying f from n to 10 works well - but was any varying of f used in the examples you provided?  For example, what was f in Figures 4, 5?

> Improve explanation

4. It would be useful to have the equations numbered.

Grammar & Typos

p. 2, line 25. training instead of raining

p. 2, l. -2. the paper instead of that paper

p. 3, l. 13. variability of parameter estimates

p. 4, l. -17. resampling with replacement I (instead of L)