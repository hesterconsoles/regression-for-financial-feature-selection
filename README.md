# regression-for-financial-feature-selection
Feature selection under regression
1. Merge all the factor data from Fame-French, PS and HXZ and construct the lagged factors with different moneths.
2. Separate the data into training set and testing set, run the LASSO and Fama-French regressions for each assets. 
3. Average the sample-in and sample out errors for each assets.
4. Collect the factor frequency.
