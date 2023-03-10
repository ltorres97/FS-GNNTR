from pandas import DataFrame
from matplotlib import pyplot
from scipy.stats import normaltest, ttest_ind

####### Statistical Significance Analysis of Experimental Results #######

#FS-GNNTR
result1 = [0.7512, 0.7478, 0.7502, 0.7487, 0.7466, 0.7499, 0.7478, 0.7477, 0.7482, 0.7501, 0.748, 0.7475, 0.7524, 0.7444, 0.7515, 0.7452, 0.7482, 0.7468, 0.7489, 0.7503, 0.751, 0.7496, 0.7488, 0.7492, 0.7484, 0.7452, 0.747, 0.7481, 0.7476, 0.7459]

#Baseline
result2 = [0.7421, 0.7421, 0.7326, 0.7444, 0.7438, 0.7301, 0.7392, 0.7487, 0.7338, 0.7334, 0.7388, 0.74, 0.7455, 0.737, 0.7377, 0.738, 0.7323, 0.7321, 0.7436, 0.7526, 0.7442, 0.7442, 0.7455, 0.7459, 0.7432, 0.7434, 0.744, 0.7401, 0.7219, 0.7354]

results = DataFrame()
results['A'] = result1
results['B'] = result2

# Descriptive stats
print(results.describe())

# Box and whisker plot
results.boxplot()
pyplot.show()

# Histogram
results.hist()
pyplot.show()

#Normality test
value1, p1 = normaltest(result1)
value2, p2 = normaltest(result2)
print(p1, value1)
print(p2, value2)

if p1 >= 0.05:
    print('It is likely that result1 is normal')
else:
    print('It is unlikely that result1 is normal')
 
if p2 >= 0.05:
    print('It is likely that result2 is normal')
else:
    print('It is unlikely that result1 is normal')
  
# Significance Test - T-test for independent samples - Welch’s t-test - different variance

value, pvalue = ttest_ind(result1, result2, equal_var=False)
print(value, pvalue)
if pvalue > 0.05:
    print('Samples are likely drawn from the same distributions (fail to reject H0)')
else:
    print('Samples are likely drawn from different distributions (reject H0)')

