import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/cookie_cats.csv")


example_obs = data.loc[:, ['version', 'retention_1']]
example_obs2 = data.loc[:, ['version', 'retention_7']]

ret30 = example_obs.loc[example_obs["version"]=="gate_30", "retention_1"]
ret40 = example_obs.loc[example_obs["version"]=="gate_40", "retention_1"]
ret302 = example_obs2.loc[example_obs["version"]=="gate_30", "retention_7"]
ret402 = example_obs2.loc[example_obs["version"]=="gate_40", "retention_7"]

count = [ret40.sum(), ret30.sum()]
count2 = [ret402.sum(), ret302.sum()]

n = [ret40.size, ret30.size]
n = [ret402.size, ret302.size]

stat, pval = proportions_ztest(count, n)
stat2, pval2 = proportions_ztest(count2, n)
print("z-statistic:", stat)
print("p-value:", pval)
print("z-statistic:", stat2)
print("p-value:", pval2)
