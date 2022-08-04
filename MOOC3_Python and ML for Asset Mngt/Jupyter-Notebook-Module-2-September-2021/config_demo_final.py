#Configuration File

#Do not change, these are ipython notebook demonstration

#Path for data
dataPath = 'Data/Data2016.csv'

#Define the factor names
#factorName = ['World Equities Excess Return','Treasury Bond Excess Return','Default Risk Premium','Inflation Protection','Currency Protection']
factorName = ['World Equities', '10-year US Treasuries', 'High Yield', 'Inflation Protection', 'Currency Protection']

#Names of assets
#assetName = ['US Equities Excess Return','Real Estate Excess Return','Commodities']
assetName = ['SP500 Total Return','International Equity','U.S. Treasury 20 years', 'Corporate Bond','Real Estate', 'Commodity', 'TIPS']

#Name of date column
dateName = 'Date'

#User Analysis Section.  Change the variables in this section to run user specific analysis

#isDemo is a boolean variable, set to True if the user wants to run custom analysis
isDemo = False

#dataPathUser: Path to User Defined Data
dataPathUser = 'Data/Data2016.csv'

#factorNameUser: List, defines the factors
factorNameUser = ['World Equities', '10-year US Treasuries', 'High Yield', 'Inflation Protection', 'Currency Protection']

#assetNameUser: List, defines the asset to be used
assetNameUser = 'Commodity'

#dateName: string, date column
dateNameUser = 'Date'

#lambdaHatUser: float, penalty term of user define LASSO regression
lambdaHatUser = .00005

#Start and End Dates for the Analysis
startDateUser = '1997-03-01'
endDateUser = '2014-12-01'

#Stuff for optional part of user section

#Best Subset Regression Related
maxVarsUser = 1

#Elastic Net Related
numL1RatioUser = 10
numAlphasUser = 20
