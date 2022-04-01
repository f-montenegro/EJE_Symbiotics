from C2_Execution.ScoringScale import *

##### Databases
path = "~felip/OneDrive/ProjectsGitHub/EJE_Symbiotics/Data"
data_FIR = pd.read_excel(f"{path}/Financial Institutions Ratio.xlsx", sheet_name="Brut_Data")
data_CRP = pd.read_excel(f"{path}/Credit Rating Published.xlsx", sheet_name="Brut_Data")
data_trend = pd.read_excel(f"{path}/Trend.xlsx", sheet_name="Brut_Data")
data_ext = pd.read_excel(f"{path}/External Transition Matrix.xlsx", sheet_name="Emerging_Markets")
data_ref = pd.read_excel(f"{path}/Financial Institutions Referentiels.xlsx")

##### Portfolio Data
data_portfolio_LUEIBF = pd.read_excel(f"{path}/LU-EIBF-Portfolio-2022-04-08.xlsx", sheet_name="Brut_Data")
data_portfolio_LUFNTC = pd.read_excel(f"{path}/LU-FNTC-Portfolio.xlsx", sheet_name="Input")
data_portfolio_LUDUAL = pd.read_excel(f"{path}/LU-DUAL-Portfolio.xlsx", sheet_name="Input")
data_portfolio_LUREG = pd.read_excel(f"{path}/LU-REG-Portfolio-2022-08-01.xlsx", sheet_name="Portfolio")
data_portfolio_LUGMF = pd.read_excel(f"{path}/LU-GMF-Portfolio-2022-08-01.xlsx", sheet_name="Portfolio")
data_portfolio_LUAAIF = pd.read_excel(f"{path}/LU-AAIF-Portfolio-2022-08-01.xlsx", sheet_name="Portfolio")

portfolios = {"LU-EIBF": [data_portfolio_LUEIBF, "USD"],
              "LU-FNTC": [data_portfolio_LUFNTC, "USD"],
              "LU-DUAL": [data_portfolio_LUDUAL, "EUR"],
              "LU-REG": [data_portfolio_LUREG, "USD"],
              "LU-GMF": [data_portfolio_LUGMF, "USD"],
              "LU-AAIF": [data_portfolio_LUAAIF, "USD"]}

##### Parameters Correlation Matrix
cluster_method = ["Region", "Country"]
ratio_ind = ["R600", "R622", "R625", "R626", "R628", "R629", "T205"]

##### Parameters CreditMetrics
shift_years = 1  # Years to force when no maturity is provided
M = 5000         # Number of Monte-Carlo Sim
r = 0            # Risk-free rate
T = 1            # Period
RR = 0.78        # Recovery Rate

##########
# The two next lines don't need to be modified by the user
data_TransitionMatrix = {"CRP": data_CRP, "trend": data_trend}
data_CorrelationMatrix = data_FIR
##########
