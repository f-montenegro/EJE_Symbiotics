from C0_Library.config import *


class CreditMetrics:
    def __init__(self, CorrelationMatrix_dict, TransitionMatrix_dict,
                 data_portfolio, name_portfolio, currency_portfolio,
                 data_ref, scoring_scale, shift_years,
                 M, r, RR, T):

        """
        CorrelationMatrix_dict -> dictionary containing all correlation matrix for ONE portfolio
        TransitionMatrix_dict -> dictionary containing all transition matrix (same for every portfolio)
        data_portfolio -> portfolio information
        data_ref -> Investee Types data
        scoring_scale -> table used in the rating conversion (symbols to number)
        M -> number of Monte-Carlo interactions
        r -> risk free rate over the valuation period
        RR -> Recover Rate: if default, what proportion of the amount will we recover?
        T -> valuation period length
        """

        self.CorrelationMatrix_dict = CorrelationMatrix_dict
        self.TransitionMatrix_dict = TransitionMatrix_dict
        self.data_portfolio = data_portfolio
        self.name_portfolio = name_portfolio
        self.currency_portfolio = currency_portfolio
        self.data_ref = data_ref
        self.scoring_scale = scoring_scale
        self.shift_years = shift_years
        self.M = M
        self.r = r
        self.RR = RR
        self.T = T

    def treating_data(self, indicator, cluster, correlation_matrix, transition_matrix, external):
        data_portfolio, data_ref = self.data_portfolio, self.data_ref
        scoring_scale = self.scoring_scale
        currency = self.currency_portfolio
        shift_years = self.shift_years
        r = self.r
        T = self.T
        RR = self.RR

        data_portfolio = data_portfolio[["Investee", f"Investee {cluster}",
                                         f"Out. Principal At Cost ({currency})", f"Out. Principal ({currency})",
                                         "Last CR Grade", "End Date", "Start Date"]]

        # Fill Exposure data
        data_portfolio_copy = data_portfolio.copy()
        data_portfolio_copy['Exposure'] = data_portfolio[f"Out. Principal At Cost ({currency})"] \
            .fillna(data_portfolio[f"Out. Principal ({currency})"])
        data_portfolio = data_portfolio_copy

        # Delete N/A rows
        possible_CRs = ["AAA+", "AAA", "AAA-", "AA+", "AA", "AA-", "A+", "A", "A-",
                        "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-",
                        "C"]

        exposure_not_analyzed_data = data_portfolio.loc[lambda data: (data["Exposure"].isna()) |
                                                                     (data["Last CR Grade"].isna()) |
                                                                     (~data["Last CR Grade"].isin(possible_CRs))]

        ##### Attention: This exposure is NOT updated on time (we don't have its Credit Rating to do it)
        exposure_not_analyzed_data = exposure_not_analyzed_data["Exposure"].sum()

        data_portfolio = data_portfolio.loc[lambda data: ~data["Exposure"].isna()]
        data_portfolio = data_portfolio.loc[lambda data: ~data["Last CR Grade"].isna()]
        data_portfolio = data_portfolio.loc[lambda data: data["Last CR Grade"].isin(possible_CRs)]

        # Replace B- CR Grades if any (Scoring Scale doesn't admit B- as a possible score)
        data_portfolio["Last CR Grade"] = data_portfolio["Last CR Grade"].replace(['B-'], 'B')

        # Correct the exposure time value taking into account the bond maturity time
        ### Add maturity to bonds without maturity provided based on shift_years parameter
        data_portfolio["End Date"] = data_portfolio["End Date"] \
            .fillna(dt.date.today().replace(year=dt.date.today().year + shift_years))
        data_portfolio["End Date"] = pd.to_datetime(data_portfolio["End Date"].astype(str), format="%Y-%m-%d %H:%M:%S")

        data_portfolio["YTM"] = (data_portfolio["End Date"] - pd.to_datetime(dt.date.today(),
                                                                             format="%Y-%m-%d")) \
                                    .astype('timedelta64[D]') \
                                    .astype('int') / 365

        exposure_less_1y = data_portfolio.loc[data_portfolio["YTM"] < 1]["Exposure"].sum()

        ### Change maturity for the bonds with maturity before period analysis
        date_value = True
        while date_value:
            data_portfolio["End Date"] = np.where(
                data_portfolio["End Date"] < pd.to_datetime(dt.date.today(), format="%Y-%m-%d") + pd.DateOffset(
                    years=1),
                data_portfolio["End Date"] + (data_portfolio["End Date"] - data_portfolio["Start Date"]),
                data_portfolio["End Date"])
            if (data_portfolio["End Date"] < pd.to_datetime(dt.date.today(), format="%Y-%m-%d") + pd.DateOffset(
                    years=1)).any():
                date_value = True
            else:
                date_value = False

        ### Discount Exposure Value to T as a function of the bond maturity time
        TM_PV = transition_matrix.reset_index()
        if external:
            TM_PV["PD"] = TM_PV["Default"]
        else:
            TM_PV["PD"] = TM_PV["Default"] + TM_PV["C"] + TM_PV["B"]
        TM_PV.columns = ["Last CR Grade simplified"] + list(TM_PV.columns)[1:]

        TM_PV = TM_PV[["Last CR Grade simplified", "PD"]]

        idx = list(data_portfolio["Last CR Grade"])  # Portfolio Credit Rating
        dist_change = {"AAA+": "AAA", "AAA": "AAA", "AAA-": "AAA",
                       "AA+": "AA", "AA": "AA", "AA-": "AA",
                       "A+": "A", "A": "A", "A-": "A",
                       "BBB+": "BBB", "BBB": "BBB", "BBB-": "BBB",
                       "BB+": "BB", "BB": "BB", "BB-": "BB",
                       "B+": "B", "B": "B", "B-": "B",
                       "C": "C", "D": "Default"}

        for i in range(len(idx)):
            idx[i] = dist_change[idx[i]]

        data_portfolio["Last CR Grade simplified"] = idx

        data_portfolio = data_portfolio.merge(TM_PV, how="left", on="Last CR Grade simplified")
        LGD = 1 - RR
        data_portfolio["cs"] = -np.log(1 - LGD * data_portfolio["PD"]) / T
        data_portfolio["YTM"] = (data_portfolio["End Date"] - pd.to_datetime(dt.date.today(),
                                                                             format="%Y-%m-%d")).astype(
            'timedelta64[D]').astype('int') / 365
        data_portfolio["Exposure"] = data_portfolio["Exposure"] * np.exp(
            -(r + data_portfolio["cs"]) * data_portfolio["YTM"])

        # Get exposures
        total_exposure = data_portfolio["Exposure"].sum()

        # Get Type for each portfolio row
        ref_inst = data_ref[["FI Acronym", "Type"]] \
            .rename({"FI Acronym": "Investee"}, axis=1) \
            .assign(**{"Type": lambda data: data["Type"].str[:2]}) \
            .assign(**{"Type": lambda data: np.where(data["Type"].str[:2] == "N.", "P1", data["Type"])})

        data_portfolio = data_portfolio.merge(ref_inst, on="Investee", how="left")
        data_portfolio = data_portfolio.loc[lambda data: ~(data["Type"].isna())]
        data_portfolio[f"{cluster}_Type"] = data_portfolio[f"Investee {cluster}"] + " / " + \
                                            data_portfolio["Type"] + " / " + \
                                            indicator

        # Get mean CreditRating by Cluster Type
        scoring_scale["CR number"] = (scoring_scale["upper_threshold"] + scoring_scale["lower_threshold"]) / 2
        scoring_scale = scoring_scale.rename({"grade": "Last CR Grade"}, axis=1)

        data_CR = data_portfolio[[f"{cluster}_Type", "Last CR Grade"]]
        data_CR = data_CR.merge(scoring_scale[["Last CR Grade", "CR number"]], on="Last CR Grade", how="left")
        data_CR = data_CR.drop("Last CR Grade", axis=1)
        data_CR = data_CR.groupby(by=f"{cluster}_Type").mean()

        data_ = data_CR.copy()
        data_CR = data_CR.copy().astype(str)

        for i in range(len(data_["CR number"])):
            for row_ds in scoring_scale.itertuples():
                if row_ds[1] < data_["CR number"][i] <= row_ds[2]:
                    data_CR["CR number"][i] = row_ds[3]

        data_CR = data_CR.rename({"CR number": "Last CR Grade"}, axis=1)

        # Select portfolio data
        data_portfolio = data_portfolio[[f"{cluster}_Type", "Exposure"]]
        data_portfolio = data_portfolio.groupby(by=f"{cluster}_Type").sum()

        correl_types = list(correlation_matrix.columns)
        portfolio_types = list(data_portfolio.index)

        common_types = list(set(correl_types) & set(portfolio_types))

        exposure_not_analyzed_cor = data_portfolio.loc[lambda data: ~(data.index.isin(common_types))]["Exposure"].sum()

        data_portfolio = data_portfolio.loc[lambda data: data.index.isin(common_types)]
        data_CR = data_CR.loc[lambda data: data.index.isin(common_types)]

        return data_portfolio, data_CR, total_exposure, exposure_less_1y, exposure_not_analyzed_data, exposure_not_analyzed_cor

    @staticmethod
    def nearest_PD(A):
        def is_PD(b):
            try:
                _ = la.cholesky(b)
                return True
            except la.LinAlgError:
                return False

        B = (A + A.T) / 2
        _, s, V = la.svd(B)

        H = np.dot(V.T, np.dot(np.diag(s), V))
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2

        if is_PD(A3):
            return A3

        spacing = np.spacing(la.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not is_PD(A3):
            min_eig = np.min(np.real(la.eigvals(A3)))
            A3 += I * (-min_eig * k ** 2 + spacing)
            k += 1

        return A3

    @staticmethod
    def treat_cut_offs_port(cut_offs, idx):
        # treat cut_offs matrix
        co_cols = list(cut_offs.columns)
        for i in range(len(co_cols)):
            cut_offs[co_cols[i]] = np.where(cut_offs[co_cols[i]] - cut_offs[co_cols[i]].shift(-1).fillna(0) == 0,
                                            np.sign(cut_offs[co_cols[i]]) * 100,
                                            cut_offs[co_cols[i]])
            cut_offs[co_cols[i]][-1] = 100

        return np.matrix(cut_offs[idx].T.values)  # get idx cut_offs

    def credit_metrics(self, indicator, cluster, correlation_matrix, transition_matrix, external=False):
        data_exposure, data_CR, total_exposure, exposure_less_1y, exposure_not_analyzed_data, exposure_not_analyzed_cor = \
            self.treating_data(indicator, cluster, correlation_matrix, transition_matrix, external)
        total_clusters = data_exposure.shape[0]

        # Delete Default row Transition Matrix
        transition_matrix = transition_matrix.loc[lambda data: ~(data.index == "Default")]

        M = self.M  # Number of Monte-Carlo Sim
        r = self.r  # Risk-free rate
        T = self.T  # Period
        RR = self.RR  # Recover Rate
        LGD = 1 - RR  # Loss Given Default
        # Default Probability
        PD = transition_matrix["Default"] if external else transition_matrix["B"] + \
                                                           transition_matrix["C"] + \
                                                           transition_matrix["Default"]
        cs = -np.log(1 - LGD * PD) / T  # Credit Spread
        exposure = data_exposure["Exposure"]  # Portfolio Exposure (Series)
        n = data_exposure.shape[0]  # Number of Assets
        Loss_CreditMetrics = np.zeros((n, M))  # Loss Distribution matrix from usual CreditMetrics
        Loss_Symbiotics = np.zeros((n, M))  # Loss Distribution matrix using Symbiotics Factor
        c = la.cholesky(self.nearest_PD(correlation_matrix))  # Correlation Factor

        idx = list(data_CR["Last CR Grade"])  # Portfolio Credit Rating
        dist_change = {"AAA+": "AAA", "AAA": "AAA", "AAA-": "AAA",
                       "AA+": "AA", "AA": "AA", "AA-": "AA",
                       "A+": "A", "A": "A", "A-": "A",
                       "BBB+": "BBB", "BBB": "BBB", "BBB-": "BBB",
                       "BB+": "BB", "BB": "BB", "BB-": "BB",
                       "B+": "B", "B": "B", "B-": "B",
                       "C": "C"}

        for i in range(len(idx)):
            idx[i] = dist_change[idx[i]]

        factor_rule_letters = pd.Series(data=[1, 1, 1, 1, 1, 0.8, 0.5, 0.2],
                                        index=["AAA", "AA", "A", "BBB", "BB", "B", "C", "D"])

        # Enterprise value vector
        EV_CreditMetrics = np.multiply(np.array(exposure), np.exp(-(r + np.array(cs[idx])) * T))
        EV_Symbiotics = np.multiply(EV_CreditMetrics.T, np.array(factor_rule_letters[idx]))

        # Cut-offs
        Z = np.cumsum(np.flipud(transition_matrix.T), 0)
        Z[Z >= 1] = 1 - 1 / 1e12
        Z[Z <= 0] = 0 + 1 / 1e12
        cut_offs = norm.ppf(Z, 0, 1)  # compute cut-offs for each CR by inverting normal distribution
        rating_list_col = ["AAA", "AA", "A", "BBB", "BB", "B", "C"]
        rating_list_index = ["Default", "C", "B", "BB", "BBB", "A", "AA", "AAA"]
        cut_offs = pd.DataFrame(cut_offs, columns=rating_list_col, index=rating_list_index)

        # Cut-offs Portfolio
        cut_offs_portfolio = self.treat_cut_offs_port(cut_offs, idx)

        # bond state variable for Security Value
        cp = np.tile(np.array(cs).T, [n, 1])
        state = np.multiply(np.array([exposure]).T, np.exp(-(r + cp) * T))
        state_2 = np.append(state, np.multiply(np.array([exposure]).T, RR), axis=1)
        states = np.fliplr(state_2)

        # Monte Carlo Simulation Nsim times
        for i in range(0, M):
            YY = np.matrix(np.random.normal(size=n))  # Random Normal Sampling (RNS)
            rr = c * YY.T  # RNS correlated as our data (use of Cholesky Factor)

            rating = rr < cut_offs_portfolio
            rate_idx = rating.shape[1] - np.sum(rating, 1)
            row_idx = range(0, n)
            col_idx = np.squeeze(np.asarray(rate_idx))

            V_t_CreditMetrics = states[row_idx, col_idx]
            Loss_t_CreditMetrics = V_t_CreditMetrics - EV_CreditMetrics.T
            Loss_CreditMetrics[:, i] = Loss_t_CreditMetrics

            factor_rule_numbers = pd.Series(data=[.2, .5, .8, 1, 1, 1, 1, 1],
                                            index=[0, 1, 2, 3, 4, 5, 6, 7])

            V_t_Symbiotics = np.multiply(EV_Symbiotics.T, factor_rule_numbers[list(col_idx)])
            Loss_t_Symbiotics = V_t_Symbiotics - EV_Symbiotics.T
            Loss_Symbiotics[:, i] = Loss_t_Symbiotics

        Portfolio_MC_Loss_CreditMetrics = np.sum(Loss_CreditMetrics, 0)
        Port_Var_CM = -1 * np.percentile(Portfolio_MC_Loss_CreditMetrics, 1)
        ES_CM = -1 * np.mean(Portfolio_MC_Loss_CreditMetrics[Portfolio_MC_Loss_CreditMetrics <= -1 * Port_Var_CM])

        Portfolio_MC_Loss_Symbiotics = np.sum(Loss_Symbiotics, 0)
        Port_Var_Symbiotics = -1 * np.percentile(Portfolio_MC_Loss_Symbiotics, 1)
        ES_Symbiotics = -1 * np.mean(
            Portfolio_MC_Loss_Symbiotics[Portfolio_MC_Loss_Symbiotics <= -1 * Port_Var_Symbiotics])

        return Port_Var_CM, ES_CM, \
               Port_Var_Symbiotics, ES_Symbiotics, \
               total_exposure, exposure_less_1y, exposure_not_analyzed_data, exposure_not_analyzed_cor, \
               total_clusters

    def result_set(self):
        correlation_matrix_collection = self.CorrelationMatrix_dict
        transition_matrix_collection = self.TransitionMatrix_dict

        name_portfolio = self.name_portfolio
        r = self.r
        RR = self.RR

        df_result = pd.DataFrame(data=[], columns=["Portfolio",
                                                   "Transition Matrix",
                                                   "Cluster Method",
                                                   "Indicator",
                                                   "VaR CreditMetrics",
                                                   "ES CreditMetrics",
                                                   "VaR Symbiotics",
                                                   "ES Symbiotics",
                                                   "Total Portfolio Exposure",
                                                   "Exposure Maturity <1Y",
                                                   "Exposure Not Analyzed (Data)",
                                                   "Exposure Not Analyzed (Clusters)",
                                                   "Total Clusters Analyzed",
                                                   "Number of Not Analyzed Clusters",
                                                   "Not Analyzed Clusters",
                                                   "Risk-free rate",
                                                   "Recovery Rate"])

        # Transition Matrix
        for n in range(len(transition_matrix_collection)):
            transition_matrix = list(transition_matrix_collection.values())[n]
            external = True if list(transition_matrix_collection.keys())[n] == "External" else False

            # Correlation Matrix
            for j in range(len(correlation_matrix_collection)):
                for i in range(len(correlation_matrix_collection[j])):
                    row = []
                    correlation_matrix, number_out_of_analysis, out_of_analysis = \
                        list(correlation_matrix_collection[j].values())[i]

                    indicator, cluster = list(correlation_matrix_collection[j].keys())[i].split("_")

                    VaR_CM, ES_CM, VaR_Symbiotics, ES_Symbiotics, total_exp, exposure_less_1y, exp_out_data, exp_out_cor, total_clusters = \
                        self.credit_metrics(indicator, cluster,
                                            correlation_matrix, transition_matrix, external)

                    row += [name_portfolio,
                            list(transition_matrix_collection.keys())[n],
                            cluster,
                            indicator,
                            VaR_CM,
                            ES_CM,
                            VaR_Symbiotics,
                            ES_Symbiotics,
                            total_exp,
                            exposure_less_1y,
                            exp_out_data,
                            exp_out_cor,
                            total_clusters,
                            number_out_of_analysis,
                            out_of_analysis,
                            r,
                            RR]

                    df_result = df_result.append(pd.DataFrame(np.array(row).reshape(-1, len(row)),
                                                              columns=df_result.columns))

        df_result["Total Portfolio Exposure"] = pd.to_numeric(df_result["Total Portfolio Exposure"]) \
            .map('{:>10,.0f}'.format)
        df_result["VaR CreditMetrics"] = pd.to_numeric(df_result["VaR CreditMetrics"]).map('{:>10,.0f}'.format)
        df_result["ES CreditMetrics"] = pd.to_numeric(df_result["ES CreditMetrics"]).map('{:>10,.0f}'.format)
        df_result["VaR Symbiotics"] = pd.to_numeric(df_result["VaR Symbiotics"]).map('{:>10,.0f}'.format)
        df_result["ES Symbiotics"] = pd.to_numeric(df_result["ES Symbiotics"]).map('{:>10,.0f}'.format)
        df_result["Exposure Not Analyzed (Data)"] = pd.to_numeric(df_result["Exposure Not Analyzed (Data)"]) \
            .map('{:>10,.0f}'.format)
        df_result["Exposure Not Analyzed (Clusters)"] = pd.to_numeric(df_result["Exposure Not Analyzed (Clusters)"]) \
            .map('{:>10,.0f}'.format)
        df_result["Exposure Maturity <1Y"] = pd.to_numeric(df_result["Exposure Maturity <1Y"]).map('{:>10,.0f}'.format)

        return df_result
