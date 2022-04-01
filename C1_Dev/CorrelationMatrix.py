from C0_Library.config import *


class CorrelationMatrix:
    def __init__(self, data_FIR, data_portfolio, currency, data_ref, ratio_ind,
                 cluster=None, cat_cols=6, type_na="P1 - Micro Finance Institutions"):

        """
        data_FIR -> Financial Institution Ratio (excel brut_data)
        cluster -> by Region/Country
        ratio_ind -> indicator value used in the analysis
        cat_cols -> number of categorical columns (columns before values) in the brut data
        type_na -> value to use when type is N/A in data_FIR "Type" column
        """

        if cluster is None:
            cluster = ["Region", "Country"]

        self.data_FIR = data_FIR
        self.data_portfolio = data_portfolio
        self.currency = currency
        self.data_ref = data_ref
        self.cluster = cluster
        self.ratio_indicator = ratio_ind
        self.cat_cols = cat_cols
        self.type_na = type_na

        CorrelationMatrixCollection = []
        for j in range(len(self.cluster)):
            correlation_matrix_output = {}
            data = self.treating_data(self.cluster[j])
            for i in range(len(ratio_ind)):
                corr_matrix, number_out_of_analysis, out_of_analysis = \
                                    self.correlation_matrix_by_indicator(data, self.ratio_indicator[i], self.cluster[j])

                correlation_matrix_output.update({f"{self.ratio_indicator[i]}_{self.cluster[j]}":
                                                      [corr_matrix, number_out_of_analysis, out_of_analysis]})

            CorrelationMatrixCollection += [correlation_matrix_output]

        self.CorrelationMatrixCollection = CorrelationMatrixCollection

    def treating_data(self, cluster_type):
        clustering_method = [cluster_type, "Type", "Ratio Indicator BK"]
        data = self.data_FIR
        cat_cols, type_na = self.cat_cols, self.type_na

        # Step1: fill empty values
        data_ = data.copy()
        data_ = data_.set_index(list(data_.columns[:cat_cols]))
        data_ = data_.replace(0, np.nan)
        data_ = data_.dropna(how='all')
        data_ = data_.reset_index()

        value_columns = data_.columns[cat_cols:]
        data_values = data_[value_columns]
        data_values_ = data_values.copy()

        for j in range(data_values.shape[0]):
            index = []
            for i in range(data_values.loc[j].shape[0]):
                if not data_values.loc[j].isna()[i]:
                    index += [data_values.loc[j].index[i]]
            if not index[0] == index[-1]:
                filled_row = data_values.loc[j].loc[index[0]:index[-1]].ffill()
                data_values_.loc[j] = filled_row.combine(data_values.loc[j], func=max)

        data_[data_.columns[cat_cols:]] = data_values_

        # Step2: cluster types
        type_dict = {"Type": ["N.A.",
                              'P1 - Downscaling commercial banks', 'P1 - Fintech', 'P1 - Micro Finance Institutions',
                              'P2 - Fund and structured transactions', 'P2 - SME banks',
                              'P2 - Specialized Finance companies',
                              'P3 - Project Finance Clean Energy', 'P3 - Project Finance Other',
                              'P4 - Third Party Origination'],
                     "Type_simplified": ["P1", "P1", "P1", "P1", "P2", "P2", "P2", "P3", "P3", "P4"]}

        df_type = pd.DataFrame(type_dict, columns=["Type", "Type_simplified"])
        data_ = data_.merge(df_type, on="Type", how="left")
        data_["Type"] = data_["Type_simplified"]
        data_ = data_.drop("Type_simplified", axis=1)

        # Step3: create cluster
        data_ = data_.groupby(by=clustering_method).mean()

        # Step4: Delete all clusters with only one value
        data_["NaN Quantity"] = data_.T.loc[data_.columns].isna().sum()
        data_ = data_.loc[lambda x: ~(x["NaN Quantity"] == (data_.iloc[:, cat_cols:-1].shape[1] - 1))]
        data_ = data_.drop("NaN Quantity", axis=1)

        return data_

    def correlation_matrix(self, data_indicator, cluster_type):
        data_portfolio, data_ref, type_na = self.data_portfolio, self.data_ref, self.type_na
        currency = self.currency

        possible_CRs = ["AAA+", "AAA", "AAA-", "AA+", "AA", "AA-", "A+", "A", "A-",
                        "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-",
                        "C"]

        data_portfolio_copy = data_portfolio.copy()
        data_portfolio_copy['Exposure'] = data_portfolio[f"Out. Principal At Cost ({currency})"] \
            .fillna(data_portfolio[f"Out. Principal ({currency})"])
        data_portfolio = data_portfolio_copy

        data_portfolio = data_portfolio.loc[lambda data: ~data["Exposure"].isna()]
        data_portfolio = data_portfolio.loc[lambda data: ~data["Last CR Grade"].isna()]
        data_portfolio = data_portfolio.loc[lambda data: data["Last CR Grade"].isin(possible_CRs)]

        # Step1: Prepare data_indicator
        #########

        assert len(data_indicator["Ratio Indicator BK"].unique()) == 1
        indicator = data_indicator["Ratio Indicator BK"].unique()[0]

        clustering_method = [cluster_type, "Type", "Ratio Indicator BK"]
        cat_cols = len(clustering_method)

        label = []
        for j in range(data_indicator.shape[0]):
            # getting rows/columns names
            label += [" / ".join(list(data_indicator.loc[j][:cat_cols]))]

        data_indicator["Label"] = label

        # Step2: Get portfolio clusters
        ########

        ref_inst = data_ref[["FI Acronym", "Type"]] \
                        .rename({"FI Acronym": "Investee"}, axis=1) \
                        .assign(**{"Type": lambda data: data["Type"].str[:2]}) \
                        .assign(**{"Type": lambda data: np.where(data["Type"].str[:2] == "N.",
                                                                 type_na[:2],
                                                                 data["Type"])})

        portfolio_clusters = data_portfolio[["Investee", "Investee " + cluster_type]]
        portfolio_clusters = portfolio_clusters.merge(ref_inst, on="Investee", how="left")
        portfolio_clusters = portfolio_clusters.loc[lambda data: ~(data["Type"].isna())]
        portfolio_clusters[f"{cluster_type}_Type"] = portfolio_clusters[f"Investee {cluster_type}"] + " / " + \
                                                     portfolio_clusters["Type"] + " / " + \
                                                     indicator

        # Step3: Get out of analysis clusters (column Not Analyzed Clusters in the Output)
        ########

        cluster_port = list(portfolio_clusters[f"{cluster_type}_Type"])
        out_of_analysis = []
        for i in range(len(cluster_port)):
            if cluster_port[i] not in list(data_indicator["Label"]):
                out_of_analysis += [cluster_port[i]]

        number_out_of_analysis = len(out_of_analysis)

        categories_out_of_analysis = ""
        for category in out_of_analysis:
            categories_out_of_analysis += category + " ; "

        # Step4: Get Correlation Matrix
        ########

        in_analysis = []
        for i in range(len(cluster_port)):
            if cluster_port[i] in list(data_indicator["Label"]):
                in_analysis += [cluster_port[i]]

        in_analysis = np.unique(in_analysis)

        NA_Values = True
        df_corr_matrix = pd.DataFrame([])
        while NA_Values:
            data_indicator = data_indicator.loc[lambda data: data["Label"].isin(in_analysis)]

            value_columns = data_indicator.drop("Label", axis=1).reset_index(drop=True).columns[cat_cols:]
            data_values = data_indicator[value_columns]

            df_corr_matrix = data_values.T.corr(method='pearson')

            df_corr_matrix["Label"] = in_analysis
            df_corr_matrix = df_corr_matrix.set_index("Label")
            df_corr_matrix.columns = in_analysis
            df_corr_matrix = df_corr_matrix.dropna(how='all')
            df_corr_matrix = df_corr_matrix.dropna(how='all', axis=1)
            NA_Values = df_corr_matrix.isnull().values.any()

            # If NA Values is True, it is because there are two or more time-series with no common periods
            # This situation generates N/A values in our correlation matrix and we don't want that!
            if NA_Values:
                na_series = df_corr_matrix.isna().sum()
                problem_category = na_series.idxmax()
                number_out_of_analysis += 1
                categories_out_of_analysis += problem_category + " ; "
                data_indicator = data_indicator[data_indicator["Label"] != problem_category]
                data_indicator = data_indicator.reset_index(drop=True)
                in_analysis = [x for x in in_analysis if x != problem_category]

        return df_corr_matrix, number_out_of_analysis, categories_out_of_analysis

    def correlation_matrix_by_indicator(self, data, indicator, cluster_type):
        data_indicator = data.reset_index() \
                             .loc[lambda x: x["Ratio Indicator BK"] == indicator] \
                             .reset_index(drop=True)

        return self.correlation_matrix(data_indicator, cluster_type)
