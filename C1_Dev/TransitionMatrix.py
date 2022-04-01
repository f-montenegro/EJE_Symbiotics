from C0_Library.config import *


class TransitionMatrix:
    def __init__(self, data_CR, data_type, scoring_scale, data_ext, cat_cols=4):

        """
        data_CR -> Credit Rating Published or Trend (excel brut_data)
        data_type -> CRP or trend
        scoring_scale -> table used in the rating conversion (number to symbols)
        data_ext -> external matrix input
        cat_cols -> number of categorical columns (columns before values) in the brut data
        """

        self.data_CR = data_CR
        self.data_ext = data_ext
        self.scoring_scale = scoring_scale
        self.cat_cols = cat_cols

        self.data_type = data_type
        assert self.data_type in ["CRP", "trend"], "Please select an available data_type parameter (CRP or trend)"

        self.data_CR_treated = self.treating_data()
        self.data_CR_treated['Year'] = pd.DatetimeIndex(self.data_CR_treated['Date']).year
        self.data_CR_treated['TimeDiff'] = self.data_CR_treated['Date'].diff()
        self.yend = max(self.data_CR_treated["Date"]).year
        self.size = len(self.data_CR_treated)
        self.maxID = self.data_CR_treated.iloc[self.size - 1]["ID"]

        self.number_CR = len(self.scoring_scale["grade"])
        self.CohortTransitionMatrix = np.zeros([self.number_CR, self.number_CR])
        self.HazardTransitionMatrix = np.zeros([self.number_CR, self.number_CR])
        self.TransMatrixResults = np.zeros([self.number_CR, self.number_CR])
        self.TransDenResults = np.zeros([self.number_CR])
        self.TransDenLambda = np.zeros([self.number_CR])

        self.CohortTransitionMatrix = self.cohort()
        self.HazardTransitionMatrix = self.hazard()
        self.ExternalTransitionMatrix = self.external()

    def treating_data(self):
        data_CR, cat_cols, scoring_scale = self.data_CR, self.cat_cols, self.scoring_scale

        # Step1: Layout Changing (number to letter rating)
        data_ = data_CR.copy().astype(str)

        data_values = data_CR[data_CR.columns[cat_cols:]]
        for col in data_values.columns:
            for i in range(len(data_values[col])):
                for row_ds in scoring_scale.itertuples():
                    if row_ds[1] < data_values[col][i] <= row_ds[2]:
                        data_[col][i] = row_ds[3]

        # Step2: Straight Data
        data_ = data_[["FI BK"] + list(data_.columns[cat_cols:])].melt(id_vars=["FI BK"],
                                                                       var_name="Date",
                                                                       value_name="CreditRating") \
                                                                 .loc[lambda data: ~(data["CreditRating"] == 'nan')] \
                                                                 .sort_values(by=["FI BK", "Date"]) \
                                                                 .reset_index(drop=True)

        quarters = {"Q1": "03-31", "Q2": "06-30", "Q3": "09-30", "Q4": "12-31"}
        df_quarters = pd.DataFrame(list(quarters.items()), columns=['Quarters', 'Day'])

        data_["Year"] = data_["Date"].str[:4]
        data_["Quarters"] = data_["Date"].str[5:]
        data_ = pd.merge(data_, df_quarters, on="Quarters", how="left")
        data_["Date"] = data_["Year"] + "-" + data_["Day"]
        data_ = data_.drop(["Year", "Quarters", "Day"], axis=1)

        df_ID = pd.DataFrame(data_["FI BK"].unique(), columns=['FI BK']).reset_index()
        df_ID = df_ID.rename({"index": "ID"}, axis=1)
        df_ID["ID"] = df_ID["ID"] + 1

        data_ = pd.merge(df_ID, data_, on='FI BK', how="right")

        mapping_rating = scoring_scale.reset_index()
        mapping_rating = mapping_rating.rename({"index": "RatingNumber"}, axis=1)
        mapping_rating = mapping_rating[["grade", "RatingNumber"]]
        mapping_rating = mapping_rating.rename({"grade": "CreditRating"}, axis=1)

        data_ = pd.merge(data_, mapping_rating, on='CreditRating', how="left")
        data_["Date"] = pd.to_datetime(data_["Date"], format="%Y-%m-%d").dt.date

        return data_

    @staticmethod
    def cr_distribution(data, external=False):
        dist_change = {"AAA": "AAA",
                       "AA+": "AA", "AA": "AA", "AA-": "AA",
                       "A+": "A", "A": "A", "A-": "A",
                       "BBB+": "BBB", "BBB": "BBB", "BBB-": "BBB",
                       "BB+": "BB", "BB": "BB", "BB-": "BB",
                       "B+": "B", "B": "B",
                       "C": "C", "D": "Default"}

        if external:
            dist_change = {"AAA": "AAA",
                           "AA": "AA",
                           "A": "A",
                           "BBB": "BBB",
                           "BB": "BB",
                           "B": "B",
                           "C": "C",
                           "D": "Default"}

        data = data.reset_index()
        data = data.rename({list(data.columns)[0]: "Index"}, axis=1)
        data["Index"] = list(dist_change.values())
        data = data.set_index("Index")
        data.columns = list(dist_change.values())
        data = data.reset_index()
        data = data.groupby(by=["Index"], sort=False).sum()
        data = data.groupby(data.columns, axis=1, sort=False).sum()

        data["Total"] = data.sum(axis=1)
        L = list(data.columns)[:-1]
        for i in range(len(L)):
            data = data.assign(**{L[i]: lambda x: x[L[i]] / x["Total"]})
        data = data.drop("Total", axis=1)

        return data

    def cohort(self):
        scoring_scale, data_CR, yend, maxID = self.scoring_scale, self.data_CR_treated, self.yend, self.maxID
        number_CR, CohortTransitionMatrix = self.number_CR, self.CohortTransitionMatrix
        TransMatrixResults, TransDenResults = self.TransMatrixResults, self.TransDenResults

        for i in range(1, maxID+1):
            data_CR_id = data_CR[data_CR['ID'] == i]
            ybeg = (data_CR_id.iloc[0]["Date"]).year

            RatingsBeg = [data_CR_id.iloc[0]["CreditRating"]]  # CR at the beginning of the period
            RatingsEnd = []  # CR at the end of the period

            RatingsBegIndex = [data_CR_id.iloc[0]["RatingNumber"]]  # CR index at the beginning of the period
            RatingsEndIndex = []  # CR index at the end of the period

            defaulted = data_CR_id[data_CR_id["CreditRating"] == "D"]
            defaultedSize = len(defaulted)
            defaultYes = False
            if defaultedSize > 0:
                yend = defaulted.iloc[0]["Date"].year
                defaultYes = True

            data_loop2 = pd.DataFrame([])
            for j in range(ybeg, yend + 1):
                data_loop = data_CR_id[data_CR_id['Year'] == j]

                if len(data_loop) > 0:
                    dateCond = max(data_loop["Date"])
                    data_loop2 = data_loop[data_loop['Date'] == dateCond]
                    RatingsEnd.append(data_loop2.iloc[0]["CreditRating"])
                    RatingsEndIndex.append(data_loop2.iloc[0]["RatingNumber"])

                else:
                    RatingsEnd.append(data_loop2.iloc[0]["CreditRating"])
                    RatingsEndIndex.append(data_loop2.iloc[0]["RatingNumber"])

                RatingsBeg.append(data_loop2.iloc[0]["CreditRating"])
                RatingsBegIndex.append(data_loop2.iloc[0]["RatingNumber"])

                if RatingsEnd[-1] == 'D':
                    break

            if defaultYes:
                RatingsEnd[-1] = 'D'
                RatingsEndIndex[-1] = 7

            RatingsBegIndex = RatingsBegIndex[:-1]

            TransMatrix = np.zeros([number_CR, number_CR])
            TransDen = np.zeros([number_CR])

            for k in range(len(RatingsEndIndex)):
                TransMatrix[RatingsBegIndex[k], RatingsEndIndex[k]] += 1
                TransDen[RatingsBegIndex[k]] += 1

            TransMatrixResults += TransMatrix
            TransDenResults += TransDen

        for i in range(number_CR):
            for j in range(number_CR):
                if TransDenResults[i] == 0:
                    CohortTransitionMatrix[i, j] = 0.0
                else:
                    CohortTransitionMatrix[i, j] = TransMatrixResults[i, j] / TransDenResults[i]

        for k in range(number_CR):
            if CohortTransitionMatrix[k, k] == 0:
                CohortTransitionMatrix[k, k] = 1

        CohortTransitionMatrix[number_CR-1, number_CR-1] = 1.0
        CohortTransitionMatrix = pd.DataFrame(CohortTransitionMatrix)
        CohortTransitionMatrix.columns = list(scoring_scale["grade"])
        CohortTransitionMatrix.index = list(scoring_scale["grade"])
        self.TransMatrixResults = TransMatrixResults
        self.TransDenResults = TransDenResults

        return self.cr_distribution(CohortTransitionMatrix)

    def hazard(self):
        scoring_scale, data_CR, yend, maxID = self.scoring_scale, self.data_CR_treated, self.yend, self.maxID
        number_CR, HazardTransitionMatrix = self.number_CR, self.HazardTransitionMatrix

        TransMatrixResults, TransDenLambda = self.TransMatrixResults, self.TransDenLambda

        for k in range(1, maxID+1):
            DefaultYesHazard = False
            data_CR_id = data_CR[data_CR['ID'] == k]
            size = len(data_CR_id)
            for l in range(1, size):
                valore = data_CR_id.iloc[l]["TimeDiff"].days / 365
                TransDenLambda[data_CR_id.iloc[l-1]["RatingNumber"]] += valore

                if data_CR_id.iloc[l]["CreditRating"] == 'D':
                    DefaultYesHazard = True
                    break

            # First Period
            dbeg = dt.date(data_CR_id.iloc[0]["Year"], 1, 1)
            valoreBeg = (data_CR_id.iloc[0]["Date"] - dbeg).days / 365.0
            TransDenLambda[data_CR_id.iloc[0]["RatingNumber"]] += valoreBeg

            # Last Period Analysis
            dfinal = dt.date(yend, 12, 31)
            if not DefaultYesHazard:
                valoreEnd = (dfinal - data_CR_id.iloc[size-1]["Date"]).days / 365.0
                TransDenLambda[data_CR_id.iloc[size-1]["RatingNumber"]] += valoreEnd

        for i in range(number_CR):
            for j in range(number_CR):
                if TransDenLambda[i] == 0:
                    HazardTransitionMatrix[i, j] = 0
                else:
                    HazardTransitionMatrix[i, j] = self.TransMatrixResults[i, j] / TransDenLambda[i]

        HazardTransitionMatrix[number_CR-1, number_CR-1] = 1.0
        self.TransDenLambda = TransDenLambda

        for i in range(number_CR):
            HazardTransitionMatrix[i, i] = 0
            HazardTransitionMatrix[i, i] = -sum(HazardTransitionMatrix[i, :])

        # Step 1: Calculate the maximum negative number of the generator, lmax
        lmax = 0
        for i in range(number_CR):
            if np.abs(HazardTransitionMatrix[i, i]) > lmax:
                lmax = np.abs(HazardTransitionMatrix[i, i])

        # Step 2: Create a diagonal matrix with lmax as main diagonal
        mat1 = np.zeros((number_CR, number_CR))
        np.fill_diagonal(mat1, lmax)

        # Step 3: Add the diagonal matrix in 2 to the generator to calculate lStar
        LStar = HazardTransitionMatrix + mat1

        # Step 4: Calculate the matrix exponential of (LStar)
        tmp = expm(LStar)
        vec1 = np.zeros((number_CR, number_CR))
        np.fill_diagonal(vec1, np.exp(-lmax))

        # Step 5: Multiply Exp(-1 * lmax) by the matrix in 4
        mexpgenerator = np.dot(vec1, tmp)
        HazardTransitionMatrix = pd.DataFrame(mexpgenerator).round(6)
        HazardTransitionMatrix.columns = list(scoring_scale["grade"])
        HazardTransitionMatrix.index = list(scoring_scale["grade"])
        return self.cr_distribution(HazardTransitionMatrix)

    def external(self):
        data_ext = self.data_ext
        data_ext = data_ext.rename({list(data_ext.columns)[0]: "Index"}, axis=1) \
                           .set_index("Index") \
                           .rename({"CCC/C": "C"}, axis=1) \
                           .rename({"CCC/C": "C"}, axis=0)

        D_row = pd.DataFrame({"D": [0] * 7 + [100] + [0]}, index=list(data_ext.columns)).transpose()
        data_ext = pd.concat([data_ext, D_row], axis=0)

        M = []
        for x in data_ext.itertuples():
            x_values = list(x[1:-1])
            den = sum(x_values)
            for i in range(len(x_values)):
                x_values[i] = (x_values[i]/den * x[-1] + x[i + 1])/100
            M.append(x_values)

        data_ext = pd.DataFrame(M, columns=list(data_ext.columns)[:-1], index=data_ext.index)

        return self.cr_distribution(data_ext, external="True")
