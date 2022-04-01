from C0_Library.config import *

# Scoring Scale Table
scoring_scale_columns = ["lower_threshold", "upper_threshold", "grade", "qualifier"]

scoring_scale_lower_threshold = [.94, .91, .88, .85, .82, .79, .76, .73, .7, .67, .64, .61, .58, .49, .40, .20, 0]
scoring_scale_upper_threshold = [1, .94, .91, .88, .85, .82, .79, .76, .73, .7, .67, .64, .61, .58, .49, .40, .20]
scoring_scale_grade = ["AAA",
                       "AA+", "AA", "AA-",
                       "A+", "A", "A-",
                       "BBB+", "BBB", "BBB-",
                       "BB+", "BB", "BB-",
                       "B+", "B",
                       "C", "D"]

scoring_scale_quali = ["Extremely low credit risk",
                       "Very low credit risk", "Very low credit risk", "Very low credit risk",
                       "Low credit risk", "Low credit risk", "Low credit risk",
                       "Moderate credit risk", "Moderate credit risk", "Moderate credit risk",
                       "Material credit risk", "Material credit risk", "Material credit risk",
                       "High credit risk", "High credit risk",
                       "Very high credit risk", "Very high credit risk"]

scoring_scale_values = [list(np.array(scoring_scale_lower_threshold)*5),
                        list(np.array(scoring_scale_upper_threshold)*5),
                        scoring_scale_grade,
                        scoring_scale_quali]

scoring_scale_data_dict = {scoring_scale_columns[i]: scoring_scale_values[i] for i in range(len(scoring_scale_columns))}
scoring_scale = pd.DataFrame(scoring_scale_data_dict)
