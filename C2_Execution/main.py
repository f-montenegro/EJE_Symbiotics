from C1_Dev.CorrelationMatrix import *
from C1_Dev.TransitionMatrix import *
from C1_Dev.CreditMetrics import *
from C2_Execution.Variables import *


###########################################
##### Transition Matrix for both methods
###########################################
print("START TRANSITION MATRIX INPUT GENERATION")

transition_matrix_collection = {}
for i in range(len(data_TransitionMatrix)):
    data_type = list(data_TransitionMatrix.keys())[i]
    transition_matrix = TransitionMatrix(data_TransitionMatrix[data_type], data_type, scoring_scale, data_ext)
    cohort_transition_matrix = transition_matrix.CohortTransitionMatrix
    hazard_transition_matrix = transition_matrix.HazardTransitionMatrix
    external_transition_matrix = transition_matrix.ExternalTransitionMatrix
    transition_matrix_collection.update({f"Cohort_{data_type}": cohort_transition_matrix,
                                         f"Hazard_{data_type}": hazard_transition_matrix,
                                         f"External": external_transition_matrix})

# Generate Excel Transition Matrix
writer_transition = pd.ExcelWriter('main_results/TransitionMatrix.xlsx', engine='xlsxwriter')
for i in range(len(transition_matrix_collection)):
    df = list(transition_matrix_collection.values())[i]
    df.to_excel(writer_transition, sheet_name=f"{list(transition_matrix_collection.keys())[i]}")
writer_transition.save()
print("END TRANSITION MATRIX INPUT GENERATION")


df_CreditMetrics = pd.DataFrame([])
for p in range(len(portfolios)):
    ptfl_name = list(portfolios.keys())[p]
    ptfl_data = list(portfolios.values())[p][0]
    ptfl_currency = list(portfolios.values())[p][1]
    print(f"Calculating Results for {ptfl_name}...")

    ###########################################
    ##### Correlation Matrix for all indicators
    ###########################################
    print(f"START CORRELATION MATRIX INPUT GENERATION FOR: {ptfl_name}")
    class_correl_matrix = CorrelationMatrix(data_FIR, ptfl_data, ptfl_currency, data_ref, ratio_ind, cluster_method)
    correlation_matrix_collection = class_correl_matrix.CorrelationMatrixCollection

    # Generate Excel Correlation Matrix
    writer_correl = pd.ExcelWriter(f'main_results/CorrelMatrix_{ptfl_name}.xlsx', engine='xlsxwriter')
    for j in range(len(correlation_matrix_collection)):
        for i in range(len(correlation_matrix_collection[j])):
            df = list(correlation_matrix_collection[j].values())[i][0]
            df.to_excel(writer_correl, sheet_name=f"{list(correlation_matrix_collection[j].keys())[i]}")
    writer_correl.save()

    print(f"END CORRELATION MATRIX INPUT GENERATION FOR: {ptfl_name}")

    ###########################################
    ##### Credit Metrics Result
    ###########################################
    print(f"START CREDIT METRICS RESULT GENERATION FOR: {ptfl_name}")

    class_CreditMetrics = CreditMetrics(correlation_matrix_collection, transition_matrix_collection,
                                        ptfl_data, ptfl_name, ptfl_currency,
                                        data_ref, scoring_scale, shift_years,
                                        M, r, RR, T)

    df = class_CreditMetrics.result_set()
    df_CreditMetrics = df_CreditMetrics.append(df)

    print(f"END CREDIT METRICS RESULT GENERATION FOR: {ptfl_name}")

print("Generating Excel file with all portfolios' results")
writer_credit_metrics = pd.ExcelWriter('main_results/CreditMetrics.xlsx', engine='xlsxwriter')
df_CreditMetrics.to_excel(writer_credit_metrics, sheet_name="CreditMetrics", index=False)
writer_credit_metrics.save()
