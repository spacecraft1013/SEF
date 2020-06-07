import matplotlib.pyplot as plt
import pandas as pd

selected_data = pd.read_csv("data/selected_data_10.csv", index_col=0)

averages = selected_data.mean()

# plt.bar(selected_data.index, selected_data['count_rp_gp_p25'], label='$25^{th}$ Percentile Parent Income')
# plt.bar(selected_data.index, selected_data['count_rp_gp_p25_l'], label='$25^{th}$ Percentile Parent Income (Late Cohort)')
# plt.bar(selected_data.index, selected_data['count_rp_gf_pall_l'], label='Female Parent Income (Late Cohort)')
# plt.bar(selected_data.index, selected_data['count_rp_gf_p25'], label='$25^{th}$ Percentile Female Parent Income')
# plt.bar(selected_data.index, selected_data['count_rp_gf_p25_l'], label='$25^{th}$ Percentile Female Parent Income (Late Cohort)')
# plt.bar(selected_data.index, selected_data['count_rp_gm_p25'], label='$25^{th}$ Percentile Male Parent Income')
# plt.bar(selected_data.index, selected_data['count_rp_gm_p25_l'], label='$25^{th}$ Percentile Male Parent Income (Late Cohort)')
# plt.bar(selected_data.index, selected_data['count_rh_gp_pall_l'], label='Hispanic Parent Income (Late Cohort)')
# plt.bar(selected_data.index, selected_data['count_rh_gp_p75_l'], label='$75^{th}$ Percentile Hispanic Parent Income (Late Cohort)')
# plt.bar(selected_data.index, selected_data['count_rh_gp_p25_l'], label='$25^{th}$ Percentile Hispanic Parent Income')

plt.bar(['$25^{th}$ Percentile Parent Income', '$25^{th}$ Percentile Parent Income (Late Cohort)', 
    'Female Parent Income (Late Cohort)', '$25^{th}$ Percentile Female Parent Income', '$25^{th}$ Percentile Female Parent Income (Late Cohort)',
    '$25^{th}$ Percentile Male Parent Income', '$25^{th}$ Percentile Male Parent Income (Late Cohort)', 
    'Hispanic Parent Income (Late Cohort)', '$75^{th}$ Percentile Hispanic Parent Income (Late Cohort)', 
    '$25^{th}$ Percentile Hispanic Parent Income'], selected_data.mean())

plt.title('Data Driven Correlations Between Certain Socioeconomic Variables and the Epidemiologic Profiling of COVID-19')
plt.xticks(rotation=30, fontsize=8)
plt.xlabel('Data')
plt.ylabel('Average Value')
plt.show()