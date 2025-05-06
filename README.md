# Customer Conversion Prediction Model

## Problem Framing
The first step is analyzing the dataset to identify relevant features that help pinpoint potential customers likely to say "yes." Since our goal is to avoid calling customers who would say "no," the `duration` feature becomes irrelevant—it is only known after the call ends.

- This dataset is a research dataset with a predefined order, so checking for missing values or typos is unnecessary.
- The dataset is imbalanced (`yes`: ~11%, `no`: ~89%). Since this is the first version of our model, we will apply **undersampling** to achieve a `yes`/`no` ratio of 1:3, improving our ability to correctly predict "yes" results.

## Drivers of Conversion
After running a **feature importance analyzer**, we identified the **top 10 conversion drivers** ranked by importance:

# Top Important Features

| Feature | Importance Score |
|---------|-----------------|
| nr.employed | 0.1316 |
| default_unknown | 0.1275 |
| emp.var.rate | 0.0681 |
| contact_telephone | 0.0604 |
| default_no | 0.0532 |
| housing_yes | 0.0366 |
| housing_no | 0.0342 |
| month_oct | 0.0295 |
| contact_cellular | 0.0277 |
| cons.conf.idx | 0.0273 |


## Recommendation to Reduce Call Volume
To optimize call efficiency while maintaining revenue, we suggest:

- **Building a machine learning model** using historical data to classify customers likely to convert.
- **Continuously updating the model** with new weekly customer data to improve prediction accuracy.

## Initial Model Version

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|-------|----------|--------|----------|---------|
| 0 (No) | 0.88 | 0.89 | 0.88 | 2,784 |
| 1 (Yes) | 0.65 | 0.62 | 0.64 | 928 |
| **Overall Accuracy** | | **0.82** | | **3,712** |
| **Macro Avg** | 0.76 | 0.76 | 0.76 | 3,712 |
| **Weighted Avg** | 0.82 | 0.82 | 0.82 | 3,712 |

## Comparison: Model vs. Initial Dataset

### Baseline (Full Dataset) Performance
- **Total Calls:** 41,188  
- **Total "Yes" Responses:** 4,640  
- **Profit Calculation:**  
  - **Total Cost:** 41,188 × $8 = $329,504  
  - **Total Revenue:** 4,640 × $80 = $371,200  
  - **Net Profit:** $371,200 - $329,504 = **$41,696**  
- **Conversion Rate:** ~11.27%

### Model Performance
#### Step 1: Estimated "Yes" Predictions
- Precision for predicting "yes" cases: **65%**
- Estimated true positive ("yes" cases correctly predicted) = `4,640 × 0.62 ≈ 2,876`
- **Total Predicted "Yes" Calls** = `2,876 / 0.65 ≈ 4,424`

#### Step 2: Profit Calculation with Model Applied
- **Total Cost:** 4,424 × $8 = $35,392  
- **Total Revenue:** 2,876 × $80 = $230,080  
- **Net Profit:** $230,080  - $35,392 = **$194,688**  
- **Conversion Rate:** ~65%

### Final Performance Comparison

| Approach | Calls Made | Yes Responses | Net Profit | Conversion Rate |
|----------|------------|---------------|------------|-----------------|
| Full Dataset (No Model) | 41,188 | 4,640 | $41,696 | 11.27% |
| Model Applied (Only Predicted "Yes" Calls) | 4,424 | 2,876 | $194,688 | 65% |

## Conclusion
By applying the model, call volume **reduces significantly**, and the **conversion rate increases to 64%**, leading to a **substantial boost in net profit**.

---
_This README presents an initial version of the customer conversion prediction model and outlines how optimizing call targeting can improve efficiency and profitability._
