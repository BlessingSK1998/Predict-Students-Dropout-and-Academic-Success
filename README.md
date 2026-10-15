Predicting Student Dropout and Academic Success
A machine-learning research notebook examining how accurately student academic outcomes can be predicted at different stages of study, and whether those predictions are suitable for supporting student counseling decisions.
Project overview
The project predicts three student outcomes: Dropout, Enrolled, and Graduate. It compares prediction performance using data available at different points in a student's academic journey, then examines generalization across degree courses, uncertainty, practical counseling capacity, and algorithmic fairness.
The main analysis is in [`Copy_of_Predict_Students'_Dropout_and_Academic_Success.ipynb`](Copy_of_Predict_Students'_Dropout_and_Academic_Success.ipynb).
Research objectives
- Compare early-enrolment predictions with predictions after the first and second semesters.
- Benchmark a majority-class baseline, regularized logistic regression, and histogram-based gradient boosting.
- Estimate model performance using repeated stratified cross-validation.
- Test generalization to courses not represented in training using GroupKFold on Course.
- Investigate permutation feature importance, calibration, and conformal prediction sets.
- Evaluate how well high-risk rankings could prioritize students when counseling capacity is limited.
- Audit potential disparities across gender, nationality/international status, and scholarship groups.
Dataset
The notebook expects a CSV at /content/data 2.csv, with a Target column and student-level attributes covering admissions, demographics, finances, semester results, and macroeconomic indicators.
The CSV dataset is not included with this notebook. To run the analysis, obtain the appropriately authorized dataset and put it at the specified path, or change the path in the loading cell to your local CSV location. The code uses delimiter auto-detection with pandas.read_csv(..., sep=None, engine='python').
The analysis expects the column names used in the notebook (including the original spelling Nacionality).
Feature timelines
Timeline	Information available
1 — Enrolment only	Admission, previous qualifications, student background, demographic and financial-status indicators
2 — After semester 1	Timeline 1 plus first-semester curricular activity, grades, and credits
3 — After semester 2	Timeline 2 plus second-semester curricular measures and selected macroeconomic indicators


Categorical preprocessing is performed within model pipelines, helping to avoid leakage between cross-validation training and evaluation folds.
Methods
- Baseline: DummyClassifier predicting the majority class.
- Linear model: Regularized multinomial logistic regression.
- Nonlinear model: HistGradientBoostingClassifier.
- Validation: Repeated stratified five-fold cross-validation (three repeats) and course-grouped cross-validation.
- Interpretability: Permutation importance; predictive association is not interpreted as causation.
- Uncertainty: Class-wise calibration curves and split conformal prediction.
- Decision support: Precision at a fixed counseling capacity (k = 100).
- Fairness: Error rates, selection rates, and false-positive/false-negative comparisons for selected subgroups.
Results reported in the notebook
These are reported notebook findings, not independently rerun results.
Model or validation setting	Accuracy	Macro F1
Majority baseline, enrolment only	49.93%	0.2220
Logistic regression, enrolment only	60.79%	—
Gradient boosting, enrolment only	62.24%	0.5256
Logistic regression, through semester 1	70.12%	—
Gradient boosting, through semester 1	73.88%	0.6560
Logistic regression, through semester 2	73.52%	—
Gradient boosting, through semester 2	77.43%	0.7042
Gradient boosting, unseen-course GroupKFold	74.57%	0.6661


Additional results summarized in the notebook include:
- The highest-ranked permutation feature was second-semester curricular units approved.
- Conformal prediction targeting 95% coverage reportedly achieved 95.93% empirical test coverage in the evaluated split.
- Reported precision-at-100 was 99% in the analyzed counseling simulation. This is an evaluation result, not a guarantee for future students.
- The fairness audit identified differences in error rates and risk selection across student groups, supporting careful evaluation before any operational use.
How to run
1. Open the .ipynb file in Google Colab or Jupyter Notebook/JupyterLab.
2. Make the dataset accessible and update the CSV path as necessary.
3. Install dependencies (in Colab, the notebook already contains a category-encoders installation cell):
   pip install numpy pandas scikit-learn category-encoders matplotlib jupyter
4. Run the notebook cells in order, starting from data loading. Some later analyses depend on variables and models defined by earlier cells.
5. Review the generated metrics, plots, explanatory analyses, and comprehensive synthesis at the end of the notebook.
Environment note: The notebook uses OneHotEncoder(sparse_output=False), which requires a compatible version of scikit-learn (1.2 or newer). It is organized as an exploratory Colab notebook, rather than a packaged application or fully automated pipeline. The analyses have not been verified as runnable in a fresh environment without changes.
Limitations and responsible use
- Single-institution data: Findings may not generalize to different universities, student populations, or policies.
- Time-dependent features: Semester 1 and semester 2 models cannot be used before those grades and records become available.
- Association is not causation: High importance of grades or tuition status does not establish why a student is at risk.
- Fairness and small groups: Error-rate disparities and small subgroup sizes require further investigation and monitoring.
- Calibration and uncertainty: Estimated coverage and observed top-k precision are specific to evaluation conditions and can change after deployment.
- Student impact: These predictions should support, not replace, confidential, human-led student advising. They should not be used as the sole basis for high-stakes decisions about individual students.
Suggested repository structure
student-dropout-prediction/
├── README.md
├── Copy_of_Predict_Students'_Dropout_and_Academic_Success.ipynb
└── data 2.csv                 # local dataset, if authorized; not supplied here
Project status
Exploratory research notebook with model comparisons, validation, interpretability, uncertainty analysis, counseling-prioritization simulation, and fairness auditing. Not a production deployment.
