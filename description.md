# Description of submission

Our submission used a gradient boosted random forest for classification. As input we used a subset of the 200 best performing features. These features were identified by calculating the information gain of each one when compared against a proxy gold standard (the answer to the survey   question "In how many years do you plan to have your next child?"). This question (naturally) showed a very high correlation with the target variable, so we concluded that features which perform well in predicting it should also perform well on the PreFer challenge.
