# Optimizing Customer Retention Under Budget Constraints Using Predictive Modeling

Predicting and saving potential customer churn is a prime concern in the telecom industry. Up to 30% of customers, even with major and reliable service providers, churn every month. Telecom operators often use discounts to hold back these customers. The challenge is to precisely provide these discounts to customers likely to churn in the next billing cycle and optimize outreach given budget constraints.

We used R, a language common among statisticians, for this project. R is particularly designed for working with complex datasets. While Python purists might disagree, I urge them to try libraries such as `caret`, `glmnet`, `rpart`, and `tidyverse` to see the value R adds to modeling tasks.

## Modeling Approach

We experimented with three non-parametric models and one parametric model, fine-tuned based on 10-fold cross-validation (CV) results. While Gradient Boosting Machine (GBM) and Random Forests (RF) were top contenders in CV results, RF performed best on the holdout set.

For classification, threshold selection is crucial. Off-the-shelf techniques like Youden's J were inadequate for this business problem. We tailored a cost function based on reasonable business assumptions, recognizing the higher cost of false negatives compared to false positives. This led to a decision threshold of 20%, ensuring high recall and capturing most potential churns. If we were to summarize the true business finding of our analysis in one line, it would have to be the one on slide 18 - "Using this model, **80%** of our potential churns can be saved using **only 25%** of our retention budget." How? Read on more in the slides deck to find out!

## Repository Contents

- **Slides Deck:** Presents the true business impact and future scope of improvement.
- **Excel File:** Contains calculations for threshold selection.
- **R Code:** 750 lines of R code used to arrive at the results. (Note: This was my first time doing intensive coding in R, so the code might not be as optimized as its Python counterpart would be.)

Feel free to explore and provide feedback!



**Note:** This project was part of a group assignment within the coursework STA 380 Introduction to Machine Learning by [Professor Jared S. Murray](https://stat.utexas.edu/directory/jared-s-murray), and would not have been possible without my teammates [Muhammad Ibrahim](https://www.linkedin.com/in/m-ibrahim2094/) and [Tea McCormack](https://www.linkedin.com/in/teamccormack/).

