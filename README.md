# Calorie-Prediction

## Doc.

overleaf link:  
google doc link: [Part3](https://docs.google.com/document/d/15J0G9hjMIxExnnnBLDmIgxw9vZXAuAhXgimY3jqOwBI/edit?usp=sharing)

## Result

| features                           | LR     | RF     | NN     | XGB    | GradientBoostingClassifier |
| ---------------------------------- | ------ | ------ | ------ | ------ | -------------------------- |
| Techniques                         | 0.4474 | 0.4528 | 0.4471 | 0.4512 | 0.4474                     |
| Ingredients_ids(500)               | 0.5099 | 0.5596 | 0.4480 | 0.5098 | 0.5117                     |
| Ingredients_ids(1000)              | 0.5596 | 0.5817 | 0.4496 | 0.5246 | 0.5219                     |
| Techniques + Ingredients_ids(1000) | 0.5381 | 0.5948 | 0.4521 | 0.5274 | 0.5283                     |
| Glove Ingredients                  | 0.4907 | 0.5399 | 0.5256 | 0.4928 | 0.4929                     |



## Ref.

- [Dataset/Kaggle page](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions)
- [Original paper of Prof.](https://github.com/majumderb/recipe-personalization)
- [Text-Analytics-on-Food.com-Recipes-Review-Data](https://github.com/kbpavan/Text-Analytics-on-Food.com-Recipes-Review-Data-)
- [Predict Food Recipe Ratings](https://github.com/Jimmy-Nguyen-Data-Science-Portfolio/Predict-Food-Recipe-Ratings)
- [Multi-Task Learning for Calorie Prediction on a Novel Large-Scale Recipe Dataset Enriched with Nutritional Information](https://arxiv.org/abs/2011.01082)
