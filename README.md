# Ensemble learning in multi agent system
Multi-agent system to predict house prices based on a selected data set.

## Dataset  
[Dataset](https://www.kaggle.com/gabriellima/house-sales-in-king-county-usa) - House Sales in King County, USA

## Flow diagram:  

![alt text](https://github.com/KubaBBB/Ensemble-Learning/blob/master/markdownfiles/diagram.png "Flow diagram")
  
 ### Classifiers  
 Four types of agent classifiers were used to predict house prices:

* Decision Tree
* SVR
* KNN
* Bayesian Ridge 

### Conclusion

Based on the combination of classifiers, it is possible to estimate which types of classifiers, give the best predicted models. It was also observed that in the agent system, where each agent specializes in one area, the predicted model in the master agent is better than the others. The weakness of one predicted model can boost metric value when there is esemble technique used. The metrics are compared by r2 score analysis.

### Example diagrams of ensemble classifiers:  

![alt text](https://github.com/KubaBBB/Ensemble-Learning/blob/master/figures/metric_AGENT_ARITHMETIC_0.png "Ens1")

![alt text](https://github.com/KubaBBB/Ensemble-Learning/blob/master/figures/metric_AGENT_ARITHMETIC_4.png "Ens1")
