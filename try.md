# Lime Code Review
## Short introduction
  First make a short introduction for LIME. LIME(Local Interpretable Model-Agnostic) is a class_agnostic model which is to explain the classifer of a black box model. It explains what makes and affects the model implementing classification in a human-readable way and can explain one individual prediction once. The basic methodology is to generate a new dataset consisting of some perturbed data around the target prediction and try to train a relatively simple model(most are linear) to obtain the same or similar results with the original model. More information can be found in https://arxiv.org/abs/1602.04938
## Methodology and algorithm

  Figure A shows the original model, representing by blue and gray regions. And if we try to explain an instance which is represented by the yellow point shown in Figure B, the first step is to generate a series of new instances(blakc points in Figure B). These perturbed data will be tested in the original model and obtain corresponding predictions. However, the distance between perturbed data and the instance to explain will affect the explaination(which are the weights). The weight is shown in Figure C and bigger black area means higher weight. Considering the weights and minimizing the difference between true prediction and the prediction of lime, a new linear locally learned sparse model can be generated as shown in Figure D. This linear model is simplier and can be explained, which means can output what features affect this prediction. Detailed algorithm can be found in https://arxiv.org/abs/1602.04938.
  ![Image](https://github.com/yongkangzzz/practice/blob/main/Lime.png)

##Code
  The code can be found in https://github.com/marcotcr/lime.
  The most important object is **limebase**, which is an abstract class for this linear locally learned model. Image,text and table data based lime model will be based on this abstract class. It includes several parameters: **kernel_fn** is the function to transform array of distances into floats;**random_state** is numpy.RandomState.
  
  ### limebase
  **limebase** includes some functionalities:
  **generate_lars_path(weighted_data, weighted_labels)**: Applying lasso algorithm and return lars path for **weighted_data**(alphas and coef for lasso). **weighted_labels** is the corresponding labels.
  
  **forward_selection(self, data, labels, weights, num_features)**: This will apply **Ridge regression** and implement an iteration to evaluate features(number is **num_features**):  
  **clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)  
  used_features = []  
  clf.fit(data[:, used_features + [feature]], labels,sample_weight=weights)**  
According to the score obtained the features will be added to a feature array(**used_features**) from the end to the first. Return this feature array.
  
  **feature_selection(self, data, labels, weights, num_features, method)**: This is to select an array of features according to the method(forward selection for example).
  
  **explain_instance_with_data(self,neighborhood_data,neighborhood_labels,distances,label,num_features,feature_selection='auto',model_regressor=None)**: This is the core functionality of this object. It will input perturbed data(**neighborhood_data,neighborhood_labels**),labels and distance and to output explanation. Distance will be weighted with kernel: **weights = self.kernel_fn(distances)** and features will be selected with **feature_selection** method. After that, Ridge regression will be applied to build a locally learned linear model. With this model, explanation including features(number is **num_features**), coef,,lamda(Ridge model) and local prediction will be output. 
 
 ### Example(lime_image)
 Take lime_image as an example and it will import lime_base.
 The first class is **ImageExplanation** including target image **image** and **segment** from sklearn segmentation algorithm.
 The core class is **LimeImageExplainer** and this will explain the prediction. First step in initialization function is to use kernel to generate perturbed data:  
        **kernel_fn = partial(kernel, kernel_width=kernel_width)  
        self.random_state = check_random_state(random_state)  
        self.feature_selection = feature_selection  
        self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)**  
 And the explain functionality is **explain_instance** where the key parameters include **image**, **classifier_fn**(classifier function, e.g. classifier.predict_proba for scikitClassifier) and **distance_metric**(vary according to image,text or table).  
 **segments = segmentation_fn(image)** can build a segment and this will be used in **ret_exp = ImageExplanation(image, segments)** where ret_exp is the explanation class for the target image(instance). After parameters like **label** set, ret_exp will invoke the fiunctionality **ret_exp.local_pred[label]) = self.base.explain_instance_with_data** which can be found in lime_base and ret_exp will be returned. A real example can be found in https://marcotcr.github.io/lime/tutorials/Tutorial%20-%20images.html. 
 
 The output format can be found in **Explanation.py** including: **as_list,as_map, as_pyplot_figure** and so on.
 
    
