# COMP4010 - Data Visualization - Project 2 - Report

**Group G: Dinh Van Thanh, Do Huu Dat, Ha Phuong Thao**

## Introduction

In today's data driven world, machine learning has become a powerful tool for extracting valuable insights from complex datasets. However, the complexity associated with setting up and understanding machine learning models often pose a challenge to entry for many individuals, especially those without solid technical background. As a result, in this project, we aim to address this issue by providing a user-friendly simulation of a machine learning pipeline using R Shiny. Through this platform, users can gain more intuition about how machine learning algorithms work, further enhancing the productivity in their learning process.

### Problem Statement.

The complexity of machine learning often deters individuals with non-technical backgrounds from exploring its potential applications. There are two biggest barriers for newbies and non-technical people:
Technical knowledge: Traditional machine learning platforms require programming knowledge and familiarity with complex algorithms. 
Technical set-up: Setting up the necessary environments and dependencies can be time-consuming and challenging for beginners, unlike analyzing data by tools such as Excel or PowerBI.	

Therefore, our goal is to democratize machine learning education by creating a platform that simplifies the process of building and understanding machine learning models, making it accessible to a wider audience.

### Dataset Description.

For simplicity, in our project, we will demonstrate different steps in a machine learning pipeline by using one Iris dataset.

The Iris dataset is a classic dataset in the field of machine learning and is often used for introductory purposes due to its simplicity and well-defined structure. It was introduced by the British statistician and biologist Ronald Fisher in his paper "The Use of Multiple Measurements in Taxonomic Problems" in 1936.

The Iris dataset consists of 150 samples of iris flowers, each belonging to one of three species: Setosa, Versicolor, and Virginica. For each sample, four features were measured:

- Sepal Length: Length of the sepals (the green leaf-like structures that protect the flower bud) in centimeters.
- Sepal Width: Width of the sepals in centimeters.
- Petal Length: Length of the petals (the colored part of the flower) in centimeters.
- Petal Width: Width of the petals in centimeters.

The target variable in the Iris dataset is the species of each iris flower, which is categorical and can take one of three possible values:

- Setosa
- Versicolor
- Virginica

The Iris dataset is commonly used for supervised learning tasks, particularly classification, where the goal is to predict the species of an iris flower based on its four features (sepal length, sepal width, petal length, and petal width). It serves as an excellent dataset for learning and practicing classification algorithms, such as logistic regression, decision trees, support vector machines, k-nearest neighbors, etc.

## Justification of Approach

Our approach focuses on leveraging the power of data visualization and interactivity to simplify the machine learning process. By providing a visual and intuitive interface, users can interact with different components of the machine learning pipeline, gaining a deeper understanding of how data is processed and models are trained. This approach differs from traditional machine learning platforms, which often require users to write code and understand complex algorithms without visual aids.

## Final Product

Our final product is a machine learning pipeline with different tabs and each tab representing a step in the pipeline with the description:
Data Upload: User will upload a chosen dataset
Data Summary: Give an overview by providing a data summary and a simple distribution plot
Data Exploration: Preprocess data by deleting/filling NAs and creating simple plots
Build Model: Pick variables, split train/test set, choose algorithms
Evaluation: Evaluate the model performance

### Data Upload

<img width="1210" alt="Screenshot 2024-06-04 at 21 54 02" src="https://github.com/clarissdev/data-visualization-project-2/assets/110231356/3fbc336c-e239-4bb9-ab8f-9d03f4cec8b4">

In this step, users can upload their choice of dataset and have an overview of how the dataset looks like. However, the format of the dataset should be similar to the iris database. After uploading, the user can see the data preview with all variables and entries. Below each variable name, the user can also adjust the filter to see the filtered data.

### Data summary

This tab allows user to observe a summary of their dataset. After uploading the dataset, the user will see a summary of their dataset, including mean, median, standard deviations, etc. These summary statistics help in understanding various aspects of the dataset, such as the central tendency, variability, spread, shape, and precision of the data. They are essential for exploratory data analysis, hypothesis testing, and making informed decisions based on the data.

Below the data summary table is a violin plot to help user look at the distribution of each variable.

<img width="1268" alt="Screenshot 2024-06-04 at 21 55 19" src="https://github.com/clarissdev/data-visualization-project-2/assets/110231356/0dee1d4c-6982-4d45-80b3-53457f67510d">

### Data exploration

In this stage, we provide the user with three types of visualization: one/two variable plot, a pairplot and a 3D scatter plot. Users can input their choices of variables on the side bar panel to generate a desired plot. 

### Model building

Visualizations constitute a foundational component in the process of constructing and evaluating models for both classification and regression tasks. They serve as instrumental tools for comprehensively assessing model performance and elucidating underlying data characteristics. In this application, we focus primarily on two main types of machine learning algorithms: classification and regression. 

#### Classification
Visualizing classification algorithms provides crucial insights into their operational dynamics and predictive capabilities. For KNN, visualizations typically depict decision boundaries and class distributions, revealing how the algorithm categorizes data points based on their proximity to neighboring instances. Conversely, logistic regression visualizations often feature sigmoid curves illustrating the transformation of predictor variables into class probabilities. In this project, we focus on visualizing K-Nearest Neighbors (KNN) and Logistic Regression 
For K-Nearest Neighbors, after data processing, a visual representation of the dataset, including data samples and decision boundaries, is presented. Users have the flexibility to select the number of nearest neighbors `k` interactively, observing its effect on the decision boundary and classification outcomes. This interactive exploration aids in fine-tuning the model according to specific needs and provides insights into its behavior across different 'k' values.

<img width="794" alt="Screenshot 2024-06-04 at 21 56 04" src="https://github.com/clarissdev/data-visualization-project-2/assets/110231356/275a2fa0-7b62-4d82-888f-5d2d25533534">

In Logistic Regression, we often employ gradient descent for model training, a process that unfolds across numerous steps. To grasp its intricacies, users can select specific intervals to observe the model's evolution. This allows them to witness the model's gradual refinement, enhancing their comprehension of the process. This feature offers users a window into the model's progression, facilitating a deeper understanding of its learning journey and iterative improvement.

<img width="832" alt="Screenshot 2024-06-04 at 21 56 32" src="https://github.com/clarissdev/data-visualization-project-2/assets/110231356/02323cc2-5a4f-4a82-8443-995984455eba">

Logistic Regression at time step 1

<img width="800" alt="Screenshot 2024-06-04 at 21 56 44" src="https://github.com/clarissdev/data-visualization-project-2/assets/110231356/03d11d37-9ee4-4966-bc0f-0282308951c0">

Logistic Regression at time step 3

<img width="817" alt="Screenshot 2024-06-04 at 21 56 57" src="https://github.com/clarissdev/data-visualization-project-2/assets/110231356/89eb3bab-ce72-4e17-88fd-eedf58abf2c2">

Logistic Regression at time step 10
	
#### Regression


# Discussion

Visualizing classification algorithms like KNN and Logistic Regression allowed us to discuss and analyze decision boundaries. These visualizations sparked discussions on potential weaknesses in the models, like overfitting, and guided further exploration and refinement strategies. Similarly, for regression tasks, visualizations of linear regression models led to discussions about the underlying data patterns – uncovering potential non-linear relationships that tabular data might obscure. Neural network visualizations proved particularly valuable, fostering discussions on feature importance and their influence on the model's predictions.  Overall, this project reinforces the notion that visualization is a powerful tool for demystifying the often "black box" nature of ML models. By providing a visual lens, we can engage in richer discussions about model behavior, ultimately leading to more informed decisions and robust solutions. pen_spark

# Limitations

While our platform, built on R Shiny, aims to simplify the process of building and understanding machine learning models for individuals with non-technical backgrounds, there are several limitations that need to be addressed.
- Educational depth vs. usability trade-off: The platform's primary goal is to be user-friendly and accessible, often at the expense of educational depth. Simplifying the interface and process means that the detailed explanations are omitted to avoid overwhelming users. Although we have included some tutorials for in depth explanation, we could only compromise for a high level understanding.
- Restricted flexibility: Although we do provide some parameters for users to experience with, it does not provide the flexibility required for users to deeply engage with and understand machine learning concepts and techniques. Specifically, our platform falls short in allowing hands-on practice with code and providing a flexible learning environment.

# Future directions
