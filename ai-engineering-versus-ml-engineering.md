
The rapid growth of AI has induced an incredible amount of hype and FOMO. The number of new tools, techniques, models, and applications introduced everyday can be overwhelming. 

As someone with a math background and was trained in the disciple of statistical machine learning and deep learning and now have become interested in AI for the last two years, I find that many of my colleagues just don't have the model to understand how modern LLMs work and how they are fundamentally different from classical machine learning models.

To understand AI engineering, its important to recognize that AI engineering evolved out of ML engineering. Many principles of building applications remain the same. 
- ML/AI applications need to solve business problems
- its still essential to map from business metrics to ML metrics and vice versa 
- we still need to do systematic experimentation
	- Classical ML engineering we experiment with different models/hyperparameters
	- with foundation models we experiment with different prompts, retrieval algorithms, sampling variables, and more
- important to set up feedback looks so that we can iteratively improve our application with production data

While the principles of deploying AI applications have significant overlap, its also important to understand how things have changed. 

At a high level, building applications using foundation models today differs from traditional ML engineering in three major ways:
1. With AI engineering, we use a model someone else has trained for us. This means AI engineering focuses less on modelling and training, and more on model adaption
2. AI engineering works with models that can produce open-ended outputs. Open ended outputs give models the flexibility to be used for more tasks, but they are also harder to evaluate. This make evaluation a much bigger problem in AI engineering. 
3. The models are much bigger, consume more compute resources, and incur higher latency than traditional ML models. This means there is more pressure for efficient training and inference optimization. 

AI engineering differs from ML engineering in that it's less about model development and more about adapting and evaluating models. Developing ML model requires specialized ML knowledge. It requires knowing different types of ML algorithms (such as clustering, logistic regression, decision trees, and collaborative filtering) and neural network architectures (such as feedforward, recurrent, convolutional, and transformer). Developing ML models also requires understand how a model learns, including concepts such as gradient descent, loss function, regularization, etc. 

This is not the case for AI engineering. With the availability of foundation models, ML knowledge is no longer a must-have for building AI applications. However, ML knowledge is still extremely valuable, as it expands the set of tools that you can use. 

I want to make sure we are on the same page about what I mean by model adaption. In general model adaption techniques can be divided into two categories:
- **Prompt-based techniques:** adapt the model by giving it instructions and context instead of changing the model itself
- **Fine-tuning:** adapt the model by updating the model weights 

To best understand AI engineering and how it differs from traditional ML engineering, its useful to breakdown different layers of AI application building process and look at the role each layer plays in AI engineering and ML engineering.
### Three Layers of the AI stack
There are three layers to any AI application stack: application development, model development, and infrastructure. When developing an AI application, you'll likely start from the top layer and move down as needed. 
**Application development**
- With foundation models readily available, anyone can use them to develop applications. This is the layer that has seen the most action in the last two years, and it is still rapidly evolving. Application development involves providing a model with good prompts and necessary context. This layer requires rigorous evaluation. 
**Model Development**
- Model development (modelling and training, dataset engineering) is the layer most commonly associated with traditional ML engineering. 

### Differences Among Training, Pre-Training, Fine-tuning, and Post-Training
