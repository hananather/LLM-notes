https://www.hopsworks.ai/post/mlops-to-ml-systems-with-fti-pipelines
Building production-ready ML systems is much more than just training a model. From an engineering point of view, training the model is the most straightforward step in most use cases. Training the model becomes complex when deciding on the correct architecture and hyperparameters. 

Training a model with high accuracy is extremely valuable, but just by training it on a static dataset, you are far from deploying it robustly.
- Ingest, clean and validate fresh data
- training versus inference setups
- compute and serve feature in the right environment
- serve the model in a cost effective way
- Version, track and share the dataset and models
- Monitor your infrastructure and models
- deploy the model on a scalable infrastructure
- automate the deployments and training
Figure showing the components of the Google Cloud MLOps system requires. Along with ML code, there are many moving pieces. The rest of the system comprises configuration, automation,  data collection, data verification, testing and debugging, resource management, model analysis, process and metadata management, serving infrastructure, and monitoring. The point is that there are many components we must consider when productionizing an ML system. 

The critical question is this: How do we connect all these components into a single homogenous system? we must create a boilerplate for clearly designing ML systems to answer that question. 

Similar solutions exist for classic software. If you zoom out most software applications can be split between a DB, business logic, and UI layer. Every layer can be as complex as needed, but at a high level-overview, the architecture of standard software can be boiled down to the previous three components. 

ML system can be boiled down to three pipelines: feature, training, and inference (similar to the DB, business logic, and UI layers from classic software).

Its essential to understand that each pipeline is a different component that can run on a different hardware, written using a different technology, by a different team, or scaled differently. 

**Feature Pipeline:** Takes in raw data as input, processes it, and outputs the features and labels required by the model for training and inference. Instead of directly passing them to the model, the features and labels are stored inside a **feature stored**. This component allows us to store, version, track and share the features. By saving the features in a feature store, we always have a state for our features. Thus we can easily send the features to the training and inference pipelines. As the data is versioned we can always ensure that the training and inference time features matching, thus we avoid the training-serving skew problem

### The training pipeline
Takes the features and labels from features stored as inputs and outputs to train a model or models. The models are stored in a model registry. Its role is similar to that of feature stores, but this time, the model is a first class citizen. Thus the model registry will store, version track, and share the model with inference pipeline.
Also modern registries support a metadata store that allows you to specify essential aspects of how the model was trained. The most important are features, labels, and their version used to train the model. Thus we will always know that data the model was trained on. 

### The inference pipeline
Takes features and labels from the feature store and the trained model from the model registry. With these two, predictions can be easily made in either batch or real-time mode. 

