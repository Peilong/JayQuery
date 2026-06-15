# Alex's Paper Notes

> Detailed notes are stored locally on my machine in an obsidian vault. Available to be shared if Wyatt or Dr. Li would like to look over them.

> Format: After each paper, write: (1) Key design choice it informs, (2) What you would implement differently.

## Week 1

### McMahan et al. 2017 (FedAvg)
- Design choice: This paper covers the basics of Federated Learning, and introduces the parameters that can be tuned for better optimization on runtimes by reducing the communication cost, and possibly the computation costs (though this is considered secondary).
- Implementation note: I don't have any piece of this I would implement differently at this time. I anticipate portions of this implementation changing as I read more papers and start the JayQuery Implementation itself.

### Yang et al. 2019 (VFL taxonomy)
- Design choice: This paper covers a lot of different topics. The most important sections to our project are on Vertical Federated Learning, and the architecture to create such a machine learning model implementation. It also covers a number of security considerations for federated learning, with the most interesting of which being SMC and Homomorphic Encryption. It also introduced Sharemind, an SMC framework for machine learning models that seems to be similar to what we want to achieve. We will also be taking a look at DP, which was introduced in this paper.
- Implementation note: This paper really helped inform me on a number of the security concerns that we need to take into account as we implement JayQuery. I found this to be a good resource as we start the project, and I will definitely be returning to this paper as we confirm our security methods.

### Vepakomma et al. 2018 (SplitNN)
- Design choice: This paper explores a Distributed Deep Learning (DDL) method called SplitNN. It goes into depth on how this is applicable for healthcare operations, and the benefits it offers over other DDL architectures. The paper also includes several SplitNN configurations, how they work, and why you might choose them. 
- Implementation note: When I first read this paper, I was under the impression that VFL and SplitNN were somehow different architectures. It was reinforced when reading the experiments section and the authors compared the efficiency of "Federated Learning" and "SplitNN". Upon further consideration and research, it has become more evident that VFL and SplitNN address different aspects of distributed learning. After clearing this confusion, I was able to much more clearly understand the direction of the project, and how I will go about implementing these DDL models. At this point, SplitNN seems to align closely with my intuition for how model partitioning could be used in a distributed learning format. I have no comments yet about changing these architectures to better suit the project.

---

## Week 2

### Thapa et al. 2022 (SplitFed)
- Design choice: This paper focus on a hybrid architecture of Federated Learning and Split Learning, called SplitFed (SFL). The core problem this solves is the slowness of SL with many clients, along with better privacy and resource usage of FL. 
- Implementation note: This is almost precisely aligned with what we are looking to implement with JayQuery. The difference in implementation will lie in the federated learning design, as this paper only explores horizontal federated learning. We will not be using FedAvg either, but this type of architecture is exactly what we are trying to achieve with our project.

### Romero et al. 2010
- Design choice: This paper focuses on data mining in relation to educational institutions. While it is a bit older (2010), and the machine learning algorithms are somewhat outdated, there is good information on the data collection and concerns with this type of project.
- Implementation note: There is no real implementation techniques to take from this paper in regards to our VFL or SplitNN model. However there are some considerations on how this type of project could be used and directed for an educational institution. The student success could be just the first iteration of use for a prediction model like this one.

### Kairouz et al. 2021
- Design choice: This paper focuses on pretty much every topic I have explored so far. This is a good aggregation of the different papers and concepts that have been introduced thus far. This is also the first paper I have seen to specifically mention using SplitNN with a VFL data architecture.
- Implementation note: This is an extremely useful resource, as it functions as a sort of generic lookup of where to find more detailed information relating to SpliNN and Federated Learning. Nothing specifically seems worth seeking to implement at this current point in the project, but useful notes to have on hand as we go further into the project.

---

## Week 3

### Kotsiantis et al. 2010
- Design choice: This paper covers the usage of an ensemble model with three classifiers to predict student success. This doesn't introduce really all that much for the concepts they use, but rather show that this format is better than a traditional batch-ensemble because it can accomodate new features mid-stream.
- Implementation note: This paper shows that what we are trying to achieve as a final outcome is possible and useful, but the architectures discussed here are not really anything we would want to implement on our end. It is interesting reading if only for the sake of understanding how others have attempted this problem in the past, but not much for how we are going to implement any piece of our architecture.

### Chawla et al. 2002 
- Design choice: This paper introduces SMOTE. Rather than duplicating existing minority samples (which can cause overfitting), SMOTE generates new synthetic samples by interpolating between real ones in feature space. This forces the classifier to learn broader, more general decision regions for the minority label. The combination of SMOTE on the minority class and random under-sampling on the majority class together consistently outperforms either approach alone.
- Implementation note: I use this in my prepare_silos.py file via imbalanced-learn's SMOTE. This directly alters the number of samples we see in the output of this file.

### Xu et al. 2021
> The original paper list under Xu on the research plan was actually Qiu's paper in 2023. I have also added Qiu's paper to the reading list just in case.
- Design choice: This paper proposes FedV, a VFL framework that eliminates peer-to-peer communication between parties by using Functional Encryption for gradient computation. The most important potion is that gradient aggregation in VFL can be decomposed into two inner-product operations — one across features, one across samples — both of which FE can handle non-interactively. It also extends cleanly to logistic regression and SVMs without Taylor approximation, which prior work required.
- Implementation note: FedV is not something we are implementing, we plan to use SplitNN via Flower instead, which is simpler and better suited to deep neural networks. However, this paper is useful for understanding why the communication design of a VFL system matters. The key practical takeaway is that FedV trades a simpler implementation for stronger cryptographic guarantees, and JayQuery trades those guarantees for flexibility and deep learning compatibility, which is why we have the DP wrapper to compensate.

### Qiu et al. 2023
- Design choice: 
- Implementation note:
---

## Week 4

### Fawcett 2006
- Design choice: This paper covers the metrics of ROC and AUC. It explains how it is more resistant to class-imablance in data, and why it is a good metric to use for judging classifiers. 
- Implementation note: We are planning to use ROC and AUC to score our models, due to the class imbalance we have in our data. This paper is a good resource on how we will be implementing it, as well as why we decided to use this metric.

### Chen & Guestrin 2016
> Section 2
- Design choice: This paper covers the different portions of XGBoost, and how the different parameters change the model as a whole.
- Implementation note: We use XGBoost in our baselines, but we do not plan to use it in the production workflow. This paper is helpful in understanding the algorithm and how it is a good baseline for our problem, but it lacks any importance in the implementation of our project.

---