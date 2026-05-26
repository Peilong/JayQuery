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
