# Wyatt's Paper Notes

> Format for each paper: (1) What the paper does, (2) One thing that surprised you, (3) Which part of our project it relates to.

## Week 1

### McMahan et al. 2017 (FedAvg)
1. What it does: The basis for federated learning is to have minimal updates to the server's global state. Since each bit of data will be sent quickly, as well as the bare minimum to update the model, that provides added privacy and security. With the FedAvg part, each client will do a loss calculation step multiple times before the overall averaging takes place in sync with other clients.
2. Surprising: While I did forget the fact that data loss does happen when sending or receiving, I did not honestly know there was a way to inhibit this loss in such an easy way.
3. Relates to: Taking data from both silos to update the main JayQuery server without having any excess data.

### Yang et al. 2019 (VFL taxonomy)
1. What it does: This gives a framework, explanation, and architecture on the different styles of federated learning, horizontal, vertical, and federated transfer learning. It also goes into some depth about different security and privacy measures. The different ways of federated learning is what part of the data is collected. Horizontal is good for having strong similarities in data one way, but very little overlap the other way, in this case, the features are primarily similar even if the samples are not. The security of it assumes honest participants, with security against an honest and curious server where the server is the primary threat for compromise. Vertical is useful for the opposite of horizontal, good for similar samples, even if the features are not the same. The security of it assumes honest and curious participants, with an attacker in control of one of two parties, the only data they get from the other party is the input/output, nothing more. Federated transfer learning is best with two datasets with very little overlap. The security is similar to that of vertical federated learning.
2. Surprising: "They show that it tolerates up to a constant fraction of Byzantine workers." I had to look this sentence up, as I had never seen it before. The idea relates to security in how even if an attacker controls a certain percentage of nodes, the system is still able to stay functional and more importantly, secure. The actual game theory study is the Byzantine fault, which in essence, is as described.
3. Relates to: This project uses VFL, the paper provides good detail on various parts of VFL and some security/encryption methods.

### Vepakomma et al. 2018 (SplitNN)
1. 
2. 
3. 

### Kuzilek et al. 2017 (OULAD)
1. 
2. 
3. 

---
> Continue adding one entry per paper each week. Share with Dr. Li before Monday meeting.
