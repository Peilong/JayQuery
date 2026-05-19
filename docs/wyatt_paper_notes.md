# Wyatt's Paper Notes

> Format for each paper: (1) What the paper does, (2) One thing that surprised you, (3) Which part of our project it relates to.

## Week 1

### McMahan et al. 2017 (FedAvg)
1. What it does: The basis for federated learning is to have minimal updates to the server's global state. Since each bit of data will be sent quickly, as well as the bare minimum to update the model, that provides added privacy and security. With the FedAvg part, each client will do a loss calculation step multiple times before the overall averaging takes place in sync with other clients.
2. Surprising: While I did forget the fact that data loss does happen when sending or receiving, I did not honestly know there was a way to inhibit this loss in such an easy way.
3. Relates to: Taking data from both silos to update the main JayQuery server without having any excess data.

### Yang et al. 2019 (VFL taxonomy)
1. What it does:
2. Surprising:
3. Relates to:

---
> Continue adding one entry per paper each week. Share with Dr. Li before Monday meeting.
