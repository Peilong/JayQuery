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
1. What it does: This paper is on SplitNN, a deep learning method. It specifically goes into configurations useful for a medical setting, being able to help diagnose patients by merging data from different areas, mentioning radiology and pathology as the concrete examples. It intends to do this without sharing raw data on the patient, which could result in HIPAA non-compliance. With SplitNN, it showed that in between it, federated averaging, and large batch SGD, SplitNN was among the best, especially with a large amount of clients (500).
2. Surprising: In the case of computational power at 500 clients, I found it surprising that SplitNN was little more than a half of a percent of the other two methods, at 0.03 TFlops compared to 5.89 TFlops.
3. Relates to: SplitNN is an example of what we aim to do with JayQuery, being a program that takes information from separate silos and combines the data into one, while still being legal.

### Kuzilek et al. 2017 (OULAD)
1. What it does: This paper explains the dataset of OULAD, explaining a little on how OU operates as an online university. It also provides the methodology for going from raw student data to the data given in OULAD. It also proceeds to explain how the data is presented for use.
2. Surprising: With over 32 thousand students, the option to opt-out for data-collection not being present really reminded me that 2017 was before all the companies turned to LLMs and the sort, even before the LLMs were able to do what they are able to now.
3. Relates to: This is our testing dataset to make sure JayQuery is functional.

## Week 2

### Kairouz et al. 2021 (FnTML)
NOT FULL READING
1. What it does: Very simply, it is an overview of federated learning, with different problems and challenges yet to be solved with how much research had been done on FL up to that point. As it should, it rattles off a definition for FL for those who were unaware of it.
2. Surprising: I'm glad to see such a thorough exploration of FL going from basics and slowly getting up to speed, when there could have been practically nothing like this until farther down the line.
3. Relates to: This is all federated learning related.

### Pinkas et al. 2018 (ACM CCS)
1. What it does: This entire paper details various PSI protocols and their efficiency, correctness, and security. This leads to a final evaluation of the different protocols with different settings, showing how each protocol reacts to the different inputs.
2. Surprising: The main thing I found surprising was just the sheer volume of different methods and protocols to get, what I consider to be, the same outcome.
3. Relates to: PSI protocols for adoption into the project.

### Dwork & Roth 2014 (FnTCS)
(Chapters 1-4)
1. What it does: These chapters are very focused on privacy and how to protect it by using differential privacy. It also explains a lot of the ups and downs for differential privacy, showcasing some of the downfalls. Chapter 3 goes into techniques and thorems for a whole host of different things from randomization to composition. The fourth chapter prioritizes handling queries and keeping some privacy while doing so. It has some more theorems and thoughts put into it, though as a chapter is heavy on the theorems and explaining them.
2. Surprising: It's surprising just how good the examples are for explaining the different techniques and such in a simple way to help people with practically no experience understand generally what is going on.
3. Relates to: This has a high chance of coming back for PSI or frankly just getting me up to speed on some security things.

## Week 3

### Abadi et al. 2016 (ACM CCS)
NOT FULL READING (Abstract + Sections 1-3)
1. What it does: This paper is on differential privacy, with a specific integration of it alongside deep learning. 
2. Surprising: I am constantly surprised by the explanation of these different privacy algorithms.
3. Relates to: We use deep learning, and we want to keep everything that needs to be private private.

### Huang et al. 2012 (NDSS)
NOT FULL READING (Sections 1-2)
1. What it does: Kept very simply, this paper is wide on PSI protocols. It explains all the simplicities in the little sections I was given to read.
2. Surprising: Garbled circuits is quite interesting as the very basic version of PSI from the way it is explained. 
3. Relates to: This is PSI related, which is necessary to keep privacy in JayQuery.

## Week 4

### Mironov 2017 (IEEE CSF)
NOT FULL READING (Sections 1-3)
1. What it does: This paper begins with the common definition of differential privacy, as well as some common uses for the standard (epsilon, delta)-differential privacy implementation. It goes on to explain Renyi differential privacy, a different form that is claimed to be a stronger definition than the (epsilon, delta)-differential privacy definition. The second section explains a solid amount of other "flavors" of differential privacy, just mostly going through a lot of other varients as well as their relaxations. Section 3 really goes into the Renyi differential privacy. The section defines this version of differential privacy, the bad outcomes is guaranteed to not change in probability throughout the definition. Another claim is that it appears to be robust against auxiliary information, where it will not change unexpectedly, at least from my interpretation of the written mathematical proof of it. Another claim is that with post-processing, the privacy will be preserved, it also has this last claim where it can still stay preserved even under "adaptive sequential composition". I do not know exactly the meaning of this last one, but it appears that even when adaptations are being made to an algorithm, it will stay differentially private. The section finally goes into 2 propositions of use of the differential privacy definition, where it lightly explains the cases and how they support the usage of the Renyi Differential Privacy definition.
2. Surprising: I was not entirely aware of the fact that differential privacy nomenclature used definitions for the different types of uses.
3. Relates to: This explains a version of differential privacy, possibly one that we could end up specifically using. At the very least, it is very informative on the workings of differential privacy.

### Abadi et al. 2016 (ACM CSS)
REST OF READING
1. What it does: This half of the paper goes on about the implementation of the differential privacy model, with all the little bits and pieces of it, alongside a short reasoning on why it is being done. After the implementation is a results section, comparing the various settings, their model to some other models, with multiple datasets, the highlights and the lowlights.
2. Surprising: Looking at the charts, it's odd seeing odd lines, not matching with any basic functions in the slightest.
3. Relates to: This is just an extension of the differential privacy from the previous half of the reading.

## Week 7

### Fredrikson et al. 2015 (ACM CCS)
1. What it does: 
2. Surprising: 
3. Relates to: 

### Zhao et al. 2020 (NeurIPS)
1. What it does: 
2. Surprising: 
3. Relates to: 

## Week 8

### Jin et al. 2021 (NeurIPS)
NOT FULL READING (Sections 1-4)
1. What it does: 
2. Surprising: 
3. Relates to: 

### Li et al. 2022 (ICLR)
NOT FULL READING
1. What it does: 
2. Surprising: 
3. Relates to: 

---
> Continue adding one entry per paper each week. Share with Dr. Li before Monday meeting.
