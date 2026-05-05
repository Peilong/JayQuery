# JayQuery: Private-Preserving Student Success Prediction with Federated Learning

## Introduction: 

Higher education institutions are data-rich but insight-poor due to strict privacy regulations. To accurately predict student success (e.g., identifying dropout risks), administrators need a holistic view that combines academic records (grades, attendance),
financial aid status, and behavioral metrics. However, the Family Educational Rights and Privacy Act (FERPA) and institutional data governance policies often trap this data in "silos"--the Registrar, the Financial Aid Office, and the Learning Management System (LMS)
operate as disconnected islands. Merging these datasets into a central "data lake" for machine learning creates a single point of failure and significant privacy liability.

This project, JayQuery, addresses this critical gap by developing a Vertical Federated Learning (VFL) system. Unlike standard "horizontal" federated learning where devices share the same data structure (e.g., mobile phones improving a keyboard app), VFL is
designed for organizations that hold different attributes for the same individuals. The objective is to train a unified "Student Success" neural network across these simulated silos without ever exposing the raw data or establishing a shared identifier list that could
compromise student anonymity.
