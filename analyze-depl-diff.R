library(tidyr)
library(dplyr)

data = read.csv("results/japanese/depl-difference.tsv", sep="\t", row.names=NULL)

data2 = data %>% group_by(Dependency) %>% summarise(Number = NROW(Difference), Reduction = sum(Difference < 0) / sum(Difference!=0), Mean = mean(Difference), SD = sd(Difference)) %>% mutate(t=Mean/(SD / sqrt(Number)))


data2[order(-data2$Number),] %>% print(n=40)


data = read.csv("results/japanese/depl-perSent-Japanese_2.4.tsv", sep="\t", row.names=NULL)
data$Diff = data$Real - data$Counterfactual
summary(lm(Diff ~ 1, data=data))



data = read.csv("results/japanese/depl-perSent-Japanese-GSD_2.4.tsv", sep="\t", row.names=NULL)
data$Diff = data$Real - data$Counterfactual
summary(lm(Diff ~ 1, data=data))

data = read.csv("results/japanese/depl-perSent-Japanese-BCCWJ_2.4.tsv", sep="\t", row.names=NULL)
data$Diff = data$Real - data$Counterfactual
summary(lm(Diff ~ 1, data=data))

data = read.csv("results/japanese/depl-perSent-Japanese-KTC_2.4.tsv", sep="\t", row.names=NULL)
data$Diff = data$Real - data$Counterfactual
summary(lm(Diff ~ 1, data=data))


