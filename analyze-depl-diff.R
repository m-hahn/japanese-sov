library(tidyr)
library(dplyr)
library(ggplot2)

data = read.csv("results/japanese/depl-difference.tsv", sep="\t", row.names=NULL)

data2 = data %>% group_by(Dependency) %>% summarise(Number = NROW(Difference), Reduction = sum(Difference < 0) / sum(Difference!=0), Mean = mean(Difference), SD = sd(Difference)) %>% mutate(t=Mean/(SD / sqrt(Number)))


data2[order(-data2$Number),] %>% print(n=40)




data2=data %>% filter(Dependency == "lifted_mark")
plot = ggplot(data=data2, aes(x=Difference)) + geom_bar()
plot = plot + xlim(quantile(data2$Difference, 0.01)-1, quantile(data2$Difference, 0.99)+1)



data2=data %>% filter(Dependency == "nsubj")
plot = ggplot(data=data2, aes(x=Difference)) + geom_bar()
plot = plot + xlim(quantile(data2$Difference, 0.01)-1, quantile(data2$Difference, 0.99)+1)


data2=data %>% filter(Dependency == "obl")
plot = ggplot(data=data2, aes(x=Difference)) + geom_bar()
plot = plot + xlim(quantile(data2$Difference, 0.01)-1, quantile(data2$Difference, 0.99)+1)






data = read.csv("results/japanese/depl-perSent-Japanese_2.4.tsv", sep="\t", row.names=NULL)
data$Diff = data$Real - data$Counterfactual
summary(lm(Diff ~ 1, data=data))






data = read.csv("results/japanese/depl-perSent-Japanese-GSD_2.4.tsv", sep="\t", row.names=NULL)
data$Diff = data$Real - data$Counterfactual
summary(lm(Diff ~ 1, data=data))

binom.test(sum(data$Diff < 0), n=nrow(data), p=0.5, alternative="t")

data2 = data %>% filter(Diff != 0)
binom.test(sum(data2$Diff < 0), n=nrow(data2), p=0.5, alternative="g")


GSD = data %>% mutate(Corpus = "GSD")




data = read.csv("results/japanese/depl-perSent-Japanese-BCCWJ_2.4.tsv", sep="\t", row.names=NULL)
data$Diff = data$Real - data$Counterfactual
summary(lm(Diff ~ 1, data=data))

data2 = data %>% filter(Diff != 0)
binom.test(sum(data2$Diff < 0), n=nrow(data2), p=0.5, alternative="g")


BCCJW = data %>% mutate(Corpus = "BCCJW")



data = read.csv("results/japanese/depl-perSent-Japanese-KTC_2.4.tsv", sep="\t", row.names=NULL)
data$Diff = data$Real - data$Counterfactual
summary(lm(Diff ~ 1, data=data))


data2 = data %>% filter(Diff != 0)
binom.test(sum(data2$Diff < 0), n=nrow(data2), p=0.5, alternative="g")

KTC = data %>% mutate(Corpus = "KTC")

plot = ggplot(data=rbind(GSD, BCCJW, KTC) %>% filter(Diff != 0), aes(x=Corpus, y=Diff)) + geom_violin()
plot = plot + ylim(-20, 20)


plot = ggplot(data=rbind(GSD, BCCJW, KTC) %>% filter(Diff != 0), aes(x=Diff)) + geom_bar() + facet_wrap(~Corpus)
plot = plot + xlim(-40, 40) + xlab("Dependency Length Reduction in Counterfactual Order") + ylab("Sentences")
ggsave(plot, file="figures/dependency_lentgh_differences_byCorpus.pdf", width=6, height=3)





