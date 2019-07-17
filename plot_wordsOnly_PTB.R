

    data = read.csv("results/ptb/tradeoff-summary.tsv", sep="\t")

    library(tidyr)
    library(dplyr)
    library(ggplot2)


    data2 = data %>% filter(Distance==1) %>% mutate(MI=0, Distance=0)
    data = rbind(data2, data)
    data = data %>% group_by(Model, Condition) %>% mutate(CumulativeMemory = cumsum(Distance*MI), CumulativeMI = cumsum(MI), Surprisal=UnigramCE-CumulativeMI)




    plot = ggplot(data, aes(x=CumulativeMemory, y=Surprisal, group=Model+Condition, fill=Condition, color=Condition)) + geom_line(alpha=0.5)+ theme_classic() + theme(legend.position="none") + xlab("Memory") + ylab("Surprisal") + theme(text = element_text(size=20)) 
    ggsave(plot, file=paste("figures/ptb-memsurp.pdf", sep=""))




