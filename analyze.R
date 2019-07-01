OVS = read.csv("/u/scr/mhahn/japanese/4705973", sep="\t")
SOV = read.csv("/u/scr/mhahn/japanese/5391317", sep="\t")
OVS$Type = "OVS"
SOV$Type = "SOV"

data = rbind(SOV, OVS)

library(lme4)

summary(lmer(Length ~ Type + (1|Sent), data=data))

library(dplyr)
library(tidyr)

data2 = data %>% spread(Type, Length) %>% mutate(OVSBigger = OVS > SOV, Same = OVS == SOV, SOVBigger = SOV > OVS)








# 4-dlm.py
OVS = read.csv("/u/scr/mhahn/japanese/6197316", sep="\t")
SOV = read.csv("/u/scr/mhahn/japanese/81496", sep="\t")
OVS$Type = "OVS"
SOV$Type = "SOV"

data = rbind(SOV, OVS)

library(lme4)

summary(lmer(Length ~ Type + (1|Sent), data=data))

library(dplyr)
library(tidyr)

data2 = data %>% spread(Type, Length) %>% mutate(OVSBigger = OVS > SOV, Same = OVS == SOV, SOVBigger = SOV > OVS)



# 5-dlm.py
OVS = read.csv("/u/scr/mhahn/japanese/2291345", sep="\t")
SOV = read.csv("/u/scr/mhahn/japanese/5831823", sep="\t")
OVS$Type = "OVS"
SOV$Type = "SOV"

data = rbind(SOV, OVS)

library(lme4)

summary(lmer(Length ~ Type + (1|Sent), data=data))

library(dplyr)
library(tidyr)

data2 = data %>% spread(Type, Length) %>% mutate(OVSBigger = OVS > SOV, Same = OVS == SOV, SOVBigger = SOV > OVS)






# 6-dlm.py
OVS = read.csv("/u/scr/mhahn/japanese/5620916", sep="\t")
SOV = read.csv("/u/scr/mhahn/japanese/3830390", sep="\t")
OVS$Type = "OVS"
SOV$Type = "SOV"

data = rbind(SOV, OVS)

library(lme4)

summary(lmer(Length ~ Type + (1|Sent), data=data))

library(dplyr)
library(tidyr)

data2 = data %>% spread(Type, Length) %>% mutate(OVSBigger = OVS > SOV, Same = OVS == SOV, SOVBigger = SOV > OVS)




# 7-dlm.py
OVS = read.csv("/u/scr/mhahn/japanese/1677823", sep="\t")
SOV = read.csv("/u/scr/mhahn/japanese/8972605", sep="\t")
OVS$Type = "OVS"
SOV$Type = "SOV"

data = rbind(SOV, OVS)

library(lme4)

summary(lmer(Length ~ Type + (1|Sent), data=data))

library(dplyr)
library(tidyr)

data2 = data %>% spread(Type, Length) %>% mutate(OVSBigger = OVS > SOV, Same = OVS == SOV, SOVBigger = SOV > OVS)






# 9-dlm.py -- astonishing even in spoken corpus!
OVS = read.csv("/u/scr/mhahn/japanese/7245787", sep="\t")
SOV = read.csv("/u/scr/mhahn/japanese/1941432", sep="\t")
OVS$Type = "OVS"
SOV$Type = "SOV"

data = rbind(SOV, OVS)

library(lme4)

summary(lmer(Length ~ Type + (1|Sent), data=data))

library(dplyr)
library(tidyr)

data2 = data %>% spread(Type, Length) %>% mutate(OVSBigger = OVS > SOV, Same = OVS == SOV, SOVBigger = SOV > OVS)



# 10-dlm.py -- astonishing even in spoken corpus!
OVS = read.csv("/u/scr/mhahn/japanese/2605995", sep="\t")
SOV = read.csv("/u/scr/mhahn/japanese/7746334", sep="\t")
OVS$Type = "OVS"
SOV$Type = "SOV"

data = rbind(SOV, OVS)

library(lme4)

summary(lmer(Length ~ Type + (1|Sent), data=data))

library(dplyr)
library(tidyr)

data2 = data %>% spread(Type, Length) %>% mutate(OVSBigger = OVS > SOV, Same = OVS == SOV, SOVBigger = SOV > OVS)



# 11-dlm.py -- astonishing even in spoken corpus!
OVS = read.csv("/u/scr/mhahn/japanese/8999052", sep="\t")
SOV = read.csv("/u/scr/mhahn/japanese/780644", sep="\t")
OVS$Type = "OVS"
SOV$Type = "SOV"

data = rbind(SOV, OVS)

library(lme4)

summary(lmer(Length ~ Type + (1|Sent), data=data))

library(dplyr)
library(tidyr)

data2 = data %>% spread(Type, Length) %>% mutate(OVSBigger = OVS > SOV, Same = OVS == SOV, SOVBigger = SOV > OVS)





# 12-dlm.py -- astonishing even in spoken corpus!
OVS = read.csv("/u/scr/mhahn/japanese/3835606", sep="\t")
SOV = read.csv("/u/scr/mhahn/japanese/3873780", sep="\t")
OVS$Type = "OVS"
SOV$Type = "SOV"

data = rbind(SOV, OVS)

library(lme4)

summary(lmer(Length ~ Type + (1|Sent), data=data))

library(dplyr)
library(tidyr)

data2 = data %>% spread(Type, Length) %>% mutate(OVSBigger = OVS > SOV, Same = OVS == SOV, SOVBigger = SOV > OVS)





