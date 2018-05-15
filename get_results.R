library(data.table)
library(ggplot2)
library(stringr)
library(glmnet)
library(GGally)

extract_results <- function(file){
  f <- readLines(file)
  winr <- str_extract(grep("^50000 : ", f, value = TRUE), "[0-9\\.]+$")
  line_param <- grep("'REWARD_GAME_WON': ", f, value = TRUE)
  
  
  rew_won <- str_extract(str_extract(line_param, "'REWARD_GAME_WON': -?[0-9\\.]+"), "-?[0-9]+")
  rew_lost <- str_extract(str_extract(line_param, "'REWARD_GAME_LOST': -?[0-9\\.]+"), "-?[0-9]+")
  rew_alr <- str_extract(str_extract(line_param, "'REWARD_ALREADY_SHOWN_FIELD': -?[0-9\\.]+"), "-?[0-9]+")
  rew_nber <- str_extract(str_extract(line_param, "'REWARD_NUMBER_FIELD': -?[0-9]+"), "-?[0-9]+")
  rew_zero <- str_extract(str_extract(line_param, "'REWARD_ZERO_FIELD': -?[0-9\\.]+"), "-?[0-9]+")
  return(c(winr, rew_won, rew_lost, rew_alr, rew_nber, rew_zero, line_param))
}

dt <- data.table(winrate = rep("",200),
                  rew_won = rep("",200),
                  rew_lost = rep("",200),
                  rew_already = rep("",200),
                  rew_number = rep("",200),
                  rew_zero = rep("",200))

path <- "/mnt/izblisbon/Group_proj/array_param/out/"
i <- 1
for(file in list.files(path)){
  temp <- extract_results(paste0(path, file))
  for(j in 1:ncol(dt)){
    set(dt, i, j, temp[j])
  }
  i <- i + 1
}
dt[, (colnames(dt)) := lapply(.SD, as.numeric), .SDcols = colnames(dt)]
dt[, winrate_discrete := cut(winrate, breaks = quantile(winrate, seq(0,1,0.25)), include.lowest = T)]

cor(dt)
model <- lm(winrate ~ ., data = dt)
summary(model)

melt.dt <- melt(dt, id.vars = c("winrate", "winrate_discrete"))

ggplot(melt.dt, aes(y=winrate, x=value)) +
  geom_point() +
  geom_smooth(method="lm") +
  facet_wrap("variable", scales = "free_x") +
  theme_bw()

ggpairs(dt[,1:ncol(dt)], aes(colour = winrate_discrete))
