library(ggplot2)
library(ggh4x)
#library(ggrastr)
#library(ggpattern)
library(ggfortify)
library(dplyr)
library(forcats)
library(stringr)
library(scales)
library(tidyr)
library(tibble)
library(tikzDevice)

INCH.PER.CM <- 1/2.54
#WIDTH <- 17.8*INCH.PER.CM
WIDTH <- 14*INCH.PER.CM
COL.WIDTH <- 8.5*INCH.PER.CM
BASE.SIZE <- 8
SYM.SIZE <- 1.2 ## Symol size in legends
COLOURS.LIST <- c("black", "#E69F00", "#999999", "#009371", "#CFCFCF")

theme_paper <- function() {
    return(theme_bw(base_size=BASE.SIZE) +
           theme(axis.title.x = element_text(size = BASE.SIZE),
                 axis.title.y = element_text(size = BASE.SIZE),
                 legend.title = element_text(size = BASE.SIZE),
                 legend.position="top"))
}

read.data.dwave <- function(file) {
    dat <- read.csv(file, header=TRUE, comment.char="#")
    dat$num_relations <- as.integer(dat$num_relations)
    dat$num_threshold_values <- as.ordered(dat$num_threshold_values)
    dat$num_decimal_pos <- as.ordered(dat$num_decimal_pos)

    return(dat)
}

read.data <- function(file) {
    dat <- read.csv(file, header=TRUE)
    dat$vendoropt <- str_c(dat$vendor, " O", dat$opt_level)
    dat$density <- as.ordered(dat$density)
    dat$num_relations <- as.ordered(dat$num_relations)

    dat$gateset <- recode(dat$gateset, "Any_gates"="Logical", "Native_gates"="Physical")
    
    return(dat)
}

shrink_legend <- function(boxc) {
    return(theme(legend.margin=margin(0,0,0,0),
                 legend.box.margin=margin(boxc,boxc,boxc,boxc)))
}


###########################################################################
### DWave time to optimality
read.time.to.opt.dwave <- function(file) {
    dat <- read.csv(file, header=TRUE)
    dat$query_graph <- as.factor(dat$query_graph)
    dat <- dat[dat$annealing_time!="Guessing",]
    dat$annealing_time <- as.numeric(dat$annealing_time)
    ## first_opt_index is zero-based, so the elapsed time until the first
    ## optimal solution is reached needs to consider one additional annealing
    ## time interval
    dat$opt_time <- dat$annealing_time*(dat$first_opt_index+1)
    
    dat %<>% group_by(annealing_time) %>% mutate(mean_opt_time=mean(opt_time)) %>%
        mutate(median_opt_time=median(opt_time)) %>% ungroup()
    
    return(dat)
}

read.time.to.opt.dwave.rebuttal <- function(file) {
    dat <- read.csv(file, header=TRUE)
    dat$query_graph <- as.factor(dat$query_graph)
    dat <- dat[dat$annealing_time!="Guessing",]
    dat$annealing_time <- as.numeric(dat$annealing_time)

    ## TMP: Filter out runs when no solution was reached
    dat %<>% filter(first_opt_index != "None") %>% mutate(first_opt_index=as.integer(first_opt_index))
    dat$opt_time <- dat$annealing_time*(dat$first_opt_index+1)
    
    dat %<>% group_by(annealing_time) %>% mutate(mean_opt_time=mean(opt_time)) %>%
        mutate(median_opt_time=median(opt_time)) %>% ungroup()

    return(dat)
}

##num_relations,num_decimal_pos,annealing_time,query_graph,problem_index,sample_index,first_opt_index


dat <- read.time.to.opt.dwave.rebuttal("../../base/ExperimentalAnalysis/DWave/QPUPerformance/Results/opt_index_results.txt")
dat <- dat %>% mutate(generation = as.character(ifelse(generation == "IntLog", "QPU (Hand-Crafted)", generation)),
	generation = as.character(ifelse(generation == "Steinbrunn", "QPU (Steinbrunn)", generation)),
	generation = as.character(ifelse(generation == "Mancini", "QPU (Mancini et al.)", generation)))
#dat <- read.time.to.opt.dwave.rebuttal("experimental_analysis/DWave/revised_opt_index_results.txt")
#dat.mancini <- read.time.to.opt.dwave.rebuttal("experimental_analysis/DWave/mancini_opt_index_results.txt")

dat.aila <- read.table("mancini.txt", header=TRUE, sep="\t")
colnames(dat.aila) <- c("Method", "Type", "Cores", 2, 3, 4, 5)
##dat.aila$Type <- str_c(dat.aila$Type, " [47]")
dat.aila.long <- pivot_longer(dat.aila, !Method:Cores, names_to="relations", values_to="duration")
dat.aila.long <- dat.aila.long %>% mutate(relations=as.numeric(relations))

#dat$generation <- "QPU (Steinbrunn)"
#dat.prev$generation <- "QPU (Hand-Crafted)"
#dat.mancini$generation <- "QPU (Mancini)"
#dat.all <- rbind(rbind(dat %>% select(colnames(dat.prev)), dat.prev), dat.mancini %>% select(colnames(dat.prev)))

dat$duration <- dat$opt_time*dat$annealing_time/1000 ## Duration in ms until optimal result is reached
dat$Method <- "Quantum Annealing"
dat$relations <- dat$num_relations
dat$Type <- "QPU"
dat$Cores <- 1

g <- ggplot(dat.aila.long, aes(x=relations, y=duration, colour=Type, shape=Method)) + 
    scale_y_log10(labels = label_log()) + 
    annotation_logticks(size=0.1, sides="l", short=unit(0.05, "cm"), mid=unit(0.075, "cm"), long=unit(0.1, "cm")) +
    xlab("\\# Relations") + ylab("Optimisation Time [ms] (log)") +
    geom_point(data=dat %>% filter(annealing_time==0.5), inherit.aes=FALSE,
               aes(x=relations, y=duration, colour=generation, group=interaction(relations, generation)),
               size=0.01, alpha=0.5, shape=20, position=position_jitterdodge(jitter.width=0.15, dodge.width=0.5)) +
    geom_boxplot(data=dat %>% filter(annealing_time==0.5), inherit.aes=FALSE,
                 aes(x=relations, y=duration, colour=generation, group=interaction(relations, generation)),
                 size=0.5, width=0.5, outlier.shape=NA, alpha=0.5,
                 position=position_dodge2(width=0.25, preserve = "total", padding=0.2)) +
    geom_line(size=0.5) + geom_point(size=0.75) + scale_colour_manual("", values=rev(COLOURS.LIST)) +
    theme_paper() + shrink_legend(-1) + theme(legend.key.width=unit(1.5, "mm"), legend.key.height=unit(2.5, "mm"),
                                              legend.text = element_text(size=6), legend.spacing.x=unit(0.6, "mm")) +
    scale_shape_manual("", values=c(2,3,4,5,6,7)) + guides(colour=guide_legend(nrow=2)) +
    guides(shape=guide_legend(override.aes=list(size = SYM.SIZE)))


#tikz("../../plots/fig2.pdf", width=COL.WIDTH*1.06, height=0.65*COL.WIDTH)
#print(g)
#dev.off()

g <- ggplot(dat.aila.long, aes(x=relations, y=duration, colour=Type, shape=Method)) + 
  scale_y_log10(labels = label_log()) + 
  annotation_logticks(size=0.1, sides="l", short=unit(0.05, "cm"), mid=unit(0.075, "cm"), long=unit(0.1, "cm")) +
  xlab("# Relations") + ylab("Optimisation Time [ms] (log)") +
  geom_point(data=dat %>% filter(annealing_time==0.5), inherit.aes=FALSE,
             aes(x=relations, y=duration, colour=generation, group=interaction(relations, generation)),
             size=0.01, alpha=0.5, shape=20, position=position_jitterdodge(jitter.width=0.15, dodge.width=0.5)) +
  geom_boxplot(data=dat %>% filter(annealing_time==0.5), inherit.aes=FALSE,
               aes(x=relations, y=duration, colour=generation, group=interaction(relations, generation)),
               size=0.5, width=0.5, outlier.shape=NA, alpha=0.5,
               position=position_dodge2(width=0.25, preserve = "total", padding=0.2)) +
  geom_line(size=0.5) + geom_point(size=0.75) + scale_colour_manual("", values=rev(COLOURS.LIST)) +
  theme_paper() + shrink_legend(-1) + theme(legend.key.width=unit(1.5, "mm"), legend.key.height=unit(2.5, "mm"),
                                            legend.text = element_text(size=6), legend.spacing.x=unit(0.6, "mm")) +
  scale_shape_manual("", values=c(2,3,4,5,6,7)) + guides(colour=guide_legend(nrow=2)) +
  guides(shape=guide_legend(override.aes=list(size = SYM.SIZE)))

pdf("../../plots/fig2.pdf", width=COL.WIDTH*1.06, height=0.65*COL.WIDTH)
print(g)
dev.off()

#######################################

