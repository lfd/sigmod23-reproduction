library(ggplot2)
library(ggh4x)
library(ggpmisc)
library(ggrastr)
library(ggpattern)
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


############################## IBMQ experimental measurements ##############################

dat <- read.csv("../../base/ExperimentalAnalysis/IBMQ/Embeddings/Results/depths.txt", header=TRUE)
#dat <- read.csv("experimental_analysis/IBMQ/depths.txt", header=TRUE)

dat.new <- bind_rows(
    dat %>% filter(optimizer=="Qiskit" & opt_level==1 & strategy=="predicates") %>% mutate(experiment="Varying Topology", sepp=topology),
    dat %>% filter(optimizer=="Qiskit" & opt_level==1 & topology=="Auckland") %>% mutate(experiment="Varying Property", sepp=strategy))
dat.new$strategy <- str_to_title(dat.new$strategy)

if (FALSE) {
dat %>% filter(optimizer=="Qiskit" & opt_level==1 & strategy=="Predicates") %>%
    ggplot(aes(x=num_qubits, y=depth, colour=topology, group=interaction(num_qubits, topology)), position="dodge") +
    geom_boxplot() + geom_jitter()

dat %>% filter(optimizer=="Qiskit" & opt_level==1 & topology=="Auckland") %>%
    ggplot(aes(x=num_qubits, y=depth, colour=strategy, group=interaction(num_qubits,strategy)), position="dodge") +
    geom_boxplot() + geom_jitter()
}

g <- 
    ggplot(dat.new, aes(x=num_qubits, y=depth, colour=topology, pattern=strategy,
                        group=interaction(num_qubits,sepp)), position="dodge") +
    geom_boxplot_pattern(pattern_spacing = 0.02,outlier.size=0.5) + facet_grid(~experiment) +
    scale_x_continuous("\\# Qubits", breaks=c(18,21,24,27)) + ylab("Circuit Depth") + theme_paper() +
    scale_colour_manual("System", values=COLOURS.LIST) + scale_pattern_discrete("Property") +
    guides(color=guide_legend(nrow=2)) +
    guides(pattern=guide_legend(nrow=2)) + shrink_legend(-5)

#print(g)

#tikz("../img-tikz/ibmq_experiments.tex", width=COL.WIDTH, height=0.65*COL.WIDTH)
#tikz("img-tikz/ibmq_experiment.tex", width=COL.WIDTH, height=0.55*COL.WIDTH)
pdf("../../plots/fig3.pdf", width=COL.WIDTH, height=0.55*COL.WIDTH)
print(g)
dev.off()
