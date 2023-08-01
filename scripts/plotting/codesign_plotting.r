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

## Relevant:
## Distribution of circuit depths for native gate sets, assuming a fully
## connected topology
dat <- read.data("../../base/TranspilationExperiment/Results/transpilation_results.txt")
dat$gateset[dat$gateset=="Logical"] <- "Unrestricted"
dat$gateset[dat$gateset=="Physical"] <- "Native"
dat[dat$vendor=="IonQ",]$density <- 0

##########################################################
g <- 
    dat %>% filter(((opt_level==3 & optimizer=="Qiskit") | optimizer=="Tket")) %>%
    mutate(dens=(density==0)) %>%
    group_by(vendor, optimizer, gateset, opt_level, density, num_relations) %>% # query_graph_type
    mutate(min_depth=min(depth), max_depth=max(depth), med_depth=median(depth)) %>% ungroup() %>%
    ggplot(aes(x=density, y=med_depth, shape=num_relations, 
           colour=num_relations, group=interaction(num_relations,gateset)), position="dodge") +
    geom_point(position=position_dodge(width=1), size=0.7) + ##geom_line() +
    geom_errorbar(aes(ymin=min_depth, ymax=max_depth), width=.2,
                  position=position_dodge(width=1)) +
    scale_shape_discrete("# Relations") +
    scale_colour_manual("# Relations", values=COLOURS.LIST) +
    facet_nested(optimizer~vendor+gateset, scale="free_y") + #, space="free_x") + # free_y
    scale_y_log10("Circuit Depth [log]") + scale_x_discrete("Extended Connectivity Density") +
    theme_paper() + theme(axis.text.x = element_text(angle=40, size=5, vjust=1, hjust=1), axis.text.y = element_text(size=5)) +
    guides(shape=guide_legend(override.aes=list(size = SYM.SIZE))) + shrink_legend(-5) +
    annotation_logticks(size=0.1, sides="l", short=unit(0.05, "cm"), mid=unit(0.075, "cm"), long=unit(0.1, "cm"))

if (FALSE)
    print(g)

#tikz("../img-tikz/circuit_depth.tex", width=WIDTH, height=0.4*WIDTH-1*INCH.PER.CM)
#pdf("img-tikz/circuit_depth.pdf", width=WIDTH, height=0.4*WIDTH-1*INCH.PER.CM)
pdf("../../plots/fig5.pdf", width=WIDTH, height=0.45*WIDTH-1*INCH.PER.CM)
print(g)
dev.off()

#tikz("../img-tikz/circuit_depth.tex", width=COL.WIDTH, height=0.4*WIDTH)
#print(g)
#dev.off()
