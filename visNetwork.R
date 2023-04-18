library(reticulate)
library(igraph)
library(visNetwork)


np <- import("numpy")

# data reading
mat <- np$load("network_mutated.npy")
mat<-round(mat,0)
mat<-abs(mat)

## make igraph object
test.gr <- graph_from_adjacency_matrix(mat, mode="directed", weighted=T)

## convert to VisNetwork-list
test.visn <- toVisNetworkData(test.gr)
## copy column "weight" to new column "value" in list "edges"
test.visn$edges$value <- test.visn$edges$weight
#test.visn$edges$weight <- 100

test.visn$edges
#test.visn$edges = subset(test.visn$edges, select = -c(weight,value) )
#test.visn$nodes = subset(test.visn$nodes, select = -c(label) )

write.csv(test.visn$edges, "network_mutated.csv", row.names=FALSE)

visNetwork(test.visn$nodes, test.visn$edges,height = "1000px",width="100%") %>%
  visPhysics(enabled = FALSE)%>%
  visNodes(label=test.visn$nodes$label)%>%
  visEdges(shadow = FALSE,
           arrows = list(to = list(enabled = TRUE, scaleFactor = 1)),
           color = list(color = "lightblue", highlight = "red")) %>%
  visLayout(improvedLayout=TRUE)
  #visHierarchicalLayout(enabled = TRUE)


nodes <- data.frame(id = 1:7)

edges <- data.frame(
  from = c(1,2,2,2,3,3),
  to = c(2,3,4,5,6,7)
)
visNetwork(test.visn$nodes, test.visn$edges, width = "100%") %>% 
  visEdges(arrows = "to") %>% 
  visHierarchicalLayout() 
  #visLayout(randomSeed = 12)

  #visHierarchicalLayout()
  #visHierarchicalLayout()
  
  #visIgraphLayout(layout = "layout_in_circle") 
