library(xgboost)
library(DiagrammeR)
library(mltools)
library(tidyverse)
library(dplyr)




GetDmatrix = function(df, model_formula){
  
  label_variable = all.vars(model_formula)[1]
  
  train_sparse = sparse.model.matrix(model_formula, data = df, sparseMatrixClass='Matrix')
  dtrain = xgb.DMatrix(data = train_sparse, 
                       label = df %>% select(label_variable) %>% unlist() %>% as.numeric())
  
  return(dtrain)
}




litboost.train = function(train_df, model_formula, xc = "xc"){
  
  library(xgboost)
  library(tidyverse)
  
  dtrain = GetDmatrix(df = train_df, model_formula = model_formula)
  
  # Interaction constrains:  Allow all pairs [x_c, x_i], where
  # xc:   categorical feature, e.g. CityDistrict
  # xcj:  level j of the one-hot-encoding of xc, e.g. CityDistrictA or CityDistrictB
  # xi:   every other feature, e.g. Size. 
  
  xcj = which(grepl(xc, colnames(dtrain))) - 1 #minus one because of intercept
  p = min(xcj) - 1
  
  interaction_list = list()
  for(xi in 1:p){ 
    interaction_list = append(interaction_list, list(c(xi,xcj)))
  }
  
  # Make parameter list with interaction constraints
  params = list(eta = 0.01, max_depth = 2, interaction_constraints = interaction_list)
  
  # Train model
  litboost_model = xgb.train(data = dtrain,
                             params = params,
                             nrounds = 1000,
                             tree_method = "hist",
                             verbose = 0)
  
  return(litboost_model)
}

litboost.predict = function(model, test_df, model_formula){
  
  dtest = GetDmatrix(df = test_df, model_formula = model_formula)
  preds = predict(object = model, newdata = dtest)
  
  return(preds)
}




ExtractTreeInfo = function(model){
  
  library(xgboost)
  
  # Extract tree data
  splits_df = xgb.plot.tree(model = model, trees = c(0:model$niter), render = FALSE)
  nodes = splits_df$nodes_df 
  
  # Prepare nodes
  nodes = nodes %>% 
    mutate(iteration = as.numeric(sapply(strsplit(ID,"-"), function(x) x[1])),
           node_number = as.numeric(sapply(strsplit(ID,"-"), function(x) x[2])),
           type = ifelse(data == "Leaf", "Leaf", "Split"), 
           value = as.numeric(sapply(strsplit(label, "\nValue: "), function(x) x[2])),
           gain = as.numeric(sapply(strsplit(label, "\nGain: "), function(x) x[2]))) %>% 
    rename(split_variable = data)
  
  # Prepare edges (that is, spltting_value)
  edges = splits_df$edges_df %>% 
    filter(style == "bold") %>% 
    select(-rel, -id, -style) %>% 
    rename(split_criterion_raw = label) %>% 
    mutate(split_criterion = case_when(split_criterion_raw == "" ~ NA_real_, 
                                       TRUE ~ round(as.numeric(substr(x = split_criterion_raw, start = 3, stop = 1000)),2))) %>% 
    left_join(splits_df$edges_df %>% filter(style == "solid") %>% select("from", "to"), 
              by = "from") %>%
    rename(id = from, 
           node_if_true = to.x, 
           node_if_false = to.y) %>%
    relocate(c("id","split_criterion_raw","split_criterion", "node_if_true", "node_if_false"))
  
  # Merge edges with nodes
  nodes = left_join(x = nodes, 
                    y = edges,
                    by = c("id" = "id")) 
  
  # Remove some columns
  nodes = nodes %>% 
    select(-fillcolor, -shape, -fontcolor, -ID) %>% 
    select(-label)
  
  # Clean up
  rownames(nodes) <- NULL
  nodes = nodes %>%
    relocate(c("id", "iteration", "node_number", "split_variable", "split_criterion", "split_criterion_raw", "node_if_true", "node_if_false", "value", "gain")) %>% 
    arrange(iteration, node_number)
  
  # Get parent ID
  mvp_df = nodes %>% 
    select(id, iteration, type, value, node_number) 
  
  child_nodes = nodes %>% 
    pivot_longer(cols = c("node_if_true", "node_if_false"), 
                 values_to = "child", 
                 names_to = "crit") %>% 
    select(id, child, split_variable) %>% 
    rename(id_parent = id, 
           split_variable_parent = split_variable)
  
  final_df = mvp_df %>% left_join(child_nodes, by = c("id" = "child"))
  
  #return(nodes)
  return(final_df)
}


GetSplitVariablesForEveryLeafID = function(clean_model_df){
  
  leaf_ids = clean_model_df$id[which(clean_model_df$type == "Leaf")]
  
  # This is the whole function now: 
  split_list = list()
  split_list$ids = leaf_ids
  split_list$splits = lapply(leaf_ids, FUN = GetSplitVariables, tree_df = clean_model_df)
  
  return(split_list)
  
}


GetSplitVariables = function(tree_df, id_to_trace){
  
  split_variable_parent = tree_df$split_variable_parent[which(tree_df$id == id_to_trace)]
  id_to_trace = tree_df$id_parent[which(tree_df$id == id_to_trace)]
  
  split_list = c()
  while(!is.na(split_variable_parent)){
    split_list = c(split_list, split_variable_parent)
    
    split_variable_parent = tree_df$split_variable_parent[which(tree_df$id == id_to_trace)]
    id_to_trace = tree_df$id_parent[which(tree_df$id == id_to_trace)]
    
  }
  return(split_list)
}


GetNewTestInstance = function(df,
                               var_name,
                               var_grid){
  
  # This will be handy 
  N = NROW(df)
  p = NCOL(df)
  all_columns = colnames(df)
  
  N_grid = length(var_grid)
  
  if(N_grid > N){
    # This is very hacky
    # We need at N_grid rows in the test_df; if N_grid > N we must do some tricks
    new_df = do.call("rbind", replicate(ceiling(N_grid/N), df, simplify = FALSE))
    new_df = new_df[1:N_grid,]
    
  } else {
    new_df = df[1:N_grid,]
  }
  
  # Groups
  # unique_groups = unique(df$xc)
  # K = length(unique_groups)
  K = NCOL(df %>% select(starts_with("xc")))

  # Prepare a new data frame by making K copies of test_df
  new_df = do.call("rbind", replicate(K, new_df, simplify = FALSE))
  
  # Insert the grid here at the variable of interest
  new_df[,var_name] = rep(var_grid, K)
  
  # Copy copy structure of xc N times

  xc_cols = as.numeric(which(grepl("xc", colnames(new_df))))
  multiple_blocks = do.call("rbind", replicate(N_grid, diag(x = 1, nrow = K), simplify = FALSE)) %>%
    as.data.frame()
  
  new_df[,xc_cols] = multiple_blocks
  return(new_df)
}


GetShapeFunctionValue = function(model, 
                                 tree_info, 
                                 var_name = "X1", 
                                 var_grid = seq(from = 0, to = 1, length.out = 100),
                                 dtest, 
                                 split_list){
  
  tree_ids = unique(tree_info$iteration[which(tree_info$split_variable_parent == var_name)])
  
  # Get the nodes in each tree that the new instance will land in 
  predict_leaf = predict(model, newdata = dtest, predleaf = T)
  
  N = NROW(predict_leaf)
  its = NCOL(predict_leaf)
  
  # Prepare an array to store the shape function values
  shape_values = rep(0,N)
  
  for(nn in 1:N){
    # Extract only the nodes of relevance
    relevant_model_df = data.frame(node_number = predict_leaf[nn,], 
                                   iteration = 0:(its-1)) %>%
      left_join(tree_info, by = c("node_number", "iteration"))
    
    # Get the node IDS of the leafs that depend on var_name
    relevant_part_of_list = grepl(var_name, split_list$splits)
    relevant_ids = split_list$ids[relevant_part_of_list]
    
    # Find the intersection between this and the relevant_model_df
    contains_var = intersect(relevant_ids, relevant_model_df$id)
    
    shape_values[nn] = sum(tree_info$value[which(tree_info$id %in% contains_var)])
  }
  
  return(shape_values)

}


GetShapeFunctions = function(model, 
                             formula, 
                             train_df, 
                             variables = c("X1", "X2", "X3"),
                             var_grid){
  
  all_shape_functions = data.frame()
  
  for(v in variables){
    this_shape_function = GetOneShapeFunction(model = model, 
                                              formula = formula, 
                                              train_df = train_df, 
                                              var_name = v)
    
    all_shape_functions = rbind(all_shape_functions, this_shape_function)
  }
  
  return(all_shape_functions)
}


GetOneShapeFunction = function(model, 
                             formula,
                             train_df,
                             var_name = "X1", 
                             var_grid = seq(from = 0, to = 1, length.out = 100)){
  
  
  # Extract tree info
  tree_info = ExtractTreeInfo(model)
  split_list = GetSplitVariablesForEveryLeafID(tree_info)
  

  # Prepare 
  shape_df = data.frame()
  N = NROW(train_df)
  
  
  # Create a synthetic data set where the variable at hand has an equidistant grid
  new_df = GetNewTestInstance(df = train_df, 
                               var_name = var_name, 
                               var_grid = var_grid) 
  
  dtest = GetDmatrix(df = new_df, model_formula = formula)
  
  new_df$shape_function = GetShapeFunctionValue(model = model, 
                                            tree_info = tree_info, 
                                            var_name = var_name, 
                                            var_grid = var_grid, 
                                            dtest = dtest, 
                                            split_list = split_list)
  
  shape_df = new_df %>% 
    #Pivoting from one-hot to factor with K levels to prepare for ggplot2
    pivot_longer(cols = starts_with("xc"), names_to = "group") %>% 
    filter(value == 1) %>% 
    select(shape_function, var_name, group) %>%
    rename(xc = group, 
           x = var_name) %>% 
    mutate(variable_name = var_name, 
           Group = str_replace(xc, "xc_", "Group ")) %>%
    select(-xc)
  
  return(shape_df)
}




FriedmanArticifial = function(x, 
                              seed = 123, 
                              sigma = 0.1, 
                              a = 10, 
                              b = 20, 
                              c = 10 , 
                              d = 5){
  
  f1 = a*sin(pi*x[,1]*x[,2])
  f2 = b*(x[,3]-0.5)^2
  f3 = c*x[,4]
  f4 = d*x[,5]
  y = f1 + f2 + f3 + f4
  noise = rnorm(length(y), mean = 0, sd = sigma)
  
  new_df = cbind(as.data.frame(x), f1, f2, f3, f4, noise, y, a, b, c, d)
  
  return(new_df)
}




SimulateData = function(N = 1000, K = 9, sigma = 0.1, one_hot_encoded = TRUE){
  
  p = 10
  x = matrix(data = runif(n = N*p), 
             nrow = N, 
             ncol = p)
  
  a = c(rnorm(K/3, 5, sigma), 
        rnorm(K/3, 10, sigma), 
        rnorm(K/3, 15, sigma))
  
  b = c(rnorm(K/3, 15, sigma), 
        rnorm(K/3, 20, sigma), 
        rnorm(K/3, 25, sigma))
  
  c = c(rnorm(K/3, 5, sigma), 
        rnorm(K/3, 10, sigma), 
        rnorm(K/3, 15, sigma))
  
  
  d = c(rnorm(K/3, 3, sigma), 
        rnorm(K/3, 5, sigma), 
        rnorm(K/3, 7, sigma))
  
  
  full_df = data.frame()
  
  for(xc in 1:K){
    x = matrix(data = runif(n = N*p), 
               nrow = N, 
               ncol = p) 
    colnames(x) = paste("X", 1:p, sep = "")
    this_y = FriedmanArticifial(x, a = a[xc], b = b[xc], c = c[xc], d = d[xc])
    full_df = rbind(full_df, cbind(this_y, xc))
  }
  
  full_df = full_df %>% 
    mutate(xc = as.factor(xc)) 

  full_df_one_hot = mltools::one_hot(as.data.table(full_df)) 
  
  
  sim_data = full_df_one_hot %>% select(starts_with("f"), noise, a, b, c, d)
  df = full_df_one_hot %>% select(y, starts_with("X"), starts_with("xc"))
  
  
  return(list(df = df, sim_data = sim_data))
}


