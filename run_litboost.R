# Simulate data 
friedman_data = SimulateData(N = 1000, K = 3, sigma = 0.1)
sim_data = friedman_data$sim_data
df = friedman_data$df


# Split into train and test
N = NROW(df)
ind = sample(x = 1:N, size = floor(0.5*N), replace = FALSE)
train_df = df[ind,]
test_df = df[-ind,]

model_formula = formula(paste("y ~", paste(names(df[,-1]), collapse = "+")))

# Run LitBoost 
litboost_model = litboost.train(train_df = df, model_formula = model_formula)
litboost_preds = litboost.predict(model = litboost_model, test_df = test_df, model_formula = model_formula)


# Get shape functions
shape_functions = GetShapeFunctions(model = litboost_model,
                                    formula = model_formula, 
                                    train_df = train_df, 
                                    variables = c("X1", "X2", "X3", "X4", "X5"))

# Visualize shape functions
ggplot(shape_functions, 
       aes(x = x, y = shape_function, col = Group)) + 
  geom_point() + 
  geom_line() + 
  facet_wrap(~variable_name, scales = "free") + 
  xlab(expression(x[i])) + 
  ylab(expression(f[i])) + 
  theme(legend.position = "bottom")
