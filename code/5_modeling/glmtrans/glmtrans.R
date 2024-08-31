library(glmtrans)


x_data <- read.csv("data/processed/xy/x.csv")
y_data <- read.csv("data/processed/xy/y.csv")
x_data <- x_data[which(y_data$adm1 == "Central"), ]
y_data <- y_data[which(y_data$adm1 == "Central"), ]


# Unique admins
unique_adms <- unique(y_data$adm1)

# Initialize lists
adm_ls <- list()
x_ls <- list()
y_ls <- list()
year_ls <- list()

# Split the data
for (adm in unique_adms) {
  # Get indices for current adm
  adm_indices <- which(y_data$adm1 == adm)
  
  # Append the subsets to the respective lists
  adm_ls[[adm]] <- adm
  x_ls[[adm]] <- x_data[adm_indices, ]
  y_ls[[adm]] <- y_data[adm_indices, "yield_anomaly"]
  year_ls[[adm]] <- y_data[adm_indices, "harv_year"]
}


# Initialize a list to store results
results <- list()

# Outer loop over unique admins
for (adm in names(adm_ls)) {
  # Get the data for the current admin
  x_current <- x_ls[[adm]]
  y_current <- y_ls[[adm]]
  years_current <- year_ls[[adm]]
  
  # Initialize vectors to accumulate true and predicted values
  y_true_all <- c()
  y_pred_all <- c()
  
  # Inner loop over unique years
  unique_years <- unique(years_current)
  
  for (year in unique_years) {
    # Split data into training and test sets for the current year
    test_indices <- which(years_current == year)
    train_indices <- setdiff(1:length(years_current), test_indices)
    
    x_train <- x_current[train_indices, ]
    y_train <- y_current[train_indices]
    x_test <- x_current[test_indices, ]
    y_test <- y_current[test_indices]
    
    # Create multi-source data
    D.training <- list(target = list(x = x_train, y = y_train))
    D.training$source <- list()
    
    for (source_adm in setdiff(names(adm_ls), adm)) {
      source_train_indices <- which(year_ls[[source_adm]] != year)
      D.training$source[[source_adm]] <- list(
        x = x_ls[[source_adm]][source_train_indices, ],
        y = y_ls[[source_adm]][source_train_indices]
      )
    }
    
    D.test <- list(target = list(x = x_test))
    
    # Fit the multi-source glmtrans model
    fit.gaussian <- glmtrans(D.training$target, D.training$source, cores=6, nfolds=4)
    y_pred_glmtrans <- predict(fit.gaussian, D.test$target$x)
    
    # Accumulate the true and predicted values
    y_true_all <- c(y_true_all, y_test)
    y_pred_all <- c(y_pred_all, y_pred_glmtrans)
  }
  
  # Calculate R2 for the current admin
  ss_res <- sum((y_true_all - y_pred_all)^2)
  ss_tot <- sum((y_true_all - mean(y_true_all))^2)
  r2 <- 1 - (ss_res / ss_tot)
  
  # Store the results for the current admin
  results[[adm]] <- list(
    y_true = y_true_all,
    y_pred = y_pred_all,
    R2 = r2
  )
  print(paste(adm, r2))
}

# Access the results
print(results)


 

