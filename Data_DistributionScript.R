library(ggplot2)
library(moments)

# Check data distribution
check_distribution <- function(data) {
  numeric_cols <- sapply(data, is.numeric)
  data_numeric <- data[, numeric_cols]
  
  for (col in colnames(data_numeric)) {
    cat("\nColumn:", col, "\n")
    
    # Histogram and Density Plot
    print(ggplot(data, aes_string(x = col)) + 
            geom_histogram(aes(y = ..density..), bins = 30, fill = "blue", alpha = 0.5) +
            geom_density(color = "red", size = 1) +
            ggtitle(paste("Distribution of", col)))
    
    # Skew
    skew <- skewness(data_numeric[[col]], na.rm = TRUE)
    cat("Skew:", skew, "\n")
    
  }
}

# Load data from file
file_path <- "./diabetes.csv"
data <- read.csv(file_path)

# Check distributions
check_distribution(data)
