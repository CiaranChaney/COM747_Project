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
    
    # Scatter Plot
    print(ggplot(data, aes(x = seq_along(.data[[col]]), y = .data[[col]])) +
            geom_point(color = "blue", alpha = 0.5) +
            ggtitle(paste("Scatter Plot of", col)) +
            xlab("Index") +
            ylab(col))
    
    # Skew
    skew <- skewness(data_numeric[[col]], na.rm = TRUE)
    cat("Skew:", skew, "\n")
    
    # Determine if the data is normal or skewed
    if (abs(skew) < 0.5) {
      cat("The data appears to be normally distributed.\n")
    } else if (skew > 0.5) {
      cat("The data is positively skewed (right-skewed).\n")
    } else {
      cat("The data is negatively skewed (left-skewed).\n")
    }
    
  }
}

# Load data from file
file_path <- "./diabetes.csv"
data <- read.csv(file_path)

# Check distributions
check_distribution(data)