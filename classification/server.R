library(shiny)
library(caret)
library(dplyr)
library(plotly)
library(Rtsne)

# Define server logic
server <- function(input, output, session) {
  
  data <- reactive({
    if (input$dataSource == "upload") {
      req(input$file1)
      # Read the CSV file with semicolon delimiter
      read.csv(input$file1$datapath, sep = ";")
    } else {
      # Generate random data
      set.seed(Sys.time())
      df <- iris[sample(nrow(iris), 30), ]
      df
    }
  })
  
  output$dataSummary <- renderTable({
    head(data())
  })
  
  observeEvent(input$trainModel, {
    df <- data()
    
    # Preprocess data: handle NAs by replacing with column mean
    df <- df %>% 
      mutate(across(where(is.numeric), ~ifelse(is.na(.), mean(., na.rm = TRUE), .)))
    
    # Ensure target variable is a factor
    df$Species <- as.factor(df$Species)
    
    # Split data into training and testing sets
    set.seed(123)
    trainIndex <- createDataPartition(df$Species, p = 0.8, list = FALSE)
    trainData <- df[trainIndex, ]
    testData <- df[-trainIndex, ]
    
    if (input$model == 'KNN') {
      k <- input$k_value
      # Train the KNN model with user-defined k
      model <- train(Species ~ Petal.Length + Petal.Width, data = trainData, method = 'knn', tuneGrid = data.frame(k = k))
      
      # Generate decision boundary plot
      plot_data <- expand.grid(
        Petal.Length = seq(min(df$Petal.Length), max(df$Petal.Length), length.out = 100),
        Petal.Width = seq(min(df$Petal.Width), max(df$Petal.Width), length.out = 100)
      )
      plot_data$Prediction <- predict(model, newdata = plot_data)
      
      output$plot <- renderPlotly({
        plot_ly() %>%
          add_markers(data = df, x = ~Petal.Length, y = ~Petal.Width, color = ~Species, colors = c("#eb3527", "#387e22", "#f3aa3c"), marker = list(size = 8)) %>%
          add_trace(
            data = plot_data,
            x = ~Petal.Length,
            y = ~Petal.Width,
            z = ~as.numeric(factor(Prediction)),
            type = 'heatmap', showscale = FALSE,
            colorscale = list(c(0, '#f9e0e0'), c(0.5, '#e0edde'), c(1, '#fef6e5')),
            opacity = 1) %>%
          layout(title = paste("KNN Decision Boundary (k =", k, ")"),
                 xaxis = list(title = "Petal Length"),
                 yaxis = list(title = "Petal Width"),
                 plot_bgcolor = "rgba(240, 240, 240, 0.95)")
      })
    } else if (input$model == 'Logistic Regression') {
      # Train the Logistic Regression model
      model <- train(Species ~ Petal.Length + Petal.Width, data = trainData, method = 'multinom')
      
      # Generate decision boundary plot
      plot_data <- expand.grid(
        Petal.Length = seq(min(df$Petal.Length), max(df$Petal.Length), length.out = 100),
        Petal.Width = seq(min(df$Petal.Width), max(df$Petal.Width), length.out = 100)
      )
      plot_data$Prediction <- predict(model, newdata = plot_data)
      
      output$plot <- renderPlotly({
        plot_ly() %>%
          add_markers(data = df, x = ~Petal.Length, y = ~Petal.Width, color = ~Species, colors = c("#eb3527", "#387e22", "#f3aa3c"), marker = list(size = 8)) %>%
          add_trace(
            data = plot_data,
            x = ~Petal.Length,
            y = ~Petal.Width,
            z = ~as.numeric(factor(Prediction)),
            type = 'heatmap',
            showscale = FALSE,
            colorscale = list(c(0, '#f9e0e0'), c(0.5, '#e0edde'), c(1, '#fef6e5')),
            opacity = 1
          ) %>%
          layout(title = "Logistic Regression Decision Boundary",
                 xaxis = list(title = "Petal Length"),
                 yaxis = list(title = "Petal Width"),
                 plot_bgcolor = "rgba(240, 240, 240, 0.95)")
      })
    }
  })
}
