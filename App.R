library(shiny)
library(DT)
library(ggplot2)
library(dplyr)
library(plotly)
library(GGally)
library(psych)
library(nnet)
library(dplyr)
library(caret)
library(tsne)

generate_data <- function(n = 200, dimensions = 2, num_classes = 3) {
  set.seed(123)
  centers <- matrix(runif(num_classes * dimensions, -2, 2), ncol = dimensions)
  data <- NULL
  labels <- factor()
  
  # Assign a random number of samples to each class
  # Ensuring that total samples sum up to n
  samples_per_class <- sample(1:n, num_classes, replace = TRUE)
  samples_per_class <- round(n * samples_per_class / sum(samples_per_class))
  if (sum(samples_per_class) != n) {
    samples_per_class[1] <- samples_per_class[1] + n - sum(samples_per_class)
  }
  
  for (i in 1:num_classes) {
    # Generate data for the current class
    current_data <- matrix(rnorm(samples_per_class[i] * dimensions, mean = rep(centers[i,], each = samples_per_class[i])), 
                           ncol = dimensions, byrow = TRUE)
    data <- rbind(data, current_data)
    labels <- c(labels, factor(rep(i, samples_per_class[i])))
  }
  
  data <- as.data.frame(data)
  data$y <- labels
  return(data)
}

# Function to perform t-SNE using the tsne package
perform_tsne <- function(data, dims = 2) {
  tsne_out <- tsne(data[, -ncol(data)], k = dims, perplexity = 30, max_iter = 500)
  transformed <- data.frame(tsne_out)
  colnames(transformed) <- c("X1", "X2", "X3")[1:dims]
  transformed$y <- data$y
  return(transformed)
}

# Save plot using plotly for 3D visualization
save_plotly_plot <- function(data, title) {
  plot <- plot_ly(data, x = ~X1, y = ~X2, z = ~X3, color = ~y, type = 'scatter3d', mode = 'markers') %>%
    layout(title = title, scene = list(xaxis = list(title = 't-SNE 1'),
                                       yaxis = list(title = 't-SNE 2'),
                                       zaxis = list(title = 't-SNE 3')))
  return(plot)
}
ui <- navbarPage(
  "Machine Learning Pipeline",
  tabPanel(
    "About",
    fluidPage(
      titlePanel(p("About the Project", style = "color:#3474A7")),
      h4("High-Level Goal"),
      p("Our objective is to make machine learning more accessible and inclusive. By creating an intuitive simulation of a machine learning pipeline, we aim to enhance the learning experience, allowing users to easily grasp complex concepts and apply machine learning techniques in a practical, hands-on manner."),
      br(), br(),
      h4("Project Description"),
      p("This project offers a user-friendly simulation of a machine learning pipeline using R Shiny. Through interactive modules, users can upload datasets, preprocess data, select and train machine learning algorithms, and evaluate model performance. Our platform provides an engaging, hands-on learning experience, making machine learning concepts accessible to a broad audience."),
      br(), br(),
      h4("Steps to complete and evaluate a machine learning model"),
      tags$ol(
        tags$li("Upload Data: Choose the data you want to use"),
        tags$li("Data Summary: Give an overview by providing a data summary"),
        tags$li("Data Exploration: Preprocess data by deleting/filling NAs and creating simple plots"),
        tags$li("Build Model: Pick variables, split train/test set, choose algorithms"),
        tags$li("Evaluation: Evaluate the model performance")
      ),
      br(), br()
    )
  ),
  tabPanel(
    "1. Upload",
    fluidPage(
      titlePanel(p("Upload Your Dataset")),
      sidebarLayout(
        sidebarPanel(
          fileInput(
            inputId = "upload",
            label = "Upload data (CSV format recommended)",
            accept = ".csv"
          ),
          br(),
          uiOutput("target")
        ),
        mainPanel(
          h4("Data Uploaded:"),
          tableOutput("files"),
          br(), br(),
          h4("Data Preview:"),
          DTOutput("head")
        )
      )
    )
  ),
  tabPanel(
    "2. Data Summary",
    titlePanel(p("Data Summary")),
    sidebarLayout(
      sidebarPanel(
        p("This section provides a summary of the uploaded data."),
        selectInput(inputId = "variable", label = "Select variable for violin plot:", choices = NULL)
      ),
      mainPanel(
        h4("Summary of Data"),
        tableOutput("dataSummary"),
        br(), br(),
        h4("Violin Plot"),
        plotlyOutput("violinPlot")
      )
    )
  ),
  tabPanel(
    "3. Data Exploration",
    fluidPage(
      titlePanel(p("Data Preprocessing")),
      sidebarLayout(
        sidebarPanel(
          h4("Data Preprocessing"),
          selectInput("na_action", "Choose NA handling method:",
                      choices = c("None", "Delete all NAs", "Fill NAs with mean")
          ),
          actionButton("apply_preprocess", "Apply"),
          br(), br(),
          h4("Data Plotting"),
          selectInput("singlePlotGeom", "Select plot type",
                      choices = c("point", "boxplot", "histogram", "density"),
                      selected = "histogram"
          ),
          uiOutput("expXaxisVarSelector"),
          uiOutput("expYaxisVarSelector"),
          uiOutput("expColorVarSelector"),
          br(), br(),
          h4("3D Scatter Plot Selection"), # New section for 3D plot selection
          uiOutput("exp3DXaxisVarSelector"),
          uiOutput("exp3DYaxisVarSelector"),
          uiOutput("exp3DZaxisVarSelector"),
          uiOutput("exp3DColorVarSelector")
        ),
        mainPanel(
          h4("One and Two Variable Plot"),
          plotlyOutput("expSinglePlot"),
          h4("Pairs Plot (only non-zero variance variables shown)"),
          plotlyOutput("expPairsPlot", width = "100%", height = "800px"),
          h4("3D Scatter Plot"), # New 3D scatter plot
          plotlyOutput("exp3DPlot")
        )
      )
    )
  ),
  tabPanel(
    "4. Model Building",
    fluidPage(
      titlePanel(p("Model Building")),
      sidebarLayout(
        sidebarPanel(
          h4("Model Selection"),
          selectInput(
            inputId = "model_type",
            label = "Model Type",
            choices = c("Classification", "Regression")
          ),
          conditionalPanel(
            condition = "input.model_type == 'Classification'",
            selectInput(
              inputId = "model",
              label = "Classification Algorithm",
              choices = c("KNN", "Logistic Regression", "Neural Network", "K-means Clustering", "SVM")
            ),
            conditionalPanel(
              condition = "input.model == 'KNN'",
              numericInput(
                inputId = "k_value",
                label = "Number of Neighbors (k)",
                value = 5,  # Default value for k
                min = 1,    
                max = 50
              )
            ),
            conditionalPanel(
              condition = "input.model == 'Logistic Regression'",
              sliderInput("timeStep", label = "Select Time Step:", min = 1, max = 100, value = 1)
            )
          ),
          conditionalPanel(
            condition = "input.model == 'Neural Network'",
            sliderInput("size", "Number of Hidden Units", min = 1, max = 20, value = 5, step = 1),
            sliderInput("decay", "Weight Decay", min = 0, max = 1, value = 0.1, step = 0.05),
            sliderInput("maxit", "Maximum Iterations", min = 100, max = 1000, value = 500, step = 100)
          ),
          conditionalPanel(
            condition = "input.model == 'K-means Clustering'",
            numericInput("clusters", "Number of Clusters", min = 2, max = 10, value = 3)
          ),
          actionButton("trainModel", "Train Model")
        ),
        mainPanel(
          tableOutput("dataViewer"),
          plotlyOutput("plot")
        )
      )
    )
  ),
  tabPanel(
    "5. High Dimensional Data",
    fluidPage(
      titlePanel(p("High Dimensional Data")),
      sidebarLayout(
        sidebarPanel(
          h4("Model Selection"),
          selectInput(
            inputId = "md",
            label = "Model Type",
            choices = c("Neural Network", "SVM", "K-means")
          ),
          conditionalPanel(
            condition = "input.md == 'K-means'",
            numericInput("classes", "Number of Classes", min = 2, max = 5, value = 3, step = 1),
            numericInput("dimensions", "Number of Dimensions", min = 2, max = 50, value = 2, step = 1),
            numericInput("clusters", "Number of Clusters", min = 2, max = 10, value = 3)
          ),
          conditionalPanel(
            condition = "input.md == 'Neural Network'",
            numericInput("classes", "Number of Classes", min = 2, max = 5, value = 3, step = 1),
            numericInput("dimensions", "Number of Dimensions", min = 2, max = 50, value = 2, step = 1),
            sliderInput("size", "Number of Hidden Units", min = 1, max = 20, value = 5, step = 1),
            sliderInput("decay", "Weight Decay", min = 0, max = 1, value = 0.1, step = 0.05),
            sliderInput("maxit", "Maximum Iterations", min = 100, max = 1000, value = 500, step = 100),
          ),
          conditionalPanel(
            condition = "input.md == 'SVM'",
            numericInput("classes", "Number of Classes", min = 2, max = 5, value = 3, step = 1),
            numericInput("dimensions", "Number of Dimensions", min = 2, max = 50, value = 2, step = 1),
            numericInput("clusters", "Number of Clusters", min = 2, max = 10, value = 3)
          ),
          actionButton("tsne", "Visualize")
        ),
        mainPanel(
          plotlyOutput("nnPlot")
        )
      )
    )
  )
)

server <- function(input, output, session) {
  data <- reactiveVal(NULL)
  processed_data <- reactiveVal(NULL)
  
  observeEvent(input$upload, {
    req(input$upload)
    tryCatch({
      df <- read.csv(input$upload$datapath, sep = ";")
      data(df)
      processed_data(df)
      
      output$files <- renderTable({
        data.frame(Name = input$upload$name, Size = input$upload$size)
      })
      
      output$head <- DT::renderDT({
        datatable(data(), filter = "top", editable = TRUE, rownames = FALSE, options = list(
          dom = "lrtip",
          lengthMenu = list(c(10, 50, 100, -1), c(10, 50, 100, "All"))
        ))
      })
      
      updateSelectInput(session, "variable", choices = names(df))
      updateSelectInput(session, "expXaxisVarSelector", choices = names(df))
      updateSelectInput(session, "expYaxisVarSelector", choices = c("None", names(df)), selected = "None")
      updateSelectInput(session, "expColorVarSelector", choices = c("None", names(df)))
      updateSelectInput(session, "exp3DXaxisVarSelector", choices = names(df)) # Update for 3D X-axis variable selector
      updateSelectInput(session, "exp3DYaxisVarSelector", choices = names(df)) # Update for 3D Y-axis variable selector
      updateSelectInput(session, "exp3DZaxisVarSelector", choices = names(df)) # Update for 3D Z-axis variable selector
      updateSelectInput(session, "exp3DColorVarSelector", choices = c("None", names(df))) # Update for 3D Color variable selector
    }, error = function(e) {
      # Handle errors during data upload (e.g., invalid file format)
      showModal(modalOption(title = "Error Uploading Data",
                            dialogBody = tags$p("An error occurred while uploading your data. Please check the file format and try again.")))
      return(NULL)
    })
  })
  
  output$dataSummary <- renderTable(
    {
      df <- data()
      if (is.null(df) | length(df) == 0) {
        return(NULL)
      }
      
      # Check for data type issues (e.g., convert factors to numeric if necessary)
      for (var in names(df)) {
        if (is.factor(df[[var]])) {
          df[[var]] <- as.numeric(as.character(df[[var]]))  # Convert factor to numeric (example conversion)
        }
      }
      
      summary_df <- describe(df[, !(names(df) %in% c(input$variable))])
      summary_df
    },
    rownames = TRUE
  )
  
  output$violinPlot <- renderPlotly({
    df <- data()
    var <- input$variable
    if (is.null(df) || is.null(var)) {
      return(NULL)
    }
    
    p <- ggplot(df, aes_string(x = var, y = var)) +
      geom_violin() +
      labs(title = paste("Violin Plot of", var), x = var, y = var) +
      theme_minimal()
    
    ggplotly(p)
  })
  
  observeEvent(input$apply_preprocess, {
    df <- data()
    if (is.null(df)) {
      return(NULL)
    }
    
    if (input$na_action == "Delete all NAs") {
      df <- na.omit(df)
    } else if (input$na_action == "Fill NAs with mean") {
      df <- df %>% mutate(across(everything(), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))
    }
    processed_data(df)
  })
  
  output$expXaxisVarSelector <- renderUI({
    df <- processed_data()
    if (is.null(df)) {
      return(NULL)
    }
    selectInput("expXaxisVar", "X-axis variable", choices = names(df))
  })
  
  output$expYaxisVarSelector <- renderUI({
    df <- processed_data()
    if (is.null(df)) {
      return(NULL)
    }
    selectInput("expYaxisVar", "Y-axis variable", choices = c("None", names(df)))
  })
  
  output$expColorVarSelector <- renderUI({
    df <- processed_data()
    if (is.null(df)) {
      return(NULL)
    }
    selectInput("expColorVar", "Color variable", choices = c("None", names(df)))
  })
  
  output$exp3DXaxisVarSelector <- renderUI({
    df <- processed_data()
    if (is.null(df)) {
      return(NULL)
    }
    selectInput("exp3DXaxisVar", "3D X-axis variable", choices = names(df))
  })
  
  output$exp3DYaxisVarSelector <- renderUI({
    df <- processed_data()
    if (is.null(df)) {
      return(NULL)
    }
    selectInput("exp3DYaxisVar", "3D Y-axis variable", choices = names(df))
  })
  
  output$exp3DZaxisVarSelector <- renderUI({
    df <- processed_data()
    if (is.null(df)) {
      return(NULL)
    }
    selectInput("exp3DZaxisVar", "3D Z-axis variable", choices = names(df))
  })
  
  output$exp3DColorVarSelector <- renderUI({
    df <- processed_data()
    if (is.null(df)) {
      return(NULL)
    }
    selectInput("exp3DColorVar", "3D Color variable", choices = c("None", names(df)))
  })
  
  output$expSinglePlot <- renderPlotly({
    df <- processed_data()
    if (is.null(df)) {
      return(NULL)
    }
    
    x_var <- input$expXaxisVar
    y_var <- input$expYaxisVar
    color_var <- input$expColorVar
    
    p <- ggplot(df, aes_string(x = x_var))
    
    if (input$singlePlotGeom %in% c("histogram", "density")) {
      geom <- switch(input$singlePlotGeom,
                     "histogram" = geom_histogram(),
                     "density" = geom_density()
      )
      
      p <- p + geom
    } else {
      if (y_var != "None") {
        p <- p + aes_string(y = y_var)
      }
      
      if (color_var != "None") {
        p <- p + aes_string(color = color_var)
      }
      
      geom <- switch(input$singlePlotGeom,
                     "point" = geom_point(),
                     "boxplot" = geom_boxplot()
      )
      
      p <- p + geom
    }
    
    ggplotly(p + labs(title = paste(input$singlePlotGeom, "of", x_var, if (y_var != "None") paste("and", y_var))) +
               theme_minimal())
  })
  
  output$expPairsPlot <- renderPlotly({
    df <- processed_data()
    if (is.null(df)) {
      return(NULL)
    }
    df <- df %>% select_if(~ var(.) != 0)  # Filter out variables with zero variance
    
    p <- ggpairs(df) + theme_minimal()
    ggplotly(p)
  })
  
  output$exp3DPlot <- renderPlotly({
    df <- processed_data()
    if (is.null(df)) {
      return(NULL)
    }
    
    x_var <- input$exp3DXaxisVar
    y_var <- input$exp3DYaxisVar
    z_var <- input$exp3DZaxisVar
    color_var <- input$exp3DColorVar
    
    if (is.null(x_var) || is.null(y_var) || is.null(z_var) || x_var == "" || y_var == "" || z_var == "") {
      return(NULL)
    }
    
    # Check data types for 3D plot (should be numeric)
    if (!all(sapply(df[c(x_var, y_var, z_var)], is.numeric))) {
      showModal(modalOption(title = "Error in 3D Plot",
                            dialogBody = tags$p("Selected variables for 3D plot must be numeric. Please check your data types.")))
      return(NULL)
    }
    
    plot_ly(df,
            x = ~ get(x_var), y = ~ get(y_var), z = ~ get(z_var), color = ~ get(color_var), colorscale = "Viridis",
            type = "scatter3d", mode = "markers"
    ) %>%
      layout(scene = list(
        xaxis = list(title = x_var),
        yaxis = list(title = y_var),
        zaxis = list(title = z_var)
      ))
  })
  
  observeEvent(input$trainModel, {  # Triggered when training button is clicked
    df <- processed_data()
    
    if (input$model_type == "Classification") {
      # Classification model training and evaluation
      set.seed(Sys.time())
      
      trainIndex <- createDataPartition(df$Species, p = 0.8, list = FALSE)
      trainData <- df[trainIndex, ]
      testData <- df[-trainIndex, ]
      
      output$dataViewer <- renderTable({
        head(data())
      })
      
      if (input$model == "KNN") {
        k <- as.numeric(input$k_value)
        
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
      } else if (input$model == "Neural Network") {
        # Train the Logistic Regression model
        timeStep <- input$timeStep
        
        model <- multinom(Species ~ Petal.Length + Petal.Width, data = trainData, maxit = timeStep)
        
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
    } else if (input$model == "SVM") {
        # SVM
        model <- train(Species ~ Petal.Length + Petal.Width, data = trainData, method = "svmRadial")
        
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
            layout(title = "SVM Decision Boundary",
                   xaxis = list(title = "Petal Length"),
                   yaxis = list(title = "Petal Width"),
                   plot_bgcolor = "rgba(240, 240, 240, 0.95)")
        })
      } else if (input$model == "K-means Clustering") {
        
        # Set up and run the K-means model
        clusters <- input$clusters  # user-defined number of clusters
        kmeans_model <- kmeans(trainData, centers = clusters, nstart = 25)
        
        
        # Update UI to reflect clustering, here using a simple plot or other appropriate output
        output$plot <- renderPlotly({
          plot_data <- df[, c(input$exp3DXaxisVar, input$exp3DYaxisVar)]
          plot_data <- plot_data[complete.cases(plot_data), ]  # Remove rows with missing values
          
          plot_ly(plot_data, x = ~get(input$exp3DXaxisVar), y = ~get(input$exp3DYaxisVar), color = ~df$Cluster, type = 'scatter', mode = 'markers') %>%
            layout(title = "K-means Clustering Results")
        })
      }
    } else if (input$model == "Linear Regression") {
      # Linear Regression model
      model <- lm(formula = paste(names(df)[ncol(df)], "~."), data = trainData)
      
      # Generate scatter plot with regression line
      output$plot <- renderPlotly({
        plot_data <- expand.grid(trainData[, -ncol(trainData)])
        plot_data$Prediction <- predict(model, newdata = plot_data)
        
        plot_ly(trainData, x = ~Petal.Length, y = ~Sepal.Length) %>%
          add_markers() %>%
          add_lines(data = plot_data, x = ~Petal.Length, y = ~Prediction, color = "red", line = list(width = 2)) %>%
          layout(title = "Linear Regression",
                 xaxis = list(title = "Petal Length"),
                 yaxis = list(title = "Sepal Length"),
                 plot_bgcolor = "rgba(240, 240, 240, 0.95)")
      })
    }
  })
  
  observeEvent(input$tsne, {  # Triggered when training button is clicked
    data <- reactive({
      generate_data(200, input$dimensions, input$classes)
    })
    
    trained_model <- reactiveVal(NULL)
    tsne_data <- reactiveVal(NULL)
    

    generated_data <- data()
    # Apply appropriate model based on selected method
    if (input$md == "Neural Network") {
        model <- multinom(y ~ ., data = generated_data, decay = input$decay, maxit = input$maxit)
      } else if (input$md == "K-means") {
        model <- kmeans(generated_data[, 1:input$dimensions], centers = input$clusters)
        generated_data$y <- factor(model$cluster)  # Update data with cluster labels
      } else if (input$md == "SVM") {
        model <- svm(y ~ ., data = generated_data, type = 'C-classification', kernel = 'radial')
      }
    trained_model(model)
        
    # Determine if t-SNE is needed based on dimensions
    if (input$dimensions > 3) {
      tsne_results <- perform_tsne(generated_data[, 1:input$dimensions], 3)  # Reduce to 3D for visualization
      colnames(tsne_results) <- c("X1", "X2", "X3")
      tsne_results$y <- generated_data$y  # Attach 'y' after t-SNE
      tsne_data(tsne_results)
    } else {
      # Set correct column names for the dimensions provided
      colnames(generated_data)[1:input$dimensions] <- paste0("X", 1:input$dimensions)
      tsne_data(generated_data)
    }

    
    output$nnPlot <- renderPlotly({
      req(tsne_data())
      plot_data <- tsne_data()
      
      if (input$dimensions == 3) {
        # 3D visualization for exactly three dimensions
        plot <- plot_ly(plot_data, x = ~X1, y = ~X2, z = ~X3, color = ~y, type = 'scatter3d', mode = 'markers') %>%
          layout(title = "3D Visualization of Classification",
                 scene = list(xaxis = list(title = 'Dimension 1'), yaxis = list(title = 'Dimension 2'), zaxis = list(title = 'Dimension 3')))
      } else if (input$dimensions == 2) {
        # 2D visualization for exactly two dimensions
        plot <- plot_ly(plot_data, x = ~X1, y = ~X2, color = ~y, type = 'scatter', mode = 'markers') %>%
          layout(title = "2D Visualization of Classification",
                 xaxis = list(title = 'Dimension 1'), yaxis = list(title = 'Dimension 2'))
      } else {
        # Default to 3D visualization if dimensions are more than three
        plot <- plot_ly(plot_data, x = ~X1, y = ~X2, z = ~X3, color = ~y, type = 'scatter3d', mode = 'markers') %>%
          layout(title = "3D Visualization of Multi-Class Classification",
                 scene = list(xaxis = list(title = 'Dimension 1'), yaxis = list(title = 'Dimension 2'), zaxis = list(title = 'Dimension 3')))
      }
      plot
    })
  })
}

shinyApp(ui = ui, server = server)

