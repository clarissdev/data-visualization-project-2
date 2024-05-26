library(shiny)
library(nnet)
library(ggplot2)

# Function to generate synthetic data with two classes
generate_data <- function(n = 200) {
  set.seed(123)
  x1 <- runif(n, -2, 2)
  x2 <- runif(n, -2, 2)
  y <- ifelse(x1^2 + x2^2 < 1, 1, 0)
  data.frame(x1 = x1, x2 = x2, y = as.factor(y))
}

# Define UI
ui <- fluidPage(
  titlePanel("Neural Network Decision Boundary Visualization"),
  sidebarLayout(
    sidebarPanel(
      sliderInput("size", "Number of Hidden Units", min = 1, max = 20, value = 5, step = 1),
      sliderInput("decay", "Weight Decay", min = 0, max = 1, value = 0, step = 0.1),
      sliderInput("maxit", "Maximum Iterations", min = 1, max = 500, value = 100, step = 1),
      actionButton("train", "Train Network")
    ),
    mainPanel(
      plotOutput("nnPlot")
    )
  )
)

# Define server logic
server <- function(input, output, session) {
  
  data <- reactive({
    generate_data()
  })
  
  trained_model <- reactiveVal(NULL)
  
  observeEvent(input$train, {
    # Train the neural network
    isolate({
      data <- data()
      size <- input$size
      decay <- input$decay
      maxit <- input$maxit
      nn_model <- nnet(y ~ x1 + x2, data = data, size = size, decay = decay, maxit = maxit)
      trained_model(nn_model)
    })
  })
  
  output$nnPlot <- renderPlot({
    req(trained_model())
    
    data <- data()
    nn_model <- trained_model()
    
    # Generate grid data for decision boundary
    grid <- expand.grid(x1 = seq(-2, 2, length.out = 100), x2 = seq(-2, 2, length.out = 100))
    grid$y <- predict(nn_model, newdata = grid, type = "class")
    
    # Plot decision boundary and data points
    ggplot(data, aes(x = x1, y = x2, color = y)) +
      geom_point() +
      geom_contour_filled(data = grid, aes(z = as.numeric(y)), alpha = 0.3) +
      scale_color_manual(values = c("blue", "red")) +
      labs(title = "Neural Network Decision Boundary Visualization", x = "X1", y = "X2", color = "Class") +
      theme_minimal()
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
