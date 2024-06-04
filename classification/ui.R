library(shiny)

# Define UI for application
ui <- fluidPage(
  titlePanel("KNN and Logistic Regression Visualization"),
  
  sidebarLayout(
    sidebarPanel(
      radioButtons("dataSource", "Data Source:",
                   choices = list("Upload" = "upload", "Random" = "random")),
      conditionalPanel(
        condition = "input.dataSource == 'upload'",
        fileInput("file1", "Choose CSV File",
                  accept = c(
                    "text/csv",
                    "text/comma-separated-values,text/plain",
                    ".csv"))
      ),
      selectInput("model", "Choose Model:",
                  choices = list("KNN" = "KNN", "Logistic Regression" = "Logistic Regression")),
      conditionalPanel(
        condition = "input.model == 'KNN'",
        numericInput("k_value", "Number of Neighbors (k):", value = 3, min = 1, step = 1)
      ),
      actionButton("trainModel", "Train Model")
    ),
    
    mainPanel(
      tableOutput("dataSummary"),
      plotlyOutput("plot")
    )
  )
)
